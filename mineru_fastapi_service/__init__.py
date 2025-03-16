import gc
import json
import os
from base64 import b64encode
from contextlib import asynccontextmanager
from glob import glob
from io import StringIO
from pathlib import Path
from typing import Literal, cast

import magic_pdf.model as model_config
import torch
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from magic_pdf.config.enums import SupportedPdfParseMethod
from magic_pdf.data.data_reader_writer import DataWriter, FileBasedDataWriter
from magic_pdf.data.data_reader_writer.s3 import S3DataReader, S3DataWriter
from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.libs.config_reader import get_bucket_name, get_s3_config
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
from magic_pdf.operators.models import InferenceResult
from magic_pdf.operators.pipes import PipeResult

model_config.__use_inside_model__ = True

load_dotenv()

MINERU_SERVICE_OUTPUT_DIR = os.getenv("MINERU_SERVICE_OUTPUT_DIR", "./output")
MINERU_SERVICE_OUTPUT_RETENTION_HOURS = int(os.getenv("MINERU_SERVICE_OUTPUT_RETENTION_HOURS", "6"))
MINERU_SERVICE_CLEANUP_INTERVAL_MINUTES = int(os.getenv("MINERU_SERVICE_CLEANUP_INTERVAL_MINUTES", "60"))

Path(MINERU_SERVICE_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    from magic_pdf.model.doc_analyze_by_custom_model import ModelSingleton

    model_manager = ModelSingleton()
    model_manager.get_model(True, False)
    model_manager.get_model(False, False)
    yield


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=MINERU_SERVICE_OUTPUT_DIR), name="static")


class MemoryDataWriter(DataWriter):
    def __init__(self):
        self.buffer = StringIO()

    def write(self, path: str, data: bytes | str) -> None:
        if isinstance(data, str):
            self.buffer.write(data)
        else:
            self.buffer.write(data.decode("utf-8"))

    def write_string(self, path: str, data: str) -> None:
        self.buffer.write(data)

    def get_value(self) -> str:
        return self.buffer.getvalue()

    def close(self):
        self.buffer.close()


def init_writers(
    pdf_path: str | None,
    pdf_file: UploadFile | None,
    output_path: str,
    output_image_path: str,
) -> tuple[
    S3DataWriter | FileBasedDataWriter,
    S3DataWriter | FileBasedDataWriter,
    bytes,
]:
    """
    Initialize writers based on path type

    Args:
        pdf_path: PDF file path (local path or S3 path)
        pdf_file: Uploaded PDF file object
        output_path: Output directory path
        output_image_path: Image output directory path

    Returns:
        Tuple[writer, image_writer, pdf_bytes]: Returns initialized writer tuple and PDF
        file content
    """
    if pdf_path:
        is_s3_path = pdf_path.startswith("s3://")
        if is_s3_path:
            bucket = get_bucket_name(pdf_path)
            ak, sk, endpoint = get_s3_config(bucket)

            writer = S3DataWriter(output_path, bucket=bucket, ak=ak, sk=sk, endpoint_url=endpoint)
            image_writer = S3DataWriter(output_image_path, bucket=bucket, ak=ak, sk=sk, endpoint_url=endpoint)
            # 临时创建reader读取文件内容
            temp_reader = S3DataReader("", bucket=bucket, ak=ak, sk=sk, endpoint_url=endpoint)
            pdf_bytes = temp_reader.read(pdf_path)
        else:
            writer = FileBasedDataWriter(output_path)
            image_writer = FileBasedDataWriter(output_image_path)
            os.makedirs(output_image_path, exist_ok=True)
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()
    elif pdf_file:
        # 处理上传的文件
        pdf_bytes = pdf_file.file.read()
        writer = FileBasedDataWriter(output_path)
        image_writer = FileBasedDataWriter(output_image_path)
        os.makedirs(output_image_path, exist_ok=True)
    else:
        raise ValueError("Must provide either pdf_file or pdf_path")

    return writer, image_writer, pdf_bytes


def process_pdf(
    pdf_bytes: bytes,
    parse_method: Literal["ocr", "txt", "auto"],
    image_writer: S3DataWriter | FileBasedDataWriter,
) -> tuple[InferenceResult, PipeResult]:
    """
    Process PDF file content

    Args:
        pdf_bytes: Binary content of PDF file
        parse_method: Parse method ('ocr', 'txt', 'auto')
        image_writer: Image writer

    Returns:
        Tuple[InferenceResult, PipeResult]: Returns inference result and pipeline result
    """
    ds = PymuDocDataset(pdf_bytes)

    if parse_method == "ocr" or (parse_method == "auto" and ds.classify() == SupportedPdfParseMethod.OCR):
        infer_result = cast(InferenceResult, ds.apply(doc_analyze, ocr=True))
        pipe_result = infer_result.pipe_ocr_mode(image_writer)
    elif parse_method == "txt" or (parse_method == "auto" and ds.classify() == SupportedPdfParseMethod.TXT):
        infer_result = cast(InferenceResult, ds.apply(doc_analyze, ocr=False))
        pipe_result = infer_result.pipe_txt_mode(image_writer)
    else:
        raise ValueError(f"Invalid parse method: {parse_method}")

    return infer_result, pipe_result


def encode_image(image_path: str) -> str:
    """Encode image using base64"""
    with open(image_path, "rb") as f:
        return b64encode(f.read()).decode()


@app.post(
    "/pdf_parse",
    tags=["projects"],
    summary="Parse PDF files (supports local files and S3)",
)
async def pdf_parse(
    pdf_file: UploadFile | None = None,
    pdf_path: str | None = None,
    parse_method: Literal["ocr", "txt", "auto"] = "auto",
    is_json_md_dump: bool = False,
    output_dir: str = MINERU_SERVICE_OUTPUT_DIR,
    return_layout: bool = False,
    return_info: bool = False,
    return_content_list: bool = False,
    return_images: bool = False,
):
    """
    Execute the process of converting PDF to JSON and MD, outputting MD and JSON files
    to the specified directory.

    Args:
        pdf_file: The PDF file to be parsed. Must not be specified together with
            `pdf_path`
        pdf_path: The path to the PDF file to be parsed. Must not be specified together
            with `pdf_file`
        parse_method: Parsing method, can be auto, ocr, or txt. Default is auto. If
            results are not satisfactory, try ocr
        is_json_md_dump: Whether to write parsed data to .json and .md files. Default
            to False. Different stages of data will be written to different .json files
            (3 in total), md content will be saved to .md file
        output_dir: Output directory for results. A folder named after the PDF file
            will be created to store all results
        return_layout: Whether to return parsed PDF layout. Default to False
        return_info: Whether to return parsed PDF info. Default to False
        return_content_list: Whether to return parsed PDF content list. Default to False
    """
    try:
        if pdf_file is None and pdf_path is not None:
            pdf_name = Path(pdf_path).stem
        elif pdf_file is not None and pdf_path is None:
            pdf_name = Path(cast(str, pdf_file.filename)).stem
        else:
            return JSONResponse(
                content={"error": "Must provide either pdf_file or pdf_path"},
                status_code=400,
            )
        output_path = f"{output_dir}/{pdf_name}"
        output_image_path = f"{output_path}/images"

        # Initialize readers/writers and get PDF content
        writer, image_writer, pdf_bytes = init_writers(
            pdf_path=pdf_path,
            pdf_file=pdf_file,
            output_path=output_path,
            output_image_path=output_image_path,
        )

        # Process PDF
        infer_result, pipe_result = process_pdf(pdf_bytes, parse_method, image_writer)

        # Use MemoryDataWriter to get results
        content_list_writer = MemoryDataWriter()
        md_content_writer = MemoryDataWriter()
        middle_json_writer = MemoryDataWriter()

        # Use PipeResult's dump method to get data
        pipe_result.dump_content_list(content_list_writer, "", "images")
        pipe_result.dump_md(md_content_writer, "", "images")
        pipe_result.dump_middle_json(middle_json_writer, "")

        # Get content
        content_list = json.loads(content_list_writer.get_value())
        md_content = md_content_writer.get_value()
        middle_json = json.loads(middle_json_writer.get_value())
        model_json = infer_result.get_infer_res()

        # If results need to be saved
        if is_json_md_dump:
            writer.write_string(f"{pdf_name}_content_list.json", content_list_writer.get_value())
            writer.write_string(f"{pdf_name}.md", md_content)
            writer.write_string(f"{pdf_name}_middle.json", middle_json_writer.get_value())
            writer.write_string(
                f"{pdf_name}_model.json",
                json.dumps(model_json, indent=4, ensure_ascii=False),
            )
            # Save visualization results
            pipe_result.draw_layout(os.path.join(output_path, f"{pdf_name}_layout.pdf"))
            pipe_result.draw_span(os.path.join(output_path, f"{pdf_name}_spans.pdf"))
            pipe_result.draw_line_sort(os.path.join(output_path, f"{pdf_name}_line_sort.pdf"))
            infer_result.draw_model(os.path.join(output_path, f"{pdf_name}_model.pdf"))

        # Build return data
        data = {}
        if return_layout:
            data["layout"] = model_json
        if return_info:
            data["info"] = middle_json
        if return_content_list:
            data["content_list"] = content_list
        if return_images:
            image_paths = glob(f"{output_image_path}/*.jpg")
            data["images"] = {
                os.path.basename(image_path): f"data:image/jpeg;base64,{encode_image(image_path)}"
                for image_path in image_paths
            }
        data["md_content"] = md_content  # md_content is always returned

        # Clean up memory writers
        content_list_writer.close()
        md_content_writer.close()
        middle_json_writer.close()

        return JSONResponse(data, status_code=200)

    except Exception as e:
        logger.exception(e)
        return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        gc.collect()


if __name__ == "__main__":
    from functools import partial

    import fire

    fire.Fire(partial(uvicorn.run, app))
