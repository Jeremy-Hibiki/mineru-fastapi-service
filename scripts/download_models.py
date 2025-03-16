# ruff: noqa

import json
import os
from typing import Any

import requests
from huggingface_hub import snapshot_download


def download_json(url: str) -> Any:
    # 下载JSON文件
    response = requests.get(url)
    response.raise_for_status()  # 检查请求是否成功
    return response.json()


def download_and_modify_json(
    url: str,
    local_filename: str,
    modifications: dict[str, Any],
) -> None:
    if os.path.exists(local_filename):
        data = json.load(open(local_filename))
        config_version = data.get("config_version", "0.0.0")
        if config_version < "1.1.1":
            data = download_json(url)
    else:
        data = download_json(url)

    # 修改内容
    for key, value in modifications.items():
        data[key] = value

    # 保存修改后的内容
    with open(local_filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    mineru_patterns = [
        "models/Layout/LayoutLMv3/*",
        "models/Layout/YOLO/*",
        "models/MFD/YOLO/*",
        "models/MFR/unimernet_small_2501/*",
        "models/TabRec/TableMaster/*",
        "models/TabRec/StructEqTable/*",
    ]
    model_dir = snapshot_download(
        "opendatalab/PDF-Extract-Kit-1.0",
        local_dir="/opt/models/pdf-extract-kit/",
        allow_patterns=mineru_patterns,
    )

    layoutreader_pattern = [
        "*.json",
        "*.safetensors",
    ]
    layoutreader_model_dir = snapshot_download(
        "hantian/layoutreader",
        local_dir="/opt/models/layoutreader/",
        allow_patterns=layoutreader_pattern,
    )

    model_dir = model_dir + "/models"
    print(f"model_dir is: {model_dir}")
    print(f"layoutreader_model_dir is: {layoutreader_model_dir}")

    json_url = "https://github.com/opendatalab/MinerU/raw/master/magic-pdf.template.json"
    config_file_name = "magic-pdf.json"
    home_dir = os.path.expanduser("~")
    config_file = os.path.join(home_dir, config_file_name)

    json_mods = {
        "models-dir": model_dir,
        "layoutreader-model-dir": layoutreader_model_dir,
        "device-mode": "cuda",
    }

    download_and_modify_json(json_url, config_file, json_mods)
    print(f"The configuration file has been configured successfully, the path is: {config_file}")

    paddleocr_model_dir = snapshot_download(
        "opendatalab/MinerU",
        repo_type="space",
        local_dir="/opt/models/paddleocr/",
    )
    print(f"paddleocr model_dir is: {paddleocr_model_dir}")
