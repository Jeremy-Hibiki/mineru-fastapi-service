from functools import partial

import fire
import uvicorn

from . import app

fire.Fire(partial(uvicorn.run, app))
