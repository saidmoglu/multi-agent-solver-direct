from __future__ import annotations

import os
import sys
import uuid
from datetime import datetime
from typing import Literal


def generate_run_id() -> str:
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    short_uuid = uuid.uuid4().hex[:8]
    return f"{timestamp}-{short_uuid}"


def ensure_outputs_dir(path: str = "outputs") -> str:
    os.makedirs(path, exist_ok=True)
    return path


class LogPrint:
    filename: str

    @staticmethod
    def init_file(filename: str) -> None:
        LogPrint.filename = filename
        if not os.path.exists(filename):
            with open(filename, "w") as f:
                pass

    @staticmethod
    def log(message: str, console_print: bool = True) -> None:
        with open(LogPrint.filename, "a") as f:
            f.write(message + "\n")
        if console_print:
            print(message)
        sys.stdout.flush()
