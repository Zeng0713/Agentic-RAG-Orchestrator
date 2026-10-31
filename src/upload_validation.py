"""Validation helpers for user supplied knowledge documents."""

import re
from pathlib import PurePath


ALLOWED_EXTENSIONS = {".md", ".txt"}
MAX_FILENAME_LENGTH = 120


def sanitize_filename(filename: str) -> str:
    """Return a safe display/storage filename without path components."""
    if not isinstance(filename, str):
        raise ValueError("文件名无效")

    leaf = filename.replace("\\", "/").split("/")[-1]
    leaf = re.sub(r"[\x00-\x1f\x7f]", "", leaf)
    leaf = re.sub(r"[^\w.() -]", "_", leaf, flags=re.UNICODE)
    leaf = leaf.strip(" .")[:MAX_FILENAME_LENGTH].strip(" .")
    if not leaf or leaf in {".", ".."}:
        raise ValueError("文件名无效")
    return leaf


def validate_upload(filename: str, content: bytes, max_size_bytes: int) -> tuple[str, str]:
    """Validate filename, size, and UTF-8 text content; return name and text."""
    safe_name = sanitize_filename(filename)
    extension = PurePath(safe_name).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise ValueError("仅支持 .md 和 .txt 文件")
    if not content:
        raise ValueError("上传文件不能为空")
    if len(content) > max_size_bytes:
        raise ValueError(f"文件超过大小限制（{max_size_bytes // (1024 * 1024)} MB）")
    if b"\x00" in content:
        raise ValueError("文件内容不是有效的文本文件")
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("文件必须使用 UTF-8 编码") from error
    if not text.strip():
        raise ValueError("文件内容不能为空")
    return safe_name, text
