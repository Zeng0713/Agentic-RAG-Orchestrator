import pytest

from src.upload_validation import sanitize_filename, validate_upload


def test_sanitize_filename_removes_windows_path_components():
    assert sanitize_filename(r"..\private\notes.md") == "notes.md"


@pytest.mark.parametrize("filename", ["notes.pdf", "notes.exe", "folder/"])
def test_upload_rejects_unsupported_or_empty_filename(filename):
    with pytest.raises(ValueError):
        validate_upload(filename, b"content", 1024)


def test_upload_rejects_oversize_empty_and_invalid_text():
    with pytest.raises(ValueError, match="大小限制"):
        validate_upload("notes.txt", b"12345", 4)
    with pytest.raises(ValueError, match="不能为空"):
        validate_upload("notes.md", b"", 1024)
    with pytest.raises(ValueError, match="UTF-8"):
        validate_upload("notes.txt", b"\xff", 1024)


def test_upload_accepts_markdown_and_text():
    assert validate_upload("资料.md", b"# title", 1024) == ("资料.md", "# title")
    assert validate_upload("notes.txt", b"plain text", 1024) == ("notes.txt", "plain text")
