import re
from pathlib import Path

from langchain_core.documents import Document


def parse_document_text(filename, content, metadata=None, chunk_size=1500):
    """Parse Markdown by H2 headings and split plain text into bounded chunks."""
    if chunk_size < 1:
        raise ValueError("chunk_size 必须大于 0")
    extension = Path(filename).suffix.lower()
    base_metadata = dict(metadata or {})
    base_metadata.setdefault("source", Path(filename).name)
    base_metadata["document_type"] = extension.lstrip(".")

    if extension == ".md":
        sections = re.split(r"(?m)(?=^## )", content)
        chunks = []
        for section in sections:
            section = section.strip()
            if not section:
                continue
            if len(section) <= chunk_size:
                chunks.append(section)
                continue
            heading, separator, body = section.partition("\n")
            if not separator:
                heading, body = "", section
            paragraphs = [part.strip() for part in re.split(r"\n\s*\n", body) if part.strip()]
            current = heading
            for paragraph in paragraphs or [body]:
                while len(paragraph) > chunk_size:
                    if current and len(current) + 2 < chunk_size:
                        content_size = chunk_size - len(current) - 2
                        chunks.append(f"{current}\n\n{paragraph[:content_size]}")
                        paragraph = paragraph[content_size:]
                    else:
                        if current:
                            chunks.append(current)
                            current = ""
                        chunks.append(paragraph[:chunk_size])
                        paragraph = paragraph[chunk_size:]
                candidate = f"{current}\n\n{paragraph}".strip()
                if current and len(candidate) > chunk_size:
                    chunks.append(current)
                    current = paragraph
                else:
                    current = candidate
            if current:
                chunks.append(current)
    elif extension == ".txt":
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", content) if part.strip()]
        chunks = []
        current = ""
        for paragraph in paragraphs:
            while len(paragraph) > chunk_size:
                if current:
                    chunks.append(current)
                    current = ""
                chunks.append(paragraph[:chunk_size])
                paragraph = paragraph[chunk_size:]
            candidate = f"{current}\n\n{paragraph}".strip()
            if current and len(candidate) > chunk_size:
                chunks.append(current)
                current = paragraph
            else:
                current = candidate
        if current:
            chunks.append(current)
    else:
        raise ValueError("仅支持 .md 和 .txt 文件")

    return [
        Document(page_content=chunk, metadata={**base_metadata, "chunk_index": index})
        for index, chunk in enumerate(chunks)
    ]


def load_documents(documents_dir="data/documents"):
    """
    读取知识库中的 Markdown 文件，
    按照二级标题 ## 切分，
    最后返回 Document 列表。
    """

    documents_dir = Path(documents_dir)

    chunks = []

    for file_path in documents_dir.glob("*.md"):
        content = file_path.read_text(encoding="utf-8")

        sections = re.split(
            r"(?m)(?=^## )",
            content,
        )

        for section in sections:
            section = section.strip()

            if not section:
                continue

            chunks.append(
                Document(
                    page_content=section,
                    metadata={
                        "source": str(file_path),
                        "document_type": "markdown",
                    },
                )
            )

    return chunks
