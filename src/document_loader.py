import re
from pathlib import Path

from langchain_core.documents import Document


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