from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


@dataclass
class DomainChunk:
    chunk_id: str
    text: str
    order: int
    word_count: int


@dataclass
class DomainDocument:
    doc_id: str
    title: str
    category: str
    path: str
    text: str
    chunks: list[DomainChunk]


def slugify(text: str) -> str:
    normalized = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE).strip().lower()
    normalized = re.sub(r"[-\s]+", "_", normalized)
    return normalized or "document"


def normalize_text(text: str) -> str:
    text = text.replace("\x0c", "\n")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _resolve_pdftotext_executable() -> str:
    configured_path = os.environ.get("OPERA_PDFTOTEXT") or os.environ.get("PDFTOTEXT_PATH")
    candidates = [
        configured_path,
        shutil.which("pdftotext"),
    ]
    for candidate in candidates:
        if candidate and os.path.exists(candidate) and os.access(candidate, os.X_OK):
            return candidate

    raise FileNotFoundError(
        "pdftotext executable was not found. Install Poppler, add pdftotext to PATH, "
        "or set OPERA_PDFTOTEXT to the full executable path."
    )


def _extract_pdf_text(path: str) -> str:
    pdftotext = _resolve_pdftotext_executable()
    result = subprocess.run(
        [pdftotext, path, "-"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return normalize_text(result.stdout)


def extract_document_text(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return _extract_pdf_text(path)

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return normalize_text(f.read())


def chunk_text(
    text: str,
    chunk_words: int = 500,
    overlap_words: int = 75,
) -> list[DomainChunk]:
    words = text.split()
    if not words:
        return []

    chunks: list[DomainChunk] = []
    start = 0
    order = 0
    total = len(words)
    stride = max(1, chunk_words - overlap_words)

    while start < total:
        end = min(total, start + chunk_words)
        chunk_words_list = words[start:end]
        chunk_text_value = " ".join(chunk_words_list).strip()
        if chunk_text_value:
            chunk_hash = hashlib.md5(f"{order}:{chunk_text_value[:256]}".encode("utf-8")).hexdigest()[:10]
            chunks.append(
                DomainChunk(
                    chunk_id=f"chunk_{order}_{chunk_hash}",
                    text=chunk_text_value,
                    order=order,
                    word_count=len(chunk_words_list),
                )
            )
        if end >= total:
            break
        start += stride
        order += 1

    return chunks


def discover_domain_documents(
    corpus_root: str,
    chunk_words: int = 500,
    overlap_words: int = 75,
) -> list[DomainDocument]:
    root = Path(corpus_root)
    if not root.exists():
        return []

    documents: list[DomainDocument] = []
    for file_path in sorted(root.rglob("*")):
        if not file_path.is_file():
            continue
        if file_path.name.startswith("."):
            continue
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        relative_parent = file_path.parent.relative_to(root)
        category = str(relative_parent) if str(relative_parent) != "." else "uncategorized"
        title = file_path.stem
        text = extract_document_text(str(file_path))
        if not text:
            continue

        doc_slug = slugify(f"{category}_{title}")
        chunks = chunk_text(text, chunk_words=chunk_words, overlap_words=overlap_words)
        documents.append(
            DomainDocument(
                doc_id=doc_slug,
                title=title,
                category=category,
                path=file_path.relative_to(root).as_posix(),
                text=text,
                chunks=chunks,
            )
        )

    return documents


def iter_document_snippets(documents: Iterable[DomainDocument], max_chars: int = 700) -> list[dict]:
    snippets = []
    for doc in documents:
        preview = doc.text[:max_chars].strip()
        snippets.append(
            {
                "doc_id": doc.doc_id,
                "title": doc.title,
                "category": doc.category,
                "preview": preview,
            }
        )
    return snippets
