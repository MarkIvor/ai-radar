"""Парсинг загруженных файлов (.txt / .docx / .pdf).

Извлекает чистый текст из файлов в памяти, без сохранения на диск.
Ограничивает размер во избежание переполнения памяти.
"""

from __future__ import annotations

import io
from typing import BinaryIO

MAX_FILE_BYTES = 50 * 1024 * 1024  # 50 МБ


class FileParseError(Exception):
    """Ошибка парсинга файла."""


def parse_txt(raw: bytes, *, filename: str = "") -> str:
    """Парсинг .txt с авто-detect кодировки (UTF-8 / Windows-1251)."""
    if not raw:
        return ""
    for enc in ("utf-8", "cp1251", "latin-1"):
        try:
            return raw.decode(enc).strip()
        except UnicodeDecodeError:
            continue
    raise FileParseError(f"Не удалось декодировать {filename!r}")


def parse_docx(raw: bytes, *, filename: str = "") -> str:
    """Парсинг .docx через python-docx."""
    try:
        from docx import Document
    except ImportError as exc:  # pragma: no cover
        raise FileParseError("python-docx не установлен") from exc
    try:
        doc = Document(io.BytesIO(raw))
    except Exception as exc:
        raise FileParseError(f"Не удалось прочитать docx {filename!r}: {exc}") from exc
    parts: list[str] = []
    for p in doc.paragraphs:
        text = p.text.strip()
        if text:
            parts.append(text)
    return "\n".join(parts).strip()


def parse_pdf(raw: bytes, *, filename: str = "") -> str:
    """Парсинг .pdf через pypdf."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover
        raise FileParseError("pypdf не установлен") from exc
    try:
        reader = PdfReader(io.BytesIO(raw))
    except Exception as exc:
        raise FileParseError(f"Не удалось прочитать pdf {filename!r}: {exc}") from exc
    parts: list[str] = []
    for page in reader.pages:
        try:
            txt = page.extract_text() or ""
        except Exception:
            continue
        txt = txt.strip()
        if txt:
            parts.append(txt)
    return "\n".join(parts).strip()


def detect_mime(filename: str) -> str:
    """Грубое определение MIME по расширению."""
    name = filename.lower()
    if name.endswith(".txt"):
        return "text/plain"
    if name.endswith(".docx"):
        return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    if name.endswith(".pdf"):
        return "application/pdf"
    return "application/octet-stream"


def parse_file(*, filename: str, content: bytes) -> str:
    """Диспетчер парсеров по расширению файла."""
    if len(content) > MAX_FILE_BYTES:
        raise FileParseError(
            f"Файл слишком большой: {len(content)} байт. Максимум {MAX_FILE_BYTES}."
        )
    name = filename.lower()
    if name.endswith(".txt"):
        return parse_txt(content, filename=filename)
    if name.endswith(".docx"):
        return parse_docx(content, filename=filename)
    if name.endswith(".pdf"):
        return parse_pdf(content, filename=filename)
    raise FileParseError(f"Неподдерживаемое расширение файла: {filename!r}")


def parse_stream(*, filename: str, stream: BinaryIO) -> str:
    """Прочитать содержимое из потока и распарсить."""
    content = stream.read()
    if isinstance(content, str):
        content = content.encode("utf-8")
    return parse_file(filename=filename, content=content)
