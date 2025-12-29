"""Robust document loaders for various file formats."""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from core.document import Document

LOGGER = logging.getLogger(__name__)

DEFAULT_TEXT_EXTENSIONS = {".txt", ".md", ".rst", ".json", ".csv"}
PDF_EXTENSIONS = {".pdf"}


def _extract_text_pymupdf(path: Path) -> Optional[str]:
    """Extract text using PyMuPDF (fitz) - best for complex PDFs."""
    try:
        import fitz  # pymupdf

        doc = fitz.open(str(path))
        text_parts = []
        for page_num, page in enumerate(doc):
            text = page.get_text("text")
            if text.strip():
                text_parts.append(f"[Page {page_num + 1}]\n{text}")
        doc.close()
        return "\n\n".join(text_parts)
    except ImportError:
        return None
    except Exception as e:
        LOGGER.debug("PyMuPDF failed for %s: %s", path, e)
        return None


def _extract_text_pdfplumber(path: Path) -> Optional[str]:
    """Extract text using pdfplumber - good for tables and structured PDFs."""
    try:
        import pdfplumber

        text_parts = []
        with pdfplumber.open(str(path)) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text and text.strip():
                    text_parts.append(f"[Page {page_num + 1}]\n{text}")

                # Also try to extract tables
                tables = page.extract_tables()
                for table in tables:
                    if table:
                        table_text = "\n".join(
                            " | ".join(str(cell or "") for cell in row)
                            for row in table
                            if row
                        )
                        if table_text.strip():
                            text_parts.append(f"[Table]\n{table_text}")

        return "\n\n".join(text_parts)
    except ImportError:
        return None
    except Exception as e:
        LOGGER.debug("pdfplumber failed for %s: %s", path, e)
        return None


def _extract_text_pypdf2(path: Path) -> Optional[str]:
    """Extract text using PyPDF2 - basic fallback."""
    try:
        from PyPDF2 import PdfReader

        reader = PdfReader(str(path))
        text_parts = []
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                text_parts.append(f"[Page {page_num + 1}]\n{text}")
        return "\n\n".join(text_parts)
    except ImportError:
        return None
    except Exception as e:
        LOGGER.debug("PyPDF2 failed for %s: %s", path, e)
        return None


def extract_pdf_text(path: Path) -> Optional[str]:
    """
    Extract text from PDF using the best available library.
    Tries PyMuPDF first (best quality), then pdfplumber, then PyPDF2.
    """
    # Try each extractor in order of quality
    extractors = [
        ("PyMuPDF", _extract_text_pymupdf),
        ("pdfplumber", _extract_text_pdfplumber),
        ("PyPDF2", _extract_text_pypdf2),
    ]

    for name, extractor in extractors:
        text = extractor(path)
        if text and len(text.strip()) > 50:  # Minimum viable extraction
            LOGGER.info("Extracted %d chars from %s using %s", len(text), path.name, name)
            return text

    LOGGER.warning("Could not extract text from PDF: %s", path)
    return None


def load_text_file(path: Path) -> Optional[str]:
    """Load a plain text file."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            return path.read_text(encoding="latin-1")
        except Exception as e:
            LOGGER.warning("Could not read %s: %s", path, e)
            return None
    except Exception as e:
        LOGGER.warning("Could not read %s: %s", path, e)
        return None


def load_single_document(path: Path) -> Optional[Document]:
    """Load a single document from a file path."""
    if not path.exists() or not path.is_file():
        return None

    suffix = path.suffix.lower()

    if suffix in PDF_EXTENSIONS:
        text = extract_pdf_text(path)
    elif suffix in DEFAULT_TEXT_EXTENSIONS:
        text = load_text_file(path)
    else:
        LOGGER.debug("Unsupported file type: %s", path)
        return None

    if not text or len(text.strip()) < 10:
        LOGGER.warning("Empty or too short content from: %s", path)
        return None

    return Document(
        id=path.stem,
        text=text,
        metadata={
            "source": str(path.resolve()),
            "filename": path.name,
            "extension": suffix,
        },
    )


def _collect_paths(directory: str, extensions: Sequence[str]) -> List[Path]:
    """Recursively collect all files with given extensions."""
    root = Path(directory)
    if not root.exists():
        return []

    matches: List[Path] = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in extensions:
            matches.append(path)

    return sorted(matches, key=lambda p: p.name)


def load_documents_from_paths(paths: Iterable[str], max_workers: int = 4) -> List[Document]:
    """Load documents from a list of file paths in parallel."""
    path_list = [Path(p) for p in paths if p]
    documents: List[Document] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {
            executor.submit(load_single_document, path): path for path in path_list
        }
        for future in as_completed(future_to_path):
            path = future_to_path[future]
            try:
                doc = future.result()
                if doc:
                    documents.append(doc)
                    LOGGER.info("Loaded: %s (%d chars)", path.name, len(doc.text))
            except Exception as e:
                LOGGER.error("Failed to load %s: %s", path, e)

    return documents


def load_documents_from_directory(
    directory: str,
    extensions: Optional[Sequence[str]] = None,
    max_workers: int = 4,
) -> List[Document]:
    """Load all supported documents from a directory."""
    if extensions is None:
        extensions = list(DEFAULT_TEXT_EXTENSIONS | PDF_EXTENSIONS)

    paths = _collect_paths(directory, extensions)
    if not paths:
        LOGGER.warning("No supported files found in %s", directory)
        return []

    LOGGER.info("Found %d files to process in %s", len(paths), directory)
    return load_documents_from_paths([str(p) for p in paths], max_workers)
