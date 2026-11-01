"""
Document loading utilities.

- save_upload():   store an uploaded file safely on disk
- load_document(): turn a PDF or image file into a list of PIL images (one per page)
"""

import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Union

import fitz  # PyMuPDF
from PIL import Image, ImageSequence

PDF_EXTENSIONS = {".pdf"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
SUPPORTED_EXTENSIONS = PDF_EXTENSIONS | IMAGE_EXTENSIONS

PDF_RENDER_DPI = 200


def _sanitize_filename(filename: str) -> tuple:
    """Return a filesystem-safe (stem, extension) pair."""
    path = Path(filename)
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", path.stem).strip("_") or "document"
    return stem, path.suffix.lower()


def save_upload(uploaded_file, upload_dir: Union[str, Path]) -> Path:
    """
    Save a Streamlit UploadedFile (or any object with `.name` and `.getbuffer()`)
    into `upload_dir` using a unique, sanitized filename.
    """
    upload_dir = Path(upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    stem, ext = _sanitize_filename(uploaded_file.name)
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{ext}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique = uuid.uuid4().hex[:6]
    destination = upload_dir / f"{stem}_{timestamp}_{unique}{ext}"
    destination.write_bytes(uploaded_file.getbuffer())
    return destination


def load_document(path: Union[str, Path]) -> List[Image.Image]:
    """Load a PDF or image file and return its pages as RGB PIL images."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    ext = path.suffix.lower()
    if ext in PDF_EXTENSIONS:
        return _load_pdf(path)
    if ext in IMAGE_EXTENSIONS:
        return _load_image(path)
    raise ValueError(f"Unsupported file type: '{ext}'")


def _load_pdf(path: Path, dpi: int = PDF_RENDER_DPI) -> List[Image.Image]:
    pages: List[Image.Image] = []
    zoom = dpi / 72  # PDF native resolution is 72 DPI
    matrix = fitz.Matrix(zoom, zoom)

    with fitz.open(path) as doc:
        if doc.page_count == 0:
            raise ValueError("The PDF has no pages.")
        for page in doc:
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            pages.append(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
    return pages


def _load_image(path: Path) -> List[Image.Image]:
    # ImageSequence handles multi-page TIFFs; single images yield one frame.
    with Image.open(path) as img:
        return [frame.convert("RGB") for frame in ImageSequence.Iterator(img)]