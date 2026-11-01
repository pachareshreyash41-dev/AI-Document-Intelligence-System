"""Tests for src/document_loader.py."""

from pathlib import Path

import pytest
from PIL import Image

from src.document_loader import load_document, save_upload


# ---------------------------------------------------------------------------
# save_upload tests
# ---------------------------------------------------------------------------

def test_save_upload_writes_file(tmp_path, fake_upload):
    """Verify that an uploaded file is saved correctly."""
    saved = Path(save_upload(fake_upload, tmp_path))

    assert saved.exists()
    assert saved.is_file()
    assert saved.parent.resolve() == tmp_path.resolve()
    assert saved.suffix.lower() == ".png"
    assert saved.read_bytes() == fake_upload.getbuffer()


def test_save_upload_does_not_overwrite_existing_file(tmp_path, fake_upload):
    """Verify that saving the same upload twice creates different files."""
    first = Path(save_upload(fake_upload, tmp_path))
    second = Path(save_upload(fake_upload, tmp_path))

    assert first != second
    assert first.exists()
    assert second.exists()
    assert first.read_bytes() == fake_upload.getbuffer()
    assert second.read_bytes() == fake_upload.getbuffer()


# ---------------------------------------------------------------------------
# load_document tests
# ---------------------------------------------------------------------------

def test_load_image_returns_single_page(sample_png):
    """Verify that an image document returns exactly one page."""
    pages = load_document(sample_png)

    assert isinstance(pages, list)
    assert len(pages) == 1
    assert isinstance(pages[0], Image.Image)


def test_load_pdf_returns_one_image_per_page(sample_pdf):
    """Verify that every PDF page is converted to a PIL image."""
    pages = load_document(sample_pdf)

    assert isinstance(pages, list)
    assert len(pages) == 2
    assert all(isinstance(page, Image.Image) for page in pages)


@pytest.mark.parametrize("extension", [".jpg", ".jpeg", ".bmp", ".tiff"])
def test_load_supported_image_formats(tmp_path, text_image, extension):
    """Verify that supported image formats can be loaded."""
    image_path = tmp_path / f"scan{extension}"
    text_image.save(image_path)

    pages = load_document(image_path)

    assert isinstance(pages, list)
    assert len(pages) == 1
    assert isinstance(pages[0], Image.Image)


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------

def test_load_missing_file_raises(tmp_path):
    """Verify that loading a nonexistent document raises FileNotFoundError."""
    missing_path = tmp_path / "does_not_exist.pdf"

    with pytest.raises(FileNotFoundError):
        load_document(missing_path)


def test_load_unsupported_extension_raises(tmp_path):
    """Verify that unsupported document formats raise ValueError."""
    unsupported_path = tmp_path / "notes.xyz"
    unsupported_path.write_text("not a document", encoding="utf-8")

    with pytest.raises(ValueError):
        load_document(unsupported_path)