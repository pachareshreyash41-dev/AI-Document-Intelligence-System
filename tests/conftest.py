"""Shared fixtures for the test suite."""

import io
import sys
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont


# Ensure the project root is available for imports.
ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def make_text_image(
    text: str = "Invoice Number 12345",
    size: tuple[int, int] = (900, 200),
) -> Image.Image:
    """Create a white image containing readable black text."""
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("arial.ttf", 36)
    except OSError:
        font = ImageFont.load_default()

    draw.text((30, 60), text, fill="black", font=font)

    return image


class FakeUpload:
    """Minimal stand-in for Streamlit's UploadedFile."""

    def __init__(self, name: str, data: bytes):
        self.name = name
        self._data = data
        self.size = len(data)

    def getbuffer(self) -> bytes:
        return self._data

    def getvalue(self) -> bytes:
        return self._data

    def read(self, *args) -> bytes:
        return self._data

    def seek(self, position: int) -> None:
        """Support code that resets the upload's read position."""
        return None


@pytest.fixture
def text_image() -> Image.Image:
    """Provide a sample image for image-loading tests."""
    return make_text_image()


@pytest.fixture
def sample_png(tmp_path, text_image) -> Path:
    """Create a temporary PNG document."""
    image_path = tmp_path / "sample.png"
    text_image.save(image_path, format="PNG")

    return image_path


@pytest.fixture
def sample_pdf(tmp_path) -> Path:
    """Create a two-page PDF document."""
    import fitz

    pdf_path = tmp_path / "sample.pdf"

    document = fitz.open()

    try:
        for page_number in range(1, 3):
            page = document.new_page()
            page.insert_text(
                (72, 72),
                f"Page {page_number} Invoice Total 100.00",
            )

        document.save(str(pdf_path))
    finally:
        document.close()

    return pdf_path


@pytest.fixture
def fake_upload(text_image) -> FakeUpload:
    """Provide a fake PNG upload."""
    buffer = io.BytesIO()
    text_image.save(buffer, format="PNG")

    return FakeUpload(
        name="sample.png",
        data=buffer.getvalue(),
    )