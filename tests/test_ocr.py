"""Tests for src/ocr_engine.py

These tests need the Tesseract binary. They are skipped automatically if it
is not installed, so the rest of the suite still runs.
"""

import shutil

import pytest

from src.ocr_engine import OCREngine

pytestmark = pytest.mark.skipif(
    shutil.which("tesseract") is None,
    reason="Tesseract is not installed or not on PATH",
)


@pytest.fixture(scope="module")
def engine():
    return OCREngine(language="eng")


def test_extract_text_reads_clean_image(engine, text_image):
    text = engine.extract_text([text_image])

    assert isinstance(text, str)
    assert "invoice" in text.lower()
    assert "12345" in text


def test_extract_text_joins_multiple_pages(engine, text_image):
    text = engine.extract_text([text_image, text_image])

    assert text.lower().count("invoice") >= 2


def test_extract_text_empty_list_returns_empty_string(engine):
    assert engine.extract_text([]) == ""


def test_extract_text_blank_page_returns_blank(engine):
    from PIL import Image

    blank = Image.new("RGB", (400, 200), "white")

    assert engine.extract_text([blank]).strip() == ""