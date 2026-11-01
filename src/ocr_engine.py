"""
OCR engine built on Tesseract with optional OpenCV image preprocessing.

Image cleanup pipeline:
RGB -> Grayscale -> Denoising -> Otsu Binarization
"""

import os
from pathlib import Path
from typing import Iterable, Optional, Union

import cv2
import numpy as np
import pytesseract
from PIL import Image


class OCREngine:
    """Extract text from images using Tesseract OCR."""

    DEFAULT_TESSERACT_PATH = (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    )

    def __init__(
        self,
        language: str = "eng",
        tesseract_cmd: Optional[str] = None,
        preprocess: bool = True,
        config: str = "--oem 3 --psm 3",
    ) -> None:
        self.language = language
        self.preprocess = preprocess
        self.config = config

        # Priority:
        # 1. Explicit constructor argument
        # 2. TESSERACT_CMD environment variable
        # 3. Default Windows installation path
        # 4. Existing executable available on PATH
        configured_path = (
            tesseract_cmd
            or os.getenv("TESSERACT_CMD")
            or self.DEFAULT_TESSERACT_PATH
        )

        configured_file = Path(configured_path)

        if configured_file.is_file():
            pytesseract.pytesseract.tesseract_cmd = str(configured_file)
        elif tesseract_cmd or os.getenv("TESSERACT_CMD"):
            # Preserve an explicitly configured path so that
            # pytesseract reports a useful error if it is incorrect.
            pytesseract.pytesseract.tesseract_cmd = configured_path
        else:
            # Fall back to executable discovery through Windows PATH.
            pytesseract.pytesseract.tesseract_cmd = "tesseract"

    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """Convert a PIL image to a denoised, binarized grayscale array."""
        array = np.array(image.convert("RGB"))
        gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
        gray = cv2.fastNlMeansDenoising(gray, None, h=10)

        _, binary = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )

        return binary

    def extract_text_from_image(self, image: Image.Image) -> str:
        """Extract text from a single PIL image."""
        source = self.preprocess_image(image) if self.preprocess else image

        try:
            return pytesseract.image_to_string(
                source,
                lang=self.language,
                config=self.config,
            )
        except pytesseract.TesseractNotFoundError as exc:
            executable = pytesseract.pytesseract.tesseract_cmd

            raise RuntimeError(
                "Tesseract OCR executable could not be found. "
                f"Configured path: {executable!r}. "
                "Check that Tesseract is installed and that TESSERACT_CMD "
                "points to the correct tesseract.exe file."
            ) from exc
        except pytesseract.TesseractError as exc:
            raise RuntimeError(
                f"Tesseract OCR failed. Check that language "
                f"{self.language!r} is installed and that the OCR "
                f"configuration {self.config!r} is valid. Details: {exc}"
            ) from exc

    def extract_text(
        self,
        pages: Union[Image.Image, Iterable[Image.Image]],
    ) -> str:
        """Extract text from one image or multiple page images."""
        if isinstance(pages, Image.Image):
            pages = [pages]

        texts = [
            self.extract_text_from_image(page).strip()
            for page in pages
        ]

        return "\n\n".join(text for text in texts if text)