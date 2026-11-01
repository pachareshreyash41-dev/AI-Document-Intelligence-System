"""Text cleaning utilities for noisy OCR output."""

import re
import unicodedata

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_HYPHEN_LINEBREAK = re.compile(r"(\w)-\n(\w)")
_MULTI_SPACE = re.compile(r"[ \t]+")
_MULTI_NEWLINE = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    """
    Normalize OCR text while keeping line structure (useful for extraction).

    Steps: unicode normalization, control-character removal, re-joining words
    split across lines by hyphens, whitespace collapsing.
    """
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL_CHARS.sub("", text)
    text = _HYPHEN_LINEBREAK.sub(r"\1\2", text)
    text = _MULTI_SPACE.sub(" ", text)

    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = _MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()


def normalize_for_model(text: str) -> str:
    """Lowercase, single-line version of the text for ML classification."""
    text = clean_text(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text