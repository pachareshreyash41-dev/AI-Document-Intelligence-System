"""
Information extraction from OCR text.

Uses regular expressions for common fields (emails, phones, dates, amounts, URLs),
document-type-specific rules (invoice / resume), and spaCy named entities
when the `en_core_web_sm` model is installed.
"""

import re
from typing import Dict, List, Optional

_MONTHS = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|"
    r"Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
)
_CURRENCY = r"(?:₹|Rs\.?|INR|\$|USD|€|EUR|£|GBP)"
_AMOUNT = rf"{_CURRENCY}\s?\d[\d,]*(?:\.\d{{1,2}})?"

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
URL_RE = re.compile(r"(?:https?://|www\.)[^\s<>\"']+", re.IGNORECASE)
PHONE_RE = re.compile(r"(?<![\w.])\+?\d[\d\s().-]{8,16}\d(?![\w])")
AMOUNT_RE = re.compile(_AMOUNT, re.IGNORECASE)
DATE_RES = [
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
    re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"),
    re.compile(rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s{_MONTHS}\.?,?\s\d{{4}}\b", re.IGNORECASE),
    re.compile(rf"\b{_MONTHS}\.?\s\d{{1,2}}(?:st|nd|rd|th)?,?\s\d{{4}}\b", re.IGNORECASE),
]

INVOICE_NO_RE = re.compile(
    r"\b(?:invoice|inv)\s*(?:no\.?|number|#)?\s*[:\-]?\s*"
    r"([A-Z0-9][A-Z0-9\-/]*\d[A-Z0-9\-/]*)",
    re.IGNORECASE,
)
TOTAL_RE = re.compile(
    rf"(?:grand\s+total|total\s+due|amount\s+due|total)\s*[:\-]?\s*({_AMOUNT})",
    re.IGNORECASE,
)
DUE_DATE_RE = re.compile(r"due\s+date\s*[:\-]?\s*([^\n]+)", re.IGNORECASE)

SKILL_KEYWORDS = [
    "python", "java", "javascript", "typescript", "c++", "sql", "r", "go", "rust",
    "machine learning", "deep learning", "nlp", "computer vision", "tensorflow",
    "pytorch", "scikit-learn", "pandas", "numpy", "docker", "kubernetes", "aws",
    "azure", "gcp", "git", "linux", "react", "node.js", "django", "flask", "fastapi",
    "excel", "power bi", "tableau",
]


class InformationExtractor:
    def __init__(self, use_spacy: bool = True):
        self._nlp = None
        self._use_spacy = use_spacy

    # ------------------------------------------------------------------ #
    def extract(self, text: str, doc_type: str = "unknown") -> Dict:
        """Return a dict of extracted fields; empty fields are omitted."""
        if not text or not text.strip():
            return {}

        fields: Dict = {
            "emails": self._unique(EMAIL_RE.findall(text)),
            "phones": self._find_phones(text),
            "dates": self._find_dates(text),
            "amounts": self._unique(AMOUNT_RE.findall(text)),
            "urls": self._unique(URL_RE.findall(text)),
        }
        fields.update(self._extract_entities(text))

        doc_type = (doc_type or "unknown").lower()
        if doc_type in {"invoice", "receipt", "bill"}:
            fields.update(self._extract_invoice_fields(text))
        elif doc_type in {"resume", "cv"}:
            fields.update(self._extract_resume_fields(text))

        return {k: v for k, v in fields.items() if v}

    # ------------------------------------------------------------------ #
    # Generic helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _unique(items: List[str]) -> List[str]:
        seen, result = set(), []
        for item in items:
            item = item.strip().rstrip(".,;:")
            if item and item.lower() not in seen:
                seen.add(item.lower())
                result.append(item)
        return result

    def _find_phones(self, text: str) -> List[str]:
        candidates = []
        for match in PHONE_RE.findall(text):
            digits = re.sub(r"\D", "", match)
            if 10 <= len(digits) <= 13:
                candidates.append(match.strip())
        return self._unique(candidates)

    def _find_dates(self, text: str) -> List[str]:
        found: List[str] = []
        for pattern in DATE_RES:
            found.extend(pattern.findall(text))
        return self._unique(found)

    # ------------------------------------------------------------------ #
    # spaCy named entities (optional)
    # ------------------------------------------------------------------ #
    def _load_nlp(self):
        if self._nlp is None and self._use_spacy:
            try:
                import spacy

                self._nlp = spacy.load("en_core_web_sm")
            except Exception:  # spaCy or model not installed
                self._use_spacy = False
        return self._nlp

    def _extract_entities(self, text: str) -> Dict:
        nlp = self._load_nlp()
        if nlp is None:
            return {}

        doc = nlp(text[:100_000])  # cap length for speed
        people, orgs, locations = [], [], []
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                people.append(ent.text)
            elif ent.label_ == "ORG":
                orgs.append(ent.text)
            elif ent.label_ in {"GPE", "LOC"}:
                locations.append(ent.text)
        return {
            "people": self._unique(people)[:10],
            "organizations": self._unique(orgs)[:10],
            "locations": self._unique(locations)[:10],
        }

    # ------------------------------------------------------------------ #
    # Document-type-specific rules
    # ------------------------------------------------------------------ #
    def _extract_invoice_fields(self, text: str) -> Dict:
        fields: Dict = {}

        match = INVOICE_NO_RE.search(text)
        if match:
            fields["invoice_number"] = match.group(1)

        totals = TOTAL_RE.findall(text)
        if totals:
            fields["total_amount"] = totals[-1].strip()  # last "total" is usually the grand total

        due = DUE_DATE_RE.search(text)
        if due:
            fields["due_date"] = due.group(1).strip()

        return fields

    def _extract_resume_fields(self, text: str) -> Dict:
        fields: Dict = {}

        name = self._guess_name(text)
        if name:
            fields["candidate_name"] = name

        lowered = text.lower()
        skills = [
            skill
            for skill in SKILL_KEYWORDS
            if re.search(rf"(?<![\w+.]){re.escape(skill)}(?![\w+])", lowered)
        ]
        if skills:
            fields["skills"] = skills

        return fields

    @staticmethod
    def _guess_name(text: str) -> Optional[str]:
        """Resumes usually start with the candidate's name: 2-4 alphabetic words."""
        for line in text.split("\n")[:5]:
            words = line.strip().split()
            if 2 <= len(words) <= 4 and all(re.fullmatch(r"[A-Za-z.'-]+", w) for w in words):
                return " ".join(words).title()
        return None