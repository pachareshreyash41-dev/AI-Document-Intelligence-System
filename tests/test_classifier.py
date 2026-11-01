"""Tests for src/document_classifier.py

A tiny model is trained inside the test, so no real classifier.pkl is needed.
Assumes DocumentClassifier loads a joblib-saved scikit-learn Pipeline
(TF-IDF + classifier exposing predict_proba).
"""

import joblib
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from src.document_classifier import DocumentClassifier

TRAIN_TEXTS = [
    "invoice number total amount due payment terms",
    "invoice date bill to subtotal tax amount due",
    "invoice for services rendered total payable",
    "resume work experience education skills objective",
    "curriculum vitae professional experience skills references",
    "resume summary employment history education degree",
]
TRAIN_LABELS = ["invoice"] * 3 + ["resume"] * 3


@pytest.fixture(scope="module")
def model_path(tmp_path_factory):
    pipeline = make_pipeline(
        TfidfVectorizer(),
        LogisticRegression(max_iter=1000),
    )
    pipeline.fit(TRAIN_TEXTS, TRAIN_LABELS)

    path = tmp_path_factory.mktemp("models") / "classifier.pkl"
    joblib.dump(pipeline, path)
    return path


@pytest.fixture(scope="module")
def classifier(model_path):
    return DocumentClassifier(model_path=str(model_path))


def test_missing_model_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        DocumentClassifier(model_path=str(tmp_path / "missing.pkl"))


def test_predict_returns_label_and_confidence(classifier):
    label, confidence = classifier.predict("invoice total amount due payment")

    assert isinstance(label, str)
    assert isinstance(confidence, float)
    assert 0.0 <= confidence <= 1.0


def test_predict_invoice(classifier):
    label, _ = classifier.predict("invoice number subtotal tax total amount due")

    assert label == "invoice"


def test_predict_resume(classifier):
    label, _ = classifier.predict("resume education skills work experience")

    assert label == "resume"


def test_predict_is_deterministic(classifier):
    text = "invoice total amount due"

    assert classifier.predict(text) == classifier.predict(text)