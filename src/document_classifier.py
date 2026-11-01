"""
Document classifier: TF-IDF + Logistic Regression, saved with joblib.

Train from the command line with a CSV that has `text` and `label` columns:

    python -m src.document_classifier data/training.csv
    python -m src.document_classifier data/training.csv --output models/classifier.pkl
"""

import argparse
from pathlib import Path
from typing import Sequence, Tuple, Union

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.text_preprocessor import normalize_for_model


class DocumentClassifier:
    def __init__(self, model_path: Union[str, Path]):
        """Load a trained model. Raises FileNotFoundError if it doesn't exist."""
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Classifier model not found: {self.model_path}")
        self.pipeline: Pipeline = joblib.load(self.model_path)

    @property
    def classes(self):
        return list(self.pipeline.classes_)

    def predict(self, text: str) -> Tuple[str, float]:
        """Return (label, confidence) for a document's text."""
        processed = normalize_for_model(text)
        probabilities = self.pipeline.predict_proba([processed])[0]
        best = probabilities.argmax()
        return str(self.pipeline.classes_[best]), float(probabilities[best])

    # ------------------------------------------------------------------ #
    @staticmethod
    def train(
        texts: Sequence[str],
        labels: Sequence[str],
        model_path: Union[str, Path],
        test_size: float = 0.2,
        verbose: bool = True,
    ) -> Pipeline:
        """Train a new model, optionally print a validation report, and save it."""
        processed = [normalize_for_model(t) for t in texts]

        pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        ngram_range=(1, 2), min_df=1, max_df=0.95, sublinear_tf=True
                    ),
                ),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]
        )

        # Hold out a validation split only when there is enough data per class.
        can_split = len(set(labels)) > 1 and min(
            labels.count(l) if isinstance(labels, list) else list(labels).count(l)
            for l in set(labels)
        ) >= 3

        if can_split:
            x_train, x_test, y_train, y_test = train_test_split(
                processed, labels, test_size=test_size, stratify=labels, random_state=42
            )
            pipeline.fit(x_train, y_train)
            if verbose:
                print(classification_report(y_test, pipeline.predict(x_test)))
            pipeline.fit(processed, labels)  # refit on everything for the final model
        else:
            pipeline.fit(processed, labels)
            if verbose:
                print("Not enough samples per class for a validation split; trained on all data.")

        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, model_path)
        if verbose:
            print(f"Model saved to {model_path}")
        return pipeline


def _main() -> None:
    import pandas as pd

    parser = argparse.ArgumentParser(description="Train the document classifier.")
    parser.add_argument("csv", help="CSV file with 'text' and 'label' columns")
    parser.add_argument("--output", default="models/classifier.pkl")
    args = parser.parse_args()

    df = pd.read_csv(args.csv).dropna(subset=["text", "label"])
    DocumentClassifier.train(
        df["text"].astype(str).tolist(), df["label"].astype(str).tolist(), args.output
    )


if __name__ == "__main__":
    _main()