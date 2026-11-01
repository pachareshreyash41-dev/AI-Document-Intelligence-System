"""
AI Document Intelligence System
Streamlit application entry point.

Pipeline:
    Upload -> Load pages -> OCR -> Preprocess -> Classify
    -> Extract information -> Save results -> Display results
"""

import json
import os
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.database import Database
from src.document_classifier import DocumentClassifier
from src.document_loader import load_document, save_upload
from src.information_extractor import InformationExtractor
from src.ocr_engine import OCREngine
from src.text_preprocessor import clean_text


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

UPLOAD_DIR = BASE_DIR / os.getenv("UPLOAD_DIR", "data/uploads")
PROCESSED_DIR = BASE_DIR / os.getenv("PROCESSED_DIR", "data/processed")
OUTPUT_DIR = BASE_DIR / os.getenv("OUTPUT_DIR", "outputs")

MODEL_PATH = BASE_DIR / os.getenv(
    "MODEL_PATH", "models/classifier.pkl"
)

DATABASE_PATH = BASE_DIR / os.getenv(
    "DATABASE_PATH", "data/documents.db"
)

OCR_LANGUAGE = os.getenv("OCR_LANGUAGE", "eng")
TESSERACT_CMD = os.getenv("TESSERACT_CMD") or None

try:
    MAX_UPLOAD_MB = max(1, int(os.getenv("MAX_UPLOAD_MB", "20")))
except ValueError:
    MAX_UPLOAD_MB = 20

ALLOWED_TYPES = [
    "pdf",
    "png",
    "jpg",
    "jpeg",
    "tiff",
    "bmp",
]

for directory in (UPLOAD_DIR, PROCESSED_DIR, OUTPUT_DIR):
    directory.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="AI Document Intelligence System",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CACHED COMPONENTS
# ============================================================

@st.cache_resource(show_spinner=False)
def get_ocr_engine():
    """Initialize and cache the OCR engine."""
    return OCREngine(
        language=OCR_LANGUAGE,
        tesseract_cmd=TESSERACT_CMD,
    )


@st.cache_resource(show_spinner=False)
def get_classifier():
    """Load the trained classifier if it exists."""
    if not MODEL_PATH.is_file():
        return None

    try:
        return DocumentClassifier(model_path=str(MODEL_PATH))
    except Exception:
        logger.exception("Could not initialize document classifier.")
        return None


@st.cache_resource(show_spinner=False)
def get_extractor():
    """Initialize and cache the information extractor."""
    return InformationExtractor()


@st.cache_resource(show_spinner=False)
def get_database():
    """Initialize and cache the database connection wrapper."""
    return Database(str(DATABASE_PATH))


# ============================================================
# DOCUMENT PROCESSING PIPELINE
# ============================================================

def process_document(uploaded_file):
    """
    Process one uploaded document.

    Returns:
        dict: Document metadata, extracted fields, and OCR text.
    """

    original_name = Path(uploaded_file.name).name
    extension = Path(original_name).suffix.lower().lstrip(".")

    if extension not in ALLOWED_TYPES:
        raise ValueError(
            f"Unsupported file type: .{extension}. "
            f"Allowed formats: {', '.join(ALLOWED_TYPES)}"
        )

    if uploaded_file.size > MAX_UPLOAD_MB * 1024 * 1024:
        raise ValueError(
            f"File exceeds the maximum size of {MAX_UPLOAD_MB} MB."
        )

    # 1. Save uploaded document.
    file_path = Path(save_upload(uploaded_file, UPLOAD_DIR))

    # 2. Load document pages.
    pages = load_document(str(file_path))

    if not pages:
        raise ValueError(
            "No pages could be loaded from the uploaded document."
        )

    # 3. Run OCR.
    ocr = get_ocr_engine()
    raw_text = ocr.extract_text(pages)

    if not isinstance(raw_text, str):
        raw_text = str(raw_text or "")

    # 4. Clean OCR text.
    cleaned_text = clean_text(raw_text)

    if not isinstance(cleaned_text, str):
        cleaned_text = str(cleaned_text or "")

    # 5. Classify document.
    classifier = get_classifier()

    if classifier is not None and cleaned_text.strip():
        try:
            label, confidence = classifier.predict(cleaned_text)
            label = str(label)
            confidence = float(confidence)
        except Exception:
            logger.exception("Document classification failed.")
            label = "unknown"
            confidence = 0.0
    else:
        label = "unknown"
        confidence = 0.0

    # 6. Extract structured information.
    extractor = get_extractor()

    try:
        fields = extractor.extract(cleaned_text, label)
    except Exception:
        logger.exception("Information extraction failed.")
        fields = {}

    if fields is None:
        fields = {}

    # 7. Prepare result.
    result = {
        "filename": original_name,
        "processed_at": datetime.now().isoformat(timespec="seconds"),
        "pages": len(pages),
        "document_type": label,
        "confidence": round(
            max(0.0, min(1.0, confidence)), 4
        ),
        "extracted_fields": fields,
        "text": cleaned_text,
    }

    # 8. Save extracted text and JSON output.
    stem = file_path.stem

    text_output = PROCESSED_DIR / f"{stem}.txt"
    json_output = OUTPUT_DIR / f"{stem}.json"

    text_output.write_text(
        cleaned_text,
        encoding="utf-8",
    )

    json_output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    # 9. Store in database.
    get_database().save_document(result)

    return result


# ============================================================
# RESULT DISPLAY
# ============================================================

def render_result(result):
    """Display document analysis results."""

    st.success("Document processing completed successfully!")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Document Type",
        str(result.get("document_type", "unknown")).title(),
    )

    col2.metric(
        "Confidence",
        f"{float(result.get('confidence', 0.0)) * 100:.1f}%",
    )

    col3.metric(
        "Pages",
        int(result.get("pages", 0)),
    )

    st.subheader("Extracted Information")

    fields = result.get("extracted_fields", {})

    if fields:
        st.json(fields, expanded=True)
    else:
        st.info(
            "No structured fields were extracted from this document."
        )

    st.subheader("Extracted Text")

    extracted_text = result.get("text", "")

    if extracted_text.strip():
        st.text_area(
            "OCR text",
            value=extracted_text,
            height=300,
            label_visibility="collapsed",
        )
    else:
        st.warning(
            "No text was detected. Try a clearer image or a searchable PDF."
        )

    json_data = json.dumps(
        result,
        indent=2,
        ensure_ascii=False,
        default=str,
    )

    st.download_button(
        label="Download JSON Results",
        data=json_data,
        file_name=f"{Path(result['filename']).stem}_results.json",
        mime="application/json",
        key="download_json",
    )

    st.download_button(
        label="Download Extracted Text",
        data=extracted_text,
        file_name=f"{Path(result['filename']).stem}_extracted.txt",
        mime="text/plain",
        key="download_text",
    )


# ============================================================
# ANALYZE PAGE
# ============================================================

def analyze_tab():
    """Render document upload and analysis interface."""

    st.header("Analyze Document")

    st.write(
        "Upload a PDF or image to extract text, "
        "classify the document, and identify useful information."
    )

    classifier = get_classifier()

    if classifier is None:
        st.warning(
            "No trained classifier was found. "
            "Documents will be labelled 'unknown' until a trained "
            "classifier is available."
        )

    uploaded_file = st.file_uploader(
        "Choose a document",
        type=ALLOWED_TYPES,
        accept_multiple_files=False,
        help=f"Maximum file size: {MAX_UPLOAD_MB} MB.",
        key="document_uploader",
    )

    if uploaded_file is None:
        st.info("Choose a document to begin.")
        return

    st.write(f"**File:** {Path(uploaded_file.name).name}")
    st.write(f"**Size:** {uploaded_file.size / 1024:.1f} KB")

    if uploaded_file.size > MAX_UPLOAD_MB * 1024 * 1024:
        st.error(
            f"The selected file exceeds {MAX_UPLOAD_MB} MB."
        )
        return

    if st.button(
        "Analyze Document",
        type="primary",
        use_container_width=True,
        key="analyze_document_button",
    ):
        try:
            with st.spinner(
                "Processing document: loading, OCR, classification, "
                "and information extraction..."
            ):
                result = process_document(uploaded_file)

            st.session_state["result"] = result

        except Exception as exc:
            logger.exception("Document processing failed.")

            st.error(
                "Document processing failed. "
                "Check the error details below."
            )

            st.exception(exc)
            return

    result = st.session_state.get("result")

    if result:
        st.divider()
        render_result(result)


# ============================================================
# HISTORY PAGE
# ============================================================

def history_tab():
    """Display previously processed documents."""

    st.header("Document History")

    try:
        rows = get_database().get_all_documents()
    except Exception as exc:
        logger.exception("Could not retrieve document history.")
        st.error("Unable to load document history.")
        st.exception(exc)
        return

    if rows is None or len(rows) == 0:
        st.info(
            "No documents have been processed yet. "
            "Analyze a document to see it here."
        )
        return

    history_df = pd.DataFrame(rows)

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True,
    )

    csv_data = history_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download History as CSV",
        data=csv_data,
        file_name="document_history.csv",
        mime="text/csv",
        key="download_history_csv",
    )


# ============================================================
# SIDEBAR
# ============================================================

def render_sidebar():
    """Display project status and configuration information."""

    with st.sidebar:
        st.title("📄 Document AI")
        st.caption("AI Document Intelligence System")

        st.divider()

        st.subheader("System Status")

        st.success("Streamlit UI: Ready")

        if MODEL_PATH.is_file():
            st.success("Classifier file: Found")
        else:
            st.warning("Classifier file: Not found")

        st.write(f"**OCR language:** `{OCR_LANGUAGE}`")
        st.write(f"**Maximum upload:** `{MAX_UPLOAD_MB} MB`")

        st.divider()

        st.caption(
            "Upload a document in the Analyze tab. "
            "Previously processed documents appear in History."
        )


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():
    """Render the main Streamlit application."""

    render_sidebar()

    st.title("📄 AI Document Intelligence System")

    st.markdown(
        """
        Extract text from documents, classify document types, and
        retrieve useful information using OCR and machine learning.
        """
    )

    tab_analyze, tab_history = st.tabs(
        ["📤 Analyze", "📚 History"]
    )

    with tab_analyze:
        analyze_tab()

    with tab_history:
        history_tab()


# Run the application.
main()                      

# Run this Command for local host(python -m streamlit run app.py --server.port 8502)