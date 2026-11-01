📄 AI Document Intelligence System
An AI-powered document processing application that extracts text from PDFs and images, classifies documents, identifies useful information, and stores analysis results for later review. The project combines OCR, natural-language processing, machine learning, and a Streamlit interface in one workflow.
Project type: AI / Machine Learning • OCR • NLP • Python • Streamlit
Status: Portfolio project

✨ Features
- PDF and image uploads — process PDF, PNG, JPG/JPEG, TIFF, and BMP files.
- OCR text extraction — use Tesseract OCR to read scanned pages and images, with optional OpenCV image preprocessing.
- Text preprocessing — normalize Unicode, remove control characters, reconnect hyphenated line breaks, and clean OCR output.
- Document classification — classify extracted text with a TF-IDF + Logistic Regression pipeline when a trained model is available.
- Information extraction — detect common fields such as email addresses, phone numbers, dates, amounts, and URLs.
- Resume-focused extraction — identify likely candidate names and skills.
- Invoice-focused extraction — extract invoice numbers, total amounts, and due-date text.
- Optional named-entity recognition — use spaCy's English model for people, organizations, and locations when installed.
- Persistent history — store document metadata, extracted fields, and text in SQLite.
- Exportable outputs — save cleaned text and structured JSON results.
- Automated tests — test core document loading, OCR, and classification functionality.
🧰 Tech Stack
Area	Technology
Language	Python
User interface	Streamlit
OCR	Tesseract OCR, pytesseract
Image processing	OpenCV, Pillow
PDF rendering	PyMuPDF
Machine learning	scikit-learn, TF-IDF, Logistic Regression
NLP / entity recognition	Regular expressions, optional spaCy
Data storage	SQLite, JSON
Model serialization	joblib
Testing	pytest
Configuration	python-dotenv


🏗️ Processing Workflow
Upload PDF or Image
        ↓
Validate and Save File
        ↓
Load PDF Pages / Image Frames
        ↓
OCR with Tesseract
        ↓
Clean and Normalize Text
        ↓
Classify Document (if a trained model exists)
        ↓
Extract Structured Information
        ↓
Save TXT + JSON + SQLite Record
        ↓
Display Results in Streamlit
📁 Project Structure
AI-Document-Intelligence-System/
├── app.py                         # Streamlit application
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment configuration template
├── .gitignore
├── src/
│   ├── database.py                 # SQLite persistence
│   ├── document_classifier.py      # TF-IDF + Logistic Regression
│   ├── document_loader.py          # Upload handling and PDF/image loading
│   ├── information_extractor.py    # Rule-based fields and optional spaCy NER
│   ├── ocr_engine.py               # Tesseract OCR and image preprocessing
│   └── text_preprocessor.py        # OCR text cleanup
├── tests/                          # Automated tests
├── data/
│   ├── uploads/                    # Uploaded documents (local runtime data)
│   ├── processed/                  # Cleaned extracted text
│   └── documents.db                # SQLite history (created at runtime)
├── models/
│   └── classifier.pkl              # Optional trained classifier (not included by default)
└── outputs/                        # Structured JSON results
⚙️ Requirements
- Python 3.10 or newer is recommended.
- Tesseract OCR must be installed separately; installing the Python package pytesseract alone is not enough.
- Windows, macOS, or Linux.
- Internet access during initial Python dependency installation.
🚀 Installation and Run (Windows CMD)
1. Clone the repository
git clone https://github.com/YOUR-USERNAME/AI-Document-Intelligence-System.git
cd AI-Document-Intelligence-System
Replace YOUR-USERNAME with your GitHub username and use the exact repository URL after creating the repository.
2. Create and activate a virtual environment
py -m venv .venv
.venv\Scripts\activate
If the py launcher is not available, use a valid Python installation path with python -m venv .venv.
3. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
4. Install Tesseract OCR
Install the Tesseract OCR engine for your operating system. On Windows, the common installation path is:
C:\Program Files\Tesseract-OCR\tesseract.exe
If Tesseract is installed elsewhere, configure its path in the .env file as described below.
5. Configure environment variables
Copy the example file:
copy .env.example .env
Add or update these settings in .env if needed:
# Optional custom executable path for Tesseract OCR
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe

# OCR language (requires the corresponding Tesseract language data)
OCR_LANGUAGE=eng

# Local storage paths
UPLOAD_DIR=data/uploads
PROCESSED_DIR=data/processed
OUTPUT_DIR=outputs
MODEL_PATH=models/classifier.pkl
DATABASE_PATH=data/documents.db

# Maximum upload size in MB
MAX_UPLOAD_MB=20
The application has defaults for these values, so .env is optional when the defaults suit your setup. Do not commit .env or private documents to GitHub.
6. Start the application
streamlit run app.py
Streamlit will print a local address in the terminal (usually http://localhost:8501). Open that address in your browser.
🧠 Train a Document Classifier (Optional)
The application can run without a saved classifier, but automatic document-type predictions require a trained model at models/classifier.pkl (or the path configured by MODEL_PATH).

Prepare a CSV file with two columns:
- text — example text from a document
- label — its category, such as invoice, resume, receipt, or letter
Example:
text,label
"Invoice number INV-100 total amount due Rs. 2500",invoice
"Invoice date subtotal tax amount payable",invoice
"Work experience education skills professional summary",resume
"Curriculum vitae employment history and technical skills",resume
Save your real, labelled training data as data/training.csv, then run:
python -m src.document_classifier data/training.csv --output models/classifier.pkl
Use a sufficiently large, representative dataset for each category. The example rows above illustrate the format only; they are not enough for a reliable production model. If no trained model is present, the app may report the document type as unknown.
🧪 Run Tests
From the project root, with the virtual environment activated:
python -m pytest -v
📦 Generated Files and Privacy
The application writes runtime files to data/uploads/, data/processed/, data/documents.db, and outputs/.
- These may contain private or personally identifiable information from uploaded documents.
- Keep real resumes, identity documents, invoices, extracted text, databases, and other private files out of your public repository.
- Before publishing, remove local test uploads and generated outputs, and ensure .gitignore excludes runtime data and Python cache files.
- OCR and extraction results can be incomplete or incorrect. Review important extracted fields manually before using them for decisions.
🔐 Limitations
- OCR quality depends on scan clarity, orientation, resolution, language data, and document layout.
- Tesseract OCR must be installed and available to the application.
- Classification quality depends on the quality and coverage of the labelled training dataset.
- TF-IDF + Logistic Regression is a lightweight baseline, not a large language model.
- spaCy entity recognition is optional and requires the en_core_web_sm model to be installed separately. If unavailable, rule-based extraction can still operate.
- The extractor is rule-based and may miss fields or incorrectly identify values in unusual layouts.
- This is a portfolio/demo project, not a certified document-verification or high-stakes decision system.
🛣️ Future Improvements
- Add a labelled benchmark dataset and report precision, recall, F1-score, and confusion matrix.
- Add support for more document categories and multilingual OCR.
- Improve table and layout extraction for invoices and forms.
- Add searchable history, filtering, and downloadable result reports.
- Add confidence thresholds and clearer handling for low-confidence predictions.
- Add authentication, access controls, and configurable retention for multi-user deployment.
- Package the application with Docker and add CI test workflows.
👨‍💻 Author
Shreyash Pachare

AI/ML portfolio project focused on document automation, OCR, NLP, and machine learning.
- GitHub: https://github.com/YOUR-USERNAME
- LinkedIn: Add your LinkedIn profile URL
Update the GitHub and LinkedIn placeholders before publishing.

📜 License
Choose a license before making the repository public. For example, the MIT License is a common option for open-source portfolio projects; add a LICENSE file if you choose it.
