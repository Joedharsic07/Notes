# app/processors/file_processor.py
import os, tempfile, shutil, hashlib, logging, subprocess, time
import spacy
from langchain.text_splitter import RecursiveCharacterTextSplitter
import fitz
import pytesseract
from pdf2image import convert_from_path
from app.models.llm.local_model import local_summarize_text, merge_with_local_llama

from app.core.config import settings

logger = logging.getLogger("file_processor")
nlp = spacy.load("en_core_web_sm")

class FileProcessor:
    def __init__(self, file, db=None, include_examples: bool = True, subject: str = "general"):
        self.file = file
        self.db = db
        self.include_examples = include_examples
        self.subject = subject
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = os.path.join(self.temp_dir, file.filename)
        with open(self.temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        self.file_hash = self._compute_hash(self.temp_path)

    def _compute_hash(self, path: str) -> str:
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(65536), b""):
                sha256.update(block)
        return sha256.hexdigest()

    def _convert_to_pdf(self, input_path: str) -> str:
        logger.info("Converting document to PDF (if needed)...")
        if input_path.lower().endswith(".pdf"):
            logger.info("Document is already PDF.")
            return input_path
        cmd = ["libreoffice", "--headless", "--convert-to", "pdf", "--outdir", self.temp_dir, input_path]
        subprocess.run(cmd, check=True)
        pdf_name = os.path.splitext(os.path.basename(input_path))[0] + ".pdf"
        pdf_path = os.path.join(self.temp_dir, pdf_name)
        logger.info(f"Converted to PDF: {pdf_path}")
        return pdf_path

    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        logger.info("Extracting text from PDF...")
        text_content = []
        doc = fitz.open(pdf_path)
        for page in doc:
            txt = page.get_text("text")
            if txt.strip():
                text_content.append(txt)
        if not text_content:
            logger.info("No text detected on PDF pages, attempting OCR...")
            images = convert_from_path(pdf_path)
            for img in images:
                text_content.append(pytesseract.image_to_string(img))
        text = "\n".join(text_content)
        logger.info(f"Text extraction finished, chars: {len(text)}")
        return text

    def _clean_text(self, text: str) -> str:
        logger.info("Cleaning text...")
        doc = nlp(text)
        tokens = [t.text for t in doc if not t.is_stop and not t.is_punct]
        clean = " ".join(tokens)
        logger.info(f"Cleaned text length: {len(clean)}")
        return clean

    def _chunk_text(self, text: str, chunk_size: int = 1200, overlap: int = 200):
        logger.info("Chunking text...")
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
        )
        chunks = splitter.split_text(text)
        logger.info(f"Created {len(chunks)} chunks")
        return chunks

    def process_file(self):
        start_time = time.time()
        try:
            step = "convert"
            pdf_path = self._convert_to_pdf(self.temp_path)
            step = "extract"
            text = self._extract_text_from_pdf(pdf_path)
            if not text.strip():
                raise ValueError("No text found in document")
            step = "clean"
            clean = self._clean_text(text)
            step = "chunk"
            chunks = self._chunk_text(clean)
            step = "summarize"
            summaries = []
            for i, chunk in enumerate(chunks, 1):
                logger.info("Summarizing chunk %d/%d", i, len(chunks))
                summaries.append(local_summarize_text(chunk, max_sentences=3, include_examples=self.include_examples, subject=self.subject))
            step = "merge"
            structured_notes = merge_with_local_llama(summaries, include_examples=self.include_examples, subject=self.subject)
            elapsed = time.time() - start_time
            logger.info("File processed: %s in %.2f minutes", self.file.filename, elapsed/60)
            return {
                "file_name": self.file.filename,
                "file_hash": self.file_hash,
                "structured_notes": structured_notes,
                "processing_time": elapsed
            }
        finally:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            logger.info("Temporary files removed")
