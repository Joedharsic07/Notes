import os
import tempfile
import shutil
import hashlib
import logging
import subprocess
import time
import json

import fitz  # PyMuPDF for PDF text extraction
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
import tiktoken

from app.models.llm.gemini_model import rewrite_notes_full

logger = logging.getLogger("file_processor")

MAX_INPUT_TOKENS = 1_048_576  # Gemini model input limit


class FileProcessor:
    def __init__(self, file, db, include_examples: bool = True):
        self.file = file
        self.db = db
        self.include_examples = include_examples

        # Temporary file handling
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = os.path.join(self.temp_dir, file.filename)

        with open(self.temp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        self.file_hash = self._compute_hash(self.temp_path)

    # -------------------------------------
    # Utility Methods
    # -------------------------------------
    def _compute_hash(self, path: str) -> str:
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(65536), b""):
                sha256.update(block)
        return sha256.hexdigest()

    def _convert_to_pdf(self, input_path: str) -> str:
        """Convert any file to PDF using LibreOffice."""
        if input_path.lower().endswith(".pdf"):
            return input_path

        cmd = [
            "libreoffice",
            "--headless",
            "--convert-to",
            "pdf",
            "--outdir",
            self.temp_dir,
            input_path,
        ]
        subprocess.run(cmd, check=True)
        pdf_name = os.path.splitext(os.path.basename(input_path))[0] + ".pdf"
        return os.path.join(self.temp_dir, pdf_name)

    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a PDF."""
        doc = fitz.open(pdf_path)
        full_text = "\n".join(page.get_text("text") for page in doc if page.get_text("text").strip())
        return full_text

    def _check_token_limit(self, text: str):
        """Ensure document size fits within Gemini model limits."""
        encoding = tiktoken.get_encoding("cl100k_base")
        num_tokens = len(encoding.encode(text))
        if num_tokens > MAX_INPUT_TOKENS:
            raise ValueError(
                f"Document too large: {num_tokens} tokens (max {MAX_INPUT_TOKENS})"
            )

    # -------------------------------------
    # Main Processing Pipeline
    # -------------------------------------
    def process_file(self):
        start_time = time.time()
        try:
            # Step 1: Convert and extract text
            pdf_path = self._convert_to_pdf(self.temp_path)
            pdf_text = self._extract_text_from_pdf(pdf_path)

            if not pdf_text.strip():
                raise ValueError("No text found in uploaded document.")

            # Step 2: Check token limit
            self._check_token_limit(pdf_text)

            # Step 3: Check cache
            existing = self.db.execute(
                text("SELECT summary_text, token_usage FROM file_summaries WHERE file_hash = :file_hash"),
                {"file_hash": self.file_hash},
            ).fetchone()

            if existing:
                logger.info("File already processed — returning cached notes.")
                token_usage = json.loads(existing[1]) if existing[1] else None
                return {
                    "file_name": self.file.filename,
                    "file_hash": self.file_hash,
                    "structured_notes": existing[0],
                    "token_usage": token_usage,
                    "processing_time": 0.0,
                    "cached": True,
                }

            # Step 4: Rewrite notes using Gemini
            logger.info("Rewriting notes using Gemini full-text mode...")
            rewritten_notes, token_usage = rewrite_notes_full(
                pdf_text, include_examples=self.include_examples
            )

            # Step 5: Save to DB
            try:
                self.db.execute(
                    text("""
                        INSERT INTO file_summaries (file_hash, file_name, summary_text, token_usage, created_at)
                        VALUES (:file_hash, :file_name, :summary_text, :token_usage, CURRENT_TIMESTAMP)
                    """),
                    {
                        "file_hash": self.file_hash,
                        "file_name": self.file.filename,
                        "summary_text": rewritten_notes,
                        "token_usage": json.dumps(token_usage),  # store as JSON
                    },
                )
                self.db.commit()
                logger.info("File processed and saved successfully.")
            except IntegrityError:
                self.db.rollback()
                existing = self.db.execute(
                    text("SELECT summary_text, token_usage FROM file_summaries WHERE file_hash = :file_hash"),
                    {"file_hash": self.file_hash},
                ).fetchone()
                rewritten_notes = existing[0]
                token_usage = json.loads(existing[1]) if existing[1] else None
                logger.info("Duplicate file detected — returning existing summary.")

            # Step 6: Return results
            processing_time = time.time() - start_time
            return {
                "file_name": self.file.filename,
                "file_hash": self.file_hash,
                "structured_notes": rewritten_notes,
                "token_usage": token_usage,
                "processing_time": round(processing_time, 2),
                "cached": False,
            }

        finally:
            # Cleanup temp files
            shutil.rmtree(self.temp_dir, ignore_errors=True)
