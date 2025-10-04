# app/processors/youtube_processor.py
import yt_dlp
import tempfile
import os
import shutil
import hashlib
import whisper
import spacy
import logging

from app.models.llm.gemini_model import generate_youtube_summary

logger = logging.getLogger("youtube_processor")
nlp = spacy.load("en_core_web_sm")


class YouTubeProcessor:
    def __init__(self, url: str):
        self.url = url
        self.temp_dir = tempfile.mkdtemp()
        self.url_hash = hashlib.sha256(url.encode()).hexdigest()

    def _sanitize_filename(self, filename: str) -> str:
        """Remove unsafe characters from filenames."""
        return "".join(c for c in filename if c.isalnum() or c in (" ", ".", "_", "-")).strip()

    def _download_audio(self) -> str:
        """Download audio from YouTube and return the final mp3 path."""
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": os.path.join(self.temp_dir, "%(title)s.%(ext)s"),
            "postprocessors": [
                {"key": "FFmpegExtractAudio", "preferredcodec": "mp3"}
            ],
            "quiet": True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(self.url, download=True)
            # Use the "filepath" from the postprocessed result
            audio_path = ydl.prepare_filename(info)
            # Replace extension with .mp3 since FFmpeg converts it
            if audio_path.endswith(".webm") or audio_path.endswith(".m4a"):
                audio_path = audio_path.rsplit(".", 1)[0] + ".mp3"

        logger.info("Downloaded audio: %s", audio_path)
        return audio_path

    def _transcribe_audio(self, audio_path: str) -> str:
        """Transcribe audio using Whisper."""
        logger.info("Transcribing audio with Whisper...")
        model = whisper.load_model("base", device="cpu")
        result = model.transcribe(audio_path)
        return result["text"]

    def _clean_text(self, text: str) -> str:
        """Remove stopwords and punctuation from transcript."""
        doc = nlp(text)
        tokens = [t.text for t in doc if not t.is_stop and not t.is_punct]
        return " ".join(tokens)

    def process_video(self) -> dict:
        """Full pipeline: download, transcribe, clean, and summarize."""
        try:
            audio_path = self._download_audio()
            transcript = self._transcribe_audio(audio_path)
            clean_text = self._clean_text(transcript)

            # Summarize with Gemini (smart chunking handled inside)
            structured_notes = generate_youtube_summary(
                clean_text, max_sentences=5, include_examples=False
            )

            return {
                "video_url": self.url,
                "url_hash": self.url_hash,
                "structured_notes": structured_notes,  
            }
        finally:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            logger.info("Cleaned up YouTube temp files")
