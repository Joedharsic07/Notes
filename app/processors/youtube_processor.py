# app/processors/youtube_processor.py
import yt_dlp
import tempfile
import os
import shutil
import hashlib
import whisper
import spacy
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.models.llm.gemini_model import GeminiModel
import logging
from app.core.config import settings

logger = logging.getLogger("youtube_processor")
nlp = spacy.load("en_core_web_sm")

class YouTubeProcessor:
    def __init__(self, url: str, gemini_api_key: str = None):
        self.url = url
        self.temp_dir = tempfile.mkdtemp()
        self.url_hash = hashlib.sha256(url.encode()).hexdigest()
        api_key = gemini_api_key or settings.GOOGLE_API_KEY
        if not api_key:
            raise ValueError("Gemini API key is required")
        self.gemini = GeminiModel(api_key=api_key)

    def _download_audio(self) -> str:
        output_path = os.path.join(self.temp_dir, "%(title)s.%(ext)s")
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": output_path,
            "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3"}],
            "quiet": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(self.url, download=True)
            filename = ydl.prepare_filename(info).replace(".webm", ".mp3").replace(".m4a", ".mp3")
            logger.info("Downloaded audio: %s", filename)
            return filename

    def _transcribe_audio(self, audio_path: str) -> str:
        logger.info("Transcribing audio with whisper...")
        model = whisper.load_model("base", device="cpu")
        result = model.transcribe(audio_path)
        return result["text"]

    def _clean_text(self, text: str) -> str:
        doc = nlp(text)
        tokens = [t.text for t in doc if not t.is_stop and not t.is_punct]
        return " ".join(tokens)

    def _chunk_text(self, text: str, chunk_size: int = 1500, overlap: int = 200) -> list:
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size,
            chunk_overlap=overlap
        )
        return splitter.split_text(text)

    def process_video(self) -> dict:
        try:
            audio_path = self._download_audio()
            transcript = self._transcribe_audio(audio_path)
            clean_text = self._clean_text(transcript)
            chunks = self._chunk_text(clean_text)
            summaries = [self.gemini.summarize(chunk) for chunk in chunks]
            structured_notes = self.gemini.merge(summaries)
            return {
                "video_url": self.url,
                "url_hash": self.url_hash,
                "structured_notes": structured_notes,
            }
        finally:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            logger.info("Cleaned up youtube temp files")
