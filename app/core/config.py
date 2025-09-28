# app/core/config.py
from pydantic_settings import BaseSettings
from typing import List, Set
import os
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    APP_NAME: str = "AI Notes Generator"
    VERSION: str = "1.0.0"
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"

    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/notes_cache")
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    MODEL_PATH: str = os.getenv("MODEL_PATH", ".app/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf")

    # Security / CORS
    ALLOWED_ORIGINS: List[str] = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
    ALLOWED_FILE_EXTENSIONS: Set[str] = {".pdf", ".docx", ".doc", ".txt",".ppt",".pptx"}

    class Config:
        env_file = ".env"

settings = Settings()
