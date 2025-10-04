from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Text, DateTime, JSON
from datetime import datetime

Base = declarative_base()

class FileSummary(Base):
    __tablename__ = "file_summaries"
    id = Column(Integer, primary_key=True, index=True)
    file_hash = Column(String, unique=True, index=True, nullable=False)
    file_name = Column(String, nullable=False)
    summary_text = Column(Text, nullable=False)
    token_usage = Column(JSON, nullable=True)  # NEW COLUMN for Gemini token usage
    created_at = Column(DateTime, default=datetime.utcnow)

class FileChunk(Base):
    __tablename__ = "file_chunks"
    id = Column(Integer, primary_key=True, index=True)
    chunk_id = Column(String, unique=True, index=True, nullable=False)
    file_name = Column(String, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    chunk_text = Column(Text, nullable=False)
    summary_text = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
