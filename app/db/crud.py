from app.db.models import FileSummary, FileChunk
from sqlalchemy.orm import Session
from datetime import datetime

def get_summary_by_file_hash(db: Session, file_hash: str):
    result = db.query(FileSummary).filter(FileSummary.file_hash == file_hash).first()
    if result:
        return {
            "summary_text": result.summary_text,
            "token_usage": result.token_usage
        }
    return None

def save_file_summary(db: Session, file_hash: str, file_name: str, summary_text: str, token_usage: dict = None):
    summary = FileSummary(
        file_hash=file_hash,
        file_name=file_name,
        summary_text=summary_text,
        token_usage=token_usage,
        created_at=datetime.utcnow()
    )
    db.add(summary)
    db.commit()
    db.refresh(summary)
    return summary

def cache_chunk(db: Session, chunk_id: str, file_name: str, chunk_index: int, chunk_text: str, summary_text: str):
    chunk = FileChunk(
        chunk_id=chunk_id,
        file_name=file_name,
        chunk_index=chunk_index,
        chunk_text=chunk_text,
        summary_text=summary_text,
        created_at=datetime.utcnow()
    )
    db.add(chunk)
    db.commit()
    db.refresh(chunk)
    return chunk

def get_chunks_by_file(db: Session, file_name: str):
    return db.query(FileChunk).filter(FileChunk.file_name == file_name).order_by(FileChunk.chunk_index).all()
