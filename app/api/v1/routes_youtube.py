# app/api/v1/routes_youtube.py
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
import hashlib
import logging
from app.processors.youtube_processor import YouTubeProcessor
from app.db.crud import get_summary_by_file_hash, save_file_summary
from app.api.deps import get_db
from app.core.config import settings

logger = logging.getLogger("routes_youtube")
router = APIRouter(prefix="/v1/youtube", tags=["youtube"])

class URLRequest(BaseModel):
    youtube_url: str

@router.post("/notes")
def youtube_notes(request: URLRequest, db=Depends(get_db)):
    url_hash = hashlib.sha256(request.youtube_url.encode()).hexdigest()
    cached = get_summary_by_file_hash(db, url_hash)
    if cached:
        return {"notes": cached, "cached": True}
    processor = YouTubeProcessor(url=request.youtube_url, gemini_api_key=settings.GOOGLE_API_KEY)
    try:
        result = processor.process_video()
        save_file_summary(db, url_hash, "youtube_audio", result["structured_notes"])
        return {"notes": result["structured_notes"], "cached": False}
    except Exception as e:
        logger.error("YouTube processing failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
