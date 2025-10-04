# app/api/v1/routes_documents.py
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
import logging
from app.processors.file_processor import FileProcessor
from app.db.crud import get_summary_by_file_hash, save_file_summary
from app.api.deps import get_db

logger = logging.getLogger("routes_documents")
router = APIRouter(prefix="/v1/documents", tags=["documents"])

@router.post("/upload")
async def upload_document(file: UploadFile = File(...), db=Depends(get_db)):
    # validate extension
    ext = "." + file.filename.split(".")[-1].lower()
    from app.core.config import settings
    if ext not in settings.ALLOWED_FILE_EXTENSIONS:
        raise HTTPException(status_code=400, detail="File type not supported")

    processor = FileProcessor(file=file, db=db, include_examples=True)
    cached = get_summary_by_file_hash(db, processor.file_hash)
    if cached:
        return {"structured_notes": cached, "cached": True, "processing_time_minutes": 0.0}

    try:
        # await the async method
        result = processor.process_file()
        processing_time_min = result["processing_time"] / 60
        return {
            "structured_notes": result["structured_notes"],
            "cached": False,
            "processing_time_minutes": round(processing_time_min, 2)
        }
    except Exception as e:
        logger.error("Document processing failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
