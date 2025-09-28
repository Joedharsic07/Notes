# app/main.py
import logging
from fastapi import FastAPI
from app.core.config import settings
from app.core.logging_config import configure_logging
from app.core.middleware import add_middlewares
from app.db.session import engine
from app.db.models import Base
from app.api.v1 import routes_documents, routes_youtube

# configure logging
configure_logging()
logger = logging.getLogger("main")

# create DB tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title=settings.APP_NAME, version=settings.VERSION)

# middlewares (CORS)
add_middlewares(app)

# include routers
app.include_router(routes_documents.router)
app.include_router(routes_youtube.router)

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Application startup")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("🛑 Application shutdown")
