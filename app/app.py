from fastapi import FastAPI
from app.api.routes import qa_routes
import uvicorn

main_app = FastAPI(
    title="PDF QA System",
    description="A system for answering questions based on PDF content",
    version="1.0.0"
)



main_app.include_router(qa_routes.router, prefix="/api/v1")

