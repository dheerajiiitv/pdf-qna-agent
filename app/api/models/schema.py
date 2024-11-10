from pydantic import BaseModel, Field
from typing import List

class Question(BaseModel):
    text: str = Field(..., description="The question to be answered")
    
class QAResponse(BaseModel):
    question: str
    answer: str
    confidence: float

class QAResult(BaseModel):
    results: List[QAResponse]

class QARequestSchema(BaseModel):
    pdf_file: str = Field(..., description="The PDF file to answer questions from")
    questions: List[Question] = Field(..., description="The questions to answer")