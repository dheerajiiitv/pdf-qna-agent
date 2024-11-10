from fastapi import APIRouter, Body

from app.api.models.schema import QARequestSchema
from app.services.pdf_processor import PDFProcessor
from app.services.qa_engine import QAEngine
from app.services.slack_service import SlackService
from app.utils.logger import get_logger
from app.services.agent import PDFQAAgent
from fastapi import Depends
from llama_index.core.tools import  FunctionTool, ToolMetadata
from pydantic import BaseModel



logger = get_logger(__name__)
router = APIRouter()

class ValidationError(Exception):
    # Exception that raise wrong arguments for the tool
    pass


def qna_tool(**kwargs):
    """
    Answer questions based on the PDF content
    """
    try:
        request = QARequestSchema(**kwargs)
    except ValidationError as e:
        return "Invalid arguments for the tool, please check the request schema" + str(e)
    
    logger.info(f"PDF processor: {request.pdf_file}")
    pdf_processor = PDFProcessor(request.pdf_file)

    text = pdf_processor.extract_text()
    text_chunks = pdf_processor.split_text(text)
    response = QAEngine(text_chunks, questions=request.questions).answer_questions()
    
    return response

def slack_service_tool():
    """
    Post the results to Slack
    """
    return SlackService().post_results

class QAResultSchema(BaseModel):
    message: str

def get_tools():
    return [FunctionTool(fn=qna_tool, metadata=ToolMetadata(name="qna_tool", description="Answer questions based on the PDF content", fn_schema=QARequestSchema)), 
            FunctionTool(fn=slack_service_tool(), metadata=ToolMetadata(name="slack_service_tool", description="Post the results to Slack", fn_schema=QAResultSchema))]

def get_qna_agent(tools = Depends(get_tools)):
    return PDFQAAgent(tools=tools)



@router.post("/qa/agent")
def answer_questions_agent(
    user_query: str = Body(...),
    request: QARequestSchema = Body(...),
    agent: PDFQAAgent = Depends(get_qna_agent)
):
    logger.info(f"Answering questions with agent: {user_query}")
    user_query = f"User query: {user_query}\n\nPDF file: {request.pdf_file}\n\nQuestions: {request.questions}"
    return agent.chat(user_query)
