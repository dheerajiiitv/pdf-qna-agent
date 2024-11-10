from typing import List, Dict, Tuple
import openai
from app.utils.logger import get_logger
import numpy as np

import json
from pydantic import BaseModel, Field
from app.api.models.schema import QAResponse, Question
from app.config import settings
logger = get_logger(__name__)

class RetrievalConfig(BaseModel):
    similarity_top_k: int = Field(default=3)

class QAEngine:
    def __init__(self, text_chunks: List[str], questions:List[Question]):
        self.text_chunks = text_chunks
        self.questions = questions
        self.client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
        
        self.document_embeddings = self._initialize_embeddings()
        logger.info("Finished initializing document embeddings")
        self.question_embeddings = self._initialize_question_embeddings()
        logger.info("Finished initializing question embeddings")
        self.retrieval_config = RetrievalConfig(
            similarity_top_k=3,
            similarity_threshold=0.75
        )
        
        

    def _initialize_embeddings(self):
        logger.info("Initializing document embeddings")
        # Call in batches of 100
        document_embeddings = []
        for i in range(0, len(self.text_chunks), 100):
            batch = self.text_chunks[i:i+100]
            embeddings = self.get_embeddings(batch)
            document_embeddings.extend(embeddings)
        
        return document_embeddings

    def _initialize_question_embeddings(self):
        question_embeddings = []
        question_texts = [question.text for question in self.questions]
        for i in range(0, len(question_texts), 100):
            batch = question_texts[i:i+100]
            embeddings = self.get_embeddings(batch)
            question_embeddings.extend(embeddings)
        return question_embeddings

    def get_embeddings(self, texts: List[str]) -> List[float]:
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )
        return [embedding.embedding for embedding in response.data]
    
    def _get_answer_from_llm(
        self, 
        question: Question, 
        context: str
    ) -> Dict:
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Answer the question based on the provided context. If you're not confident in the answer, respond with 'Data Not Available'. If you are confident, respond with the answer and the confidence score in json format. The confidence score should be between 0 and 1. JSON FORMAT: {'answer': <answer>, 'confidence': <confidence>}"},
                    {"role": "user", "content": f"Context: {context}\n\nQuestion: {question.text}"}
                ],
                temperature=0,
                response_format={
                    "type": "json_object"
                }
            )
            
            ai_response = response.choices[0].message.content
            try:
                ai_response_json = json.loads(ai_response)
                answer = ai_response_json["answer"]
                confidence = ai_response_json["confidence"]
            except json.JSONDecodeError:
                answer = "Data Not Available"
                confidence = 0.0
            logger.info(f"Answer: {answer}, Confidence: {confidence}")
            return {
                "answer": answer,
                "confidence": confidence
            }
        except Exception as e:
            logger.error(f"Error getting answer: {str(e)}")
            return {
                "answer": "Data Not Available",
                "confidence": 0.0
            }
        
    def answer_questions(self) -> List[QAResponse]:
        logger.info(f"Answering questions")
        answers = []
        # Find most relevant text chunks
        for i, question_embedding in enumerate(self.question_embeddings):
            question = self.questions[i]
            relevance_scores = [self._calculate_relevance(question_embedding, chunk_embedding)  for chunk_embedding in self.document_embeddings]
            top_k_indices = sorted(range(len(relevance_scores)), key=lambda i: relevance_scores[i], reverse=True)[:self.retrieval_config.similarity_top_k]
            logger.info(f"Top k indices: {top_k_indices}")
            # Filter out chunks that are not relevant enough
            top_k_chunks = [self.text_chunks[i] for i in top_k_indices]
            if len(top_k_chunks) == 0:
                logger.info(f"No relevant chunks found for question: {question.text}")
                answers.append(QAResponse(question=question.text, answer="Data Not Available", confidence=0.0))
                continue
            context = "\n".join(top_k_chunks)
            # Get answer from language model
            response = self._get_answer_from_llm(question=question, context=context)
            answer = response["answer"]
            confidence = response["confidence"]
            answers.append(QAResponse(question=question.text, answer=answer, confidence=confidence))
        logger.info(f"Answers: {answers}")
        return answers
            
    def _calculate_relevance(self, query_embedding: List[float], chunk_embedding: List[float]) -> float:
        # Cosine similarity
        return np.dot(query_embedding, chunk_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding))


    
    

# Test the QAEngine
if __name__ == "__main__":
    text_chunks = ["The capital of France is Paris.", "France is a country in Europe.", "Paris is the capital of France."]
   
    qa_engine = QAEngine(text_chunks=text_chunks)
    answer = qa_engine.answer_question(question="What is the capital of France?", top_k=3, similarity_threshold=0.75)
    logger.info(answer)
    # answer = qa_engine.answer_question(question="What is the capital of France?", top_k=3, similarity_threshold=0.75)
    
     # Output:
     # {'answer': 'The capital of France is Paris.', 'confidence': 1.0}
