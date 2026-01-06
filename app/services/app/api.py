"""
FastAPI entry point for the Arabic RAG system.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.services.embedding_service import EmbeddingService
from app.services.llm_service import LLMService
from app.services.vector_store import InMemoryVectorStore
from app.services.rag_pipeline import RAGPipeline

app = FastAPI(title="Arabic RAG Assistant")


class QuestionRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    answer: str


@app.on_event("startup")
def startup_event():
    """
    Initialize services and load sample documents.
    """
    embedding_service = EmbeddingService()
    llm_service = LLMService()
    vector_store = InMemoryVectorStore()

    pipeline = RAGPipeline(
        embedding_service=embedding_service,
        llm_service=llm_service,
        vector_store=vector_store,
    )

    documents = [
        "الذكاء الاصطناعي هو مجال يهتم ببناء أنظمة ذكية.",
        "تستخدم أنظمة الذكاء الاصطناعي في الطب والتعليم والصناعة.",
        "تعتمد نماذج اللغة الكبيرة على التعلم العميق.",
    ]

    pipeline.add_documents(documents)

    app.state.pipeline = pipeline


@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    """
    Ask a question to the Arabic RAG system.
    """
    try:
        answer = app.state.pipeline.answer(request.question)
        return AnswerResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
