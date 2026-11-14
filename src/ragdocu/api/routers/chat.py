from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from ragdocu.langchain_rag.lang_rag_pipeline import ask_question

router = APIRouter(prefix="/chat", tags=["Chat"])

class ChatRequest(BaseModel):
    question: str = Field(min_length=1, description="Pertanyaan yang ingin diajukan")
    top_k: int = Field(default=3, ge=1, le=10, description="Jumlah dokumen yang akan diretrieve")
    
@router.post("/")
def chat_request(request: ChatRequest):
    try:
        result = ask_question(
            question = request.question,
            top_k= request.top_k
        )
        
        return result
    
    except Exception as error:
        raise HTTPException(status_code=500, detail="Terjadi kesalahan ketika memproses pertanyaan") from error