from fastapi import FastAPI
from ragdocu.api.routers import health, chat, documents

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to RAGDocu"}

app.include_router(health.router)
app.include_router(chat.router)
app.include_router(documents.router)