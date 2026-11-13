from fastapi import FastAPI
from ragdocu.api.routers import health

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to RAGDocu"}

app.include_router(health.router)