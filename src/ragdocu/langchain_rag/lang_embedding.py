from langchain_ollama import OllamaEmbeddings

URL="http://localhost:11434"
EMBED_MODEL_NAME="nomic-embed-text:latest"

embeddings = OllamaEmbeddings(
    model=EMBED_MODEL_NAME,
    base_url=URL
)

def generate_embeddings(texts):
    if not texts:
        return []
    
    return embeddings.embed_documents(texts)

def generate_query_embedding(query):
    return embeddings.embed_query(query)