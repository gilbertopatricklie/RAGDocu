import requests
import numpy as np

URL="http://localhost:11434/api/embed"
EMBED_MODEL_NAME="nomic-embed-text:latest"

def generate_embeddings(texts):
    
    if not texts:
        return []
    
    payload = {
        "model": EMBED_MODEL_NAME,
        "input": texts
    }
    
    response = requests.post(
        URL,
        json=payload,
        timeout=120
    )
    
    response.raise_for_status()
    result = response.json()
    embeddings = np.array(result["embeddings"])
    
    return embeddings