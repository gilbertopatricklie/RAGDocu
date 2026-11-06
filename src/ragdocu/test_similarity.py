import requests
import numpy as np

URL="http://localhost:11434/api/embed" 
EMBED_MODEL_NAME="nomic-embed-text:latest"

sentences = [
    "Saya suka belajar machine learning.",
    "Saya senang mempelajari kecerdasan buatan.",
    "Harga laptop terbaru sangat mahal."
]

payload = {
    "model": EMBED_MODEL_NAME,
    "input": sentences
}

response = requests.post(URL, json=payload)
response.raise_for_status()
result = response.json()
embeddings = np.array(result["embeddings"])

def cosine_similarity(vector_a, vector_b):
    dot_product = np.dot(vector_a, vector_b)
    
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    
    return dot_product / (norm_a * norm_b)

similarity_ab = cosine_similarity(embeddings[0], embeddings[1])
similarity_ac = cosine_similarity(embeddings[0], embeddings[2])

print("Similarity A-B: ", similarity_ab)
print("Similarity A-C: ", similarity_ac)