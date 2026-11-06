import requests

URL="http://localhost:11434/api/embed" 
EMBED_MODEL_NAME="nomic-embed-text:latest"

payload = {
    "model": EMBED_MODEL_NAME,
    "input": [
            "Saya sedang belajar Retrieval Augmented Generation (RAG).",
            "Saya suka machine learning.",
            "Saya ingin punya banyak RAM."
        ]
}

response = requests.post(URL, json=payload)
response.raise_for_status()
result = response.json()
embedding = result["embeddings"][0]

print("Tipe data: ", type(embedding))
print("Dimensi: ", len(embedding))
print("5 Angka pertama: ", embedding[:5])