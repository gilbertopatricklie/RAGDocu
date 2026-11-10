import requests

URL="http://localhost:11434/api/chat" 
CHAT_MODEL_NAME="qwen3:4b-instruct"

payload = {
    "model": CHAT_MODEL_NAME,
    "messages": [
        {
            "role": "user",
            "content": "Jelaskan apa itu RAG dalam bahasa Indonesia."
        }
    ],
    "stream": False
}

response = requests.post(URL, json=payload)
result = response.json()
print(result["message"]["content"])