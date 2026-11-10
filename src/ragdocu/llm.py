import requests

URL="http://localhost:11434/api/chat" 
CHAT_MODEL_NAME="qwen3:4b-instruct"

def generate_answer(prompt):
    payload = {
        "model": CHAT_MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "stream": False
    }
    
    response = requests.post(URL, json=payload, timeout=180)
    response.raise_for_status()
    result = response.json()
    return result["message"]["content"]