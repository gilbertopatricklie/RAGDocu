from langchain_ollama import ChatOllama

URL="http://localhost:11434" 
CHAT_MODEL_NAME="qwen3:4b-instruct"

llm = ChatOllama(
    model=CHAT_MODEL_NAME,
    base_url=URL,
    temperature=0
)

def generate_answer(prompt):
    response = llm.invoke(prompt)
    
    return response.content