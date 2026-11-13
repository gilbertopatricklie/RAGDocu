from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

URL="http://localhost:11434" 
CHAT_MODEL_NAME="qwen3:4b-instruct"

llm = ChatOllama(
    model=CHAT_MODEL_NAME,
    base_url=URL,
    temperature=0
)

rag_prompt = ChatPromptTemplate.from_template(
    """
    Kamu adalah asisten AI yang menjawab pertanyaan
    berdasarkan dokumen yang diberikan.

    Gunakan hanya informasi dari CONTEXT.
    Jika informasi tidak tersedia dalam CONTEXT,
    katakan bahwa informasi tersebut tidak ditemukan
    dalam dokumen.

    Jangan mengarang informasi.

    CONTEXT:
    {context}

    QUESTION:
    {question}

    Berikan jawaban dalam bahasa Indonesia.
    """
)

def generate_answer(prompt):
    response = llm.invoke(prompt)
    
    return response.content

