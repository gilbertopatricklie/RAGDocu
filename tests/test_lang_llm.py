from ragdocu.langchain_rag.lang_llm import generate_answer

def main():
    prompt = """
    Kamu adalah asisten AI yang menjawab dalam bahasa Indonesia.

    Jelaskan secara singkat apa itu Machine learning.
    """
    
    answer = generate_answer(prompt=prompt)
    
    print("===== QWEN3 RESPONSE =====")
    print(answer)
    
if __name__ == "__main__":
    main()