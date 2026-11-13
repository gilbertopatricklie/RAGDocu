from ragdocu.langchain_rag.lang_rag_pipeline import ask_question


def main():
    question = "Apa itu machine learning?"
    
    result = ask_question(question, top_k=3)
    
    print("===== QUESTION =====")
    print(result["question"])

    print("\n===== ANSWER =====")
    print(result["answer"])
    
    print("\n===== SOURCES =====")

    for source in result["sources"]:
        metadata = source["metadata"]

        print(
            f"- {metadata['source']} "
            f"(Page {metadata['page']})"
        )

        print("Content:", source["text"])
    
if __name__ == "__main__":
    main()