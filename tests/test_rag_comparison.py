import json

from pathlib import Path
from ragdocu.rag_pipeline import ask_question
from ragdocu.langchain_rag.lang_rag_pipeline import ask_question as ask_question_langchain

QUESTION_PATH=Path("data/evaluation/questions.json")

def evaluate_pipeline(name, ask_function, questions):
    results = []
    
    print(f"\n{'=' * 50}")
    print(f"EVALUASI: {name}")
    print("=" * 50)
    
    for index, item in enumerate(questions, start=1):
        question = item["question"]
        
        print(f"\n[{index}] {question}")
        
        result = ask_function(
            question=question,
            top_k=3
        )
        
        answer = result["answer"]
        sources = result["sources"]
        
        print("Answer:", answer)
        print("Jumlah sources:", len(sources))
        
        for source in sources:
            metadata = source["metadata"]
            print(
                f"- {metadata['source']} "
                f"(Page {metadata['page']})"
            )
            
        results.append(
            {
                "question": question,
                "expected_answer": item["expected_answer"],
                "answer": answer,
                "source_count": len(sources),
                "sources": sources
            }
        )

    return results

def main():
    with open(QUESTION_PATH, "r", encoding="utf-8") as file:
        questions = json.load(file)
        
    manual_results = evaluate_pipeline(
        "Manual RAG",
        ask_question,
        questions
    )
    
    langchain_results = evaluate_pipeline(
        "Langchain RAG",
        ask_question_langchain,
        questions
    )
    
    output = {
        "manual": manual_results,
        "langchain": langchain_results
    }
    
    output_path = Path("data/evaluation/comparison_results.json")
    
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=4, ensure_ascii=False)
        
    print("\nEvaluasi selesai.")
    print("Hasil tersimpan di:", output_path)
    
if __name__ == "__main__":
    main()