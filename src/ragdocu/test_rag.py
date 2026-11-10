from ragdocu.rag_pipeline import ask_question

question = "Apa itu machine learning?"

result_1 = ask_question(question, top_k=3)
result_2 = ask_question(question, top_k=5, max_distance=0.35)
experiments = [
    ("Top-K 3", result_1),
    ("Top-K 5 + Threshold 0.5", result_2)
]

for name, result in experiments:
    print(f"\n===== {name} =====")

    print("\nAnswer:")
    print(result["answer"])

    print("\nSources:")

    for source in result["sources"]:
        print(
            source["metadata"]["source"],
            "Page:", source["metadata"]["page"],
            "Distance:", round(source["distance"], 4)
        )