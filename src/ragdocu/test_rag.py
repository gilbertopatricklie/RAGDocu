from ragdocu.rag_pipeline import ask_question

question = "Apa yang dimaksud dengan Machine learning?"

result = ask_question(question)

print("\n===== QUESTION =====")
print(result["question"])


print("\n===== ANSWER =====")
print(result["answer"])


print("\n===== SOURCES =====")

for source in result["sources"]:

    print(
        f"- {source['metadata']['source']}, "
        f"Page {source['metadata']['page']}, "
        f"Distance: {source['distance']:.4f}"
    )