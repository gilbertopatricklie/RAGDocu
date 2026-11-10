from ragdocu.rag_pipeline import ask_question

def main():
    print("================================")
    print("            RAGDocu")
    print("================================")
    print("Ketik 'exit' untuk keluar.\n")

    while True:
        question = input("Question/Pertanyaan: ").strip()

        if question.lower() == "exit":
            print("Berhasil keluar dari RAGDocu")
            break

        if not question:
            continue

        try:
            result = ask_question(
                question=question,
                top_k=3,
                max_distance=None
            )

            print(f"\nRAGDocu: {result['answer']}")

            print("\nSources:")
            for source in result["sources"]:
                metadata = source["metadata"]

                print(
                    f"- {metadata['source']} "
                    f"(Page {metadata['page']}, "
                    f"Distance: {source['distance']:.4f})"
                )

            print()

        except Exception as error:
            print(f"\nTerjadi kesalahan: {error}\n")


if __name__ == "__main__":
    main()
