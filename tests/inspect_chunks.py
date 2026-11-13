
from ragdocu.vector_store import collection as manual_collection
from ragdocu.langchain_rag.lang_vector_store import lang_vector_store as langchain_vector_store


def inspect_collection(name, results):
    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    ids = results["ids"]
    documents = results["documents"]
    metadatas = results["metadatas"]

    for chunk_id, text, metadata in zip(ids, documents, metadatas):
        print(f"\nID: {chunk_id}")
        print(f"Page: {metadata['page']}")
        print(f"Chunk ID: {metadata['chunk_id']}")
        print(f"Content:\n{text}")
        print("-" * 60)


manual_results = manual_collection.get(
    include=["documents", "metadatas"]
)

langchain_results = langchain_vector_store.get(
    include=["documents", "metadatas"]
)

inspect_collection("MANUAL RAG", manual_results)
inspect_collection("LANGCHAIN RAG", langchain_results)
