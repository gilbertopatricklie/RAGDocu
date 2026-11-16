from ragdocu.langchain_rag.lang_vector_store import lang_vector_store

data = lang_vector_store.get(
    include=["metadatas", "documents"]
)

print(f"Total chunks: {len(data['ids'])}")

for doc_id, metadata, content in zip(
    data["ids"],
    data["metadatas"],
    data["documents"]
):
    print("=" * 60)
    print(f"ID: {doc_id}")
    print(f"Metadata: {metadata}")
    print(f"Content: {content[:150]}")