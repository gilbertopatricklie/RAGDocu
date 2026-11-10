import chromadb

client = chromadb.PersistentClient(path="./chroba_db")
collection = client.get_or_create_collection(
    name="documents",
    metadata={
        "hnsw:space": "cosine"
    }
)

def store_documents(documents, embeddings):
    ids = [document["id"] for document in documents]
    texts = [document["text"] for document in documents]
    metadatas = [document["metadata"] for document in documents]
    
    collection.upsert(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=metadatas
    )
    
    print(f"{len(documents)} chunks berhasil disimpan")
    
    
def search_documents(query_embedding, n_results=3):
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=n_results
    )
    
    return results