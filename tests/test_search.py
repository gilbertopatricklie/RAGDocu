from ragdocu.embedding import generate_embeddings
from ragdocu.vector_store import search_documents

query = "Apa itu Kue Pukis?"

query_embedding = generate_embeddings([query])[0]


for k in [3, 5]:
    print(f"\n========== TOP-K: {k} ==========")
    
    results = search_documents(query_embedding, n_result=k)

    for i in range(len(results["ids"][0])):
        print(f"\n===== Ranking {i + 1} =====")
        
        print("ID:", results["ids"][0][i])
        print("Distance:", results["distances"][0][i])
        print("Text:", results["documents"][0][i])
        print("Metadata:", results["metadatas"][0][i])