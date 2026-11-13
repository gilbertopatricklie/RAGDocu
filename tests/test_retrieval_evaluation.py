from ragdocu.embedding import generate_embeddings as manual_embed
from ragdocu.vector_store import search_documents as manual_search
from ragdocu.langchain_rag.lang_vector_store import lang_search_documents as langchain_search


TEST_CASES = [
    {
        "question": "Apa contoh algoritma supervised learning?",
        "relevant_pages": [1],
    },
    {
        "question": "Apa contoh algoritma unsupervised learning?",
        "relevant_pages": [1],
    },
    {
        "question": "Apa bedanya Latency dengan Throughput?",
        "relevant_pages": [2],
    },
    {
        "question": "Apa yang dimaksud dengan learning rate dalam machine?",
        "relevant_pages": [],
    },
]

def retrieve_manual(question, k):
    query_embedding = manual_embed([question])[0]
    
    results = manual_search(
        query_embedding, n_results=k
    )
    
    retrieved = []
    
    for i in range (len(results["ids"][0])):
        retrieved.append(
            {
                "page": results["metadatas"][0][i]["page"],
                "chunk_id": results["metadatas"][0][i]["chunk_id"],
                "distance": results["distances"][0][i]
            }
        )
        
    return retrieved

def retrieved_langchain(question, k):
    results = langchain_search(question, k=k)
    
    retrieved = []
    
    for document, distance in results:
        retrieved.append(
            {
                "page": document.metadata["page"],
                "chunk_id": document.metadata["chunk_id"],
                "distance": distance
            }
        )
        
    return retrieved

def evaluate_system(system_name, retrieve_function, k=3):
    total_hits = 0
    total_in_scope = 0
    
    print(f"\n{'=' * 60}")
    print(f"{system_name} - Hit Rate@{k}")
    print(f"{'=' * 60}")
    
    for test_case in TEST_CASES:
        question = test_case["question"]
        relevant_pages = test_case["relevant_pages"]
        
        retrieved = retrieve_function(question, k)
        
        print(f"\nQuestion: {question}")
        
        for rank, document in enumerate(retrieved, start=1):
            print(
                f"  Rank {rank}: "
                f"Page {document['page']}, "
                f"Chunk {document['chunk_id']}, "
                f"Distance {document['distance']:.4f}"
            )
            
        if not relevant_pages:
            print(" Evaluation: OUT OF SCOPE")
            continue
        
        total_in_scope += 1
        
        hit = any(
            document["page"] in relevant_pages for document in retrieved
        )
        
        if hit:
            total_hits += 1
            print(" Hit: YES")
        else:
            print(" Hit: NO")
            
        hit_rate = total_hits / total_in_scope
        
        print(f"\nHit Rate@{k}: {hit_rate:.2%}")
        print(f"Hits: {total_hits}/{total_in_scope}")
        
if __name__ == "__main__":
    evaluate_system(
        "Manual RAG",
        retrieve_manual,
        k=3
    )
    
    evaluate_system(
        "Langchain RAG",
        retrieved_langchain,
        k=3
    )