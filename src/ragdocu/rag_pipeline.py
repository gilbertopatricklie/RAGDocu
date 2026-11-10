from ragdocu.embedding import generate_embeddings
from ragdocu.vector_store import search_documents
from ragdocu.llm import generate_answer

def ask_question(question, top_k=3, max_distance=None):
    query_embedding = generate_embeddings([question])[0]
    results = search_documents(query_embedding, n_results=top_k)
    
    retrieved_documents = []
    
    for i in range(len(results["ids"][0])):
        retrieved_documents.append(
            {
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i]
            }
        )
        
    if max_distance is not None:
        retrieved_documents = [
            document for document in retrieved_documents if document["distance"] <= max_distance
        ]
        
    if not retrieved_documents:
        return {
            "question": question,
            "answer": "Informasi yang dibutuhkan tidak ditemukan dalam dokumen.",
            "sources": []
        }
        
    context_parts = []
    
    for document in retrieved_documents:
        metadata = document["metadata"]
        
        context_parts.append(
            f"""
            Source: {metadata['source']}
            Page: {metadata['page']}
            Content: {document['text']}
            """
        )
        
    context = "\n".join(context_parts)
    
    prompt = f"""
    Kamu adalah asisten yang menjawab pertanyaan
    berdasarkan dokumen yang diberikan.

    Gunakan hanya informasi dari CONTEXT.

    Jika informasi tidak tersedia di CONTEXT,
    katakan bahwa informasi tersebut tidak ditemukan
    dalam dokumen.

    Jangan mengarang informasi.

    CONTEXT:
    {context}

    QUESTION:
    {question}

    Berikan jawaban dalam bahasa Indonesia.
    """
    
    answer = generate_answer(prompt)
    
    return {
        "question": question,
        "answer": answer,
        "sources": retrieved_documents
    }