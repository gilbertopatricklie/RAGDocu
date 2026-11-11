from ragdocu.langchain_rag.lang_embedding import generate_embeddings, generate_query_embedding

def main():
    documents = [
        "Machine Learning adalah cabang dari Artificial Intelligence.",
        "Deep Learning menggunakan neural network."
    ]
    
    document_embeddings = generate_embeddings(documents)
    query_embedding = generate_query_embedding("Apa itu machine learning?")
    
    print("Jumlah document embeddings: ", len(document_embeddings))
    print("Dimensi document embeddings: ", len(document_embeddings[0]))
    print("Dimensi query embeddings: ", len(query_embedding))
    
if __name__ == "__main__":
    main()

