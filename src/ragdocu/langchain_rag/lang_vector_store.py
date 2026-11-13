from langchain_chroma import Chroma
from langchain_core.documents import Document
from ragdocu.langchain_rag.lang_embedding import embeddings

VECTOR_DB_PATH="./chroma_db_langchain"
COLLECTION_NAME="documents_langchain_recursive"

lang_vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=VECTOR_DB_PATH,
    collection_metadata={"hnsw:space": "cosine"}
)

def lang_store_documents(documents):
    langchain_documents = []
    
    for document in documents:
        langchain_documents.append(
            Document(
                page_content=document["text"],
                metadata=document["metadata"]
            )
        )
    
    ids = [document["id"] for document in documents]
    
    lang_vector_store.add_documents(
        documents=langchain_documents,
        ids=ids
    )
    
    print(f"{len(documents)} chunks berhasil disimpan")
    
def lang_search_documents(query, k=3):
    results = lang_vector_store.similarity_search_with_score(query=query, k=k)
    
    return results

def get_retriever(k=3):
    return lang_vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )