from ragdocu.document_processor import process_pdf
from ragdocu.langchain_rag.lang_vector_store import lang_store_documents, lang_search_documents

def main():
    documents = process_pdf("data/documents/sample.pdf")
    
    lang_store_documents(documents)
    
    results = lang_search_documents(
        query="Apa itu machine learning?",
        k=3
    )
    
    for rank, (document, distance) in enumerate(results, start=1):
        print(f"\n===== RANKING {rank} =====")
        print("Distance:", distance)
        print("Content:", document.page_content)
        print("Metadata:", document.metadata)
        
if __name__ == "__main__":
    main()