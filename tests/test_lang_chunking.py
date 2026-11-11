from ragdocu.langchain_rag.lang_document_processor import lang_process_pdf


def main():
    documents = lang_process_pdf("data/documents/sample.pdf")
    
    print("Total chunks: ", len(documents))
    
    for document in documents[:3]:
        print("\n==============================")
        print("ID:", document["id"])
        print("Metadata:", document["metadata"])
        print("Content:", document["text"])
        
    
if __name__ == "__main__":
    main()