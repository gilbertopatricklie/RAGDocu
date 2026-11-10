from ragdocu.document_processor import process_pdf
from ragdocu.embedding import generate_embeddings
from ragdocu.vector_store import store_documents, collection

documents = process_pdf("data/documents/sample.pdf")
print("Total chunks:", len(documents))

texts = [document["text"] for document in documents]
embeddings = generate_embeddings(texts)

print("Embedding shape:", embeddings.shape)

store_documents(documents, embeddings)
print("Total data di chromadb:", collection.count())
