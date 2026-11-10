from ragdocu.document_processor import process_pdf
from ragdocu.embedding import generate_embeddings

documents = process_pdf("data/documents/sample.pdf")
print("Total chunks:", len(documents))

texts = [
    document["text"] for document in documents
]

embeddings = generate_embeddings(texts)

print("Embedding shape:", embeddings.shape)
print("\nEmbedding chunk pertama:")
print(embeddings[0])
print("\nDimensi Embedding:", len(embeddings[0]))