from pathlib import Path
import pymupdf

def chunk_text(text, chunk_size=1000, chunk_overlap=200) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size harus lebih besar dari 0")
    
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap tidak boleh negatif dan harus kurang dari chunk_size")
    
    chunks = []
    start = 0
    step = chunk_size - chunk_overlap
    
    while start < len(text):
        chunk = text[start:start+chunk_size].strip()
        if chunk:
            chunks.append(chunk)

            
        start += step
    
    return chunks

def process_pdf(pdf_path, chunk_size=1000, chunk_overlap=200) -> list[dict]:
    pdf_path = Path(pdf_path)
    documents = []
    
    with pymupdf.open(pdf_path) as pdf:
        chunk_id = 0
        
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()
            
            if not text:
                continue
            
            page_chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            
            for chunk in page_chunks:
                documents.append({
                    "id": f"{pdf_path.stem}-p{page_number}-c{chunk_id}",
                    "text": chunk,
                    "metadata": {
                        "source": pdf_path.name,
                        "page": page_number,
                        "chunk_id": chunk_id
                    }
                })
                
                chunk_id += 1
                
    return documents

if __name__ == "__main__":
    result = process_pdf("data/documents/sample.pdf")
    print("Total chunks: ", len(result))
    
    for document in result[:3]:
        print("\n==============================")
        print("ID:", document["id"])
        print("Metadata:", document["metadata"])
        print("Text:", document["text"][:200])
    