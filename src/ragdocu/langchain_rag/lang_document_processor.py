import pymupdf

from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200,
    separators=["\n\n", "\n", ". ", " ", ""],
    keep_separator="end"
)

def lang_process_pdf(pdf_path):
    pdf_path = Path(pdf_path)
    documents = []
    
    with pymupdf.open(pdf_path) as pdf:
        chunk_id = 0
        
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()
            
            if not text:
                continue
            
            chunks = text_splitter.split_text(text)
            
            for chunk in chunks:
                documents.append(
                    {
                        "id": f"{pdf_path.stem}-p{page_number}-c{chunk_id}",
                        "text": chunk,
                        "metadata": {
                            "source": pdf_path.name,
                            "page": page_number,
                            "chunk_id": chunk_id
                        }
                    }
                )
                
                chunk_id += 1
    
    return documents