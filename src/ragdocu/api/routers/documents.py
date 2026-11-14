import shutil

from pathlib import Path
from uuid import uuid4
from fastapi import APIRouter, UploadFile, File, HTTPException
from ragdocu.langchain_rag.lang_document_processor import lang_process_pdf
from ragdocu.langchain_rag.lang_vector_store import lang_store_documents


router = APIRouter(prefix="/documents", tags=["Documents"])

UPLOAD_DIR=Path("data/uploads")
MAX_FILE_SIZE = 10 * 1024 * 1024 #10 MB
CHUNK_SIZE = 1024 * 1024 #1 MB
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

@router.post("/upload")
def upload_document(file: UploadFile = File(...)):
    file_path = None
    
    try:
        if not file.filename or Path(file.filename).suffix.lower() != ".pdf":
            raise HTTPException(status_code=400, detail="Format/ekstensi file harus .pdf")
    
        unique_filename = f"{uuid4().hex}.pdf"
        file_path = UPLOAD_DIR / unique_filename
        
        total_size = 0
    
        with file_path.open("wb") as destination:
            while True:
                chunk = file.file.read(CHUNK_SIZE)
                
                if not chunk:
                    break
                
                total_size += len(chunk)
                
                if total_size > MAX_FILE_SIZE:
                    raise HTTPException(status_code=413, detail="Ukuran file melebihi 10 MB")
                
                destination.write(chunk)
                
        with file_path.open("rb") as uploaded_file:
            pdf_signature = uploaded_file.read(5)
            
            if pdf_signature != b"%PDF-":
                raise HTTPException(status_code=400, detail="File yang diunduh bukan PDF yang valid")
            
            try:
                documents = lang_process_pdf(file_path)
                
            except Exception as error:
                raise HTTPException(status_code=400, detail="PDF tidak dapat diproses") from error
            
            if not documents:
                raise HTTPException(status_code=400, detail="PDF tidak mengandung teks yang dapat diekstrak")
            
            try:
                lang_store_documents(documents)
                
            except Exception as error:
                raise HTTPException(status_code=500, detail="Gagal menyimpan dokumen ke Chroma") from error
            
            return {
                "message": "Dokumen berhasil diupload dan diproses",
                "filename": file.filename,
                "file_size": total_size,
                "chunks_created": len(documents)
            }
            
    except HTTPException:
        raise
    
    except Exception as error:
        raise HTTPException(status_code=500, detail="Terjadi kesalahan ketika memproses dokumen") from error
    
    finally:
        if file_path:
            file_path.unlink(missing_ok=True)