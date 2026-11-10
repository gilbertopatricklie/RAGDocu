import pymupdf

PDF_PATH="data/documents/sample.pdf"

document = pymupdf.open(PDF_PATH)

print("Jumlah halaman: ", len(document))
print("Metadata: ", document.metadata)

extracted_pages = []

for page_number, page in enumerate(document, start=1):
    text = page.get_text("text")
    
    extracted_pages.append({"page_number": page_number, "text": text})
    
    print(f"\n===== HALAMAN {page_number} =====")
    print(text[:1000])
    
document.close()
