def chunk_text(text, chunk_size=500, chunk_overlap=100) -> str:
    
    if chunk_size <= 0:
        raise ValueError("chunk_size harus lebih dari 0")
    
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap harus lebih dari 0 dan kurang dari atau sama dengan chunk_size")
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        
        start += chunk_size - chunk_overlap
        
    return chunks
    
text = """
Machine learning merupakan cabang dari artificial intelligence.
Teknologi ini memungkinkan komputer mempelajari pola dari data.

Deep learning merupakan bagian dari machine learning
yang menggunakan neural network dengan banyak layer.

Natural language processing memungkinkan komputer
memahami dan memproses bahasa manusia.
"""

chunks = chunk_text(text, chunk_size=100, chunk_overlap=21)

for i, chunk in enumerate(chunks, start=1):
    print(f"\n===== CHUNK {i} =====")
    print(chunk)
    print(type(chunk))