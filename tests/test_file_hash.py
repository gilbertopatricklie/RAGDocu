from pathlib import Path
from ragdocu.api.routers.documents import calculate_file_hash

file_path = Path("data/documents/sample.pdf")
file_hash = calculate_file_hash(file_path)

print("SHA-256:", file_hash)