from ragdocu.langchain_rag.lang_vector_store import lang_vector_store


# Dokumen yang ingin dihapus
duplicate_sources = [
    "113c7e75898540adb96253b4854870f8.pdf",
    "261b7293da9e474fad4704cd2ea4f4dc.pdf",
]

# Ambil ID chunks dari kedua dokumen
ids_to_delete = []

for source in duplicate_sources:
    result = lang_vector_store.get(
        where={"source": source},
        include=["metadatas"]
    )

    ids = result["ids"]
    ids_to_delete.extend(ids)

    print(f"{source}: {len(ids)} chunks ditemukan")

print(f"\nTotal chunks yang akan dihapus: {len(ids_to_delete)}")

# Konfirmasi sebelum menghapus
confirmation = input("Hapus chunks tersebut? (yes/no): ")

if confirmation.lower() == "yes":
    if ids_to_delete:
        lang_vector_store.delete(ids=ids_to_delete)

    print("Cleanup berhasil.")

else:
    print("Cleanup dibatalkan.")

# Periksa jumlah chunks tersisa
remaining = lang_vector_store.get(
    include=["metadatas"]
)

print(f"Total chunks tersisa: {len(remaining['ids'])}")