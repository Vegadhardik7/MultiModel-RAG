from app.vectorstore.chroma_client import get_collection

col = get_collection()
print("Total documents in Chroma:", col.count())
