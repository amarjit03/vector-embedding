from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from app.utils import unzip_codebase, get_python_files
from app.embedder import embed_code
from app.vector_store import store_embeddings, search
import shutil, os

app = FastAPI(title="Code Embedding API")

# ----------------------------
# Pydantic Model for /search
# ----------------------------
class SearchRequest(BaseModel):
    text: str

# ----------------------------
# Upload Endpoint
# ----------------------------
@app.post("/upload")
async def upload_zip(file: UploadFile = File(...)):
    zip_path = "uploaded_code.zip"
    with open(zip_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    folder = unzip_codebase(zip_path)
    snippets = get_python_files(folder)

    vectors, metas = [], []
    for code in snippets:
        try:
            vectors.append(embed_code(code))
            metas.append(code[:60].replace("\n", " "))
        except Exception as e:
            print("Skipping snippet due to error:", e)

    store_embeddings(vectors, metas)
    return {"message": f"✅ Embedded and stored {len(vectors)} code snippets."}

# ----------------------------
# Search Endpoint
# ----------------------------
@app.post("/search")
def search_code(request: SearchRequest):
    query_vector = embed_code(request.text)
    results = search(query_vector)
    return results
