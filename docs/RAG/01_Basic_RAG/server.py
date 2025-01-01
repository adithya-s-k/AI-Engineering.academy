# File: app.py

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
import os
import tempfile
import shutil
from dotenv import load_dotenv
import qdrant_client
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from chainlit.utils import mount_chainlit

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))

# Set up LlamaIndex
Settings.llm = OpenAI(model="gpt-4", api_key=OPENAI_API_KEY)
Settings.embed_model = OpenAIEmbedding(api_key=OPENAI_API_KEY)

# Set up Qdrant client
client = qdrant_client.QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
vector_store = QdrantVectorStore(client=client, collection_name="01_Basic_RAG")

# Create index
index = VectorStoreIndex.from_vector_store(vector_store)

def get_ingestion_pipeline():
    return IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=1024, chunk_overlap=20),
            Settings.embed_model,
        ],
        vector_store=vector_store,
    )

def get_query_engine():
    return index.as_query_engine(similarity_top_k=5)

class QueryRequest(BaseModel):
    query: str

app = FastAPI()

@app.post("/api/ingest")
async def ingest_files(files: List[UploadFile] = File(...)):
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save uploaded files to the temporary directory
            for file in files:
                file_path = os.path.join(temp_dir, file.filename)
                with open(file_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
            
            # Use SimpleDirectoryReader to load documents
            reader = SimpleDirectoryReader(temp_dir, recursive=True)
            documents = reader.load_data()
            
            # Process documents through the ingestion pipeline
            pipeline = get_ingestion_pipeline()
            nodes = pipeline.run(documents=documents)
            
            return JSONResponse(content={
                "message": f"Successfully ingested {len(files)} files",
                "ingested_files": [file.filename for file in files],
                "total_nodes": len(nodes)
            })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/query")
async def query_documents(request: QueryRequest):
    try:
        query_engine = get_query_engine()
        response = query_engine.query(request.query)        
        return {"response": str(response)}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)