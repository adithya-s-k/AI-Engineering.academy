import os
import openai
import chainlit as cl
import argparse
from dotenv import load_dotenv
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
)
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.callbacks import CallbackManager
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import get_response_synthesizer
from llama_index.core.response_synthesizers import ResponseMode

# Load environment variables from .env file
print("Loading Environment variables")
load_dotenv()

# Set OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

# Configure LLM settings
Settings.llm = OpenAI(
    model="gpt-4",
    temperature=0.1,
    max_tokens=1024,
    streaming=True,
    api_key=OPENAI_API_KEY,
)

# Set embedding model and context window
# Settings.embed_model = OpenAIEmbedding(embed_batch_size=10, api_key=OPENAI_API_KEY)
Settings.context_window = 4096
Settings.callback_manager = CallbackManager([cl.LlamaIndexCallbackHandler()])

# Initialize the document store
# docstore = SimpleDocumentStore(namespace="BM25_RAG").persist(persist_path="docstore")
# docstore = SimpleDocumentStore().from_persist_path("docstore")

DOCSTORE_PATH = "docstore.json"

def initialize_docstore():
    if os.path.exists(DOCSTORE_PATH):
        print("Loading existing docstore...")
        docstore = SimpleDocumentStore.from_persist_path(DOCSTORE_PATH)
    else:
        print("Creating new docstore...")
        docstore = SimpleDocumentStore()
    return docstore

docstore = initialize_docstore()

## Mongo DB as Document Store

# !pip install llama-index-storage-index-store-mongodb
# !pip install llama-index-storage-docstore-mongodb

# from llama_index.storage.docstore.mongodb import MongoDocumentStore
# from llama_index.storage.kvstore.mongodb import MongoDBKVStore
# from pymongo import MongoClient
# from motor.motor_asyncio import AsyncIOMotorClient

# MONGO_URI = os.getenv("MONGO_URI")
# kv_store = MongoDBKVStore(mongo_client=MongoClient(MONGO_URI) , mongo_aclient=AsyncIOMotorClient(MONGO_URI))
# docstore = MongoDocumentStore(namespace="BM25_RAG" ,mongo_kvstore=kv_store).from_uri(uri=MONGO_URI)

# # # !pip install llama-index-storage-docstore-redis
# # # !pip install llama-index-storage-index-store-redis
# from llama_index.storage.docstore.redis import RedisDocumentStore

# REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
# REDIS_PORT = os.getenv("REDIS_PORT", 6379)

# docstore=RedisDocumentStore.from_host_and_port(
#         host=REDIS_HOST, port=REDIS_PORT, namespace="BM25_RAG"
#     )


def ingest_documents(data_dir):
    global docstore
    # Load documents from a directory
    documents = SimpleDirectoryReader(data_dir, recursive=True).load_data(
        show_progress=True
    )

    # Ingest data into the document store
    print("Ingesting Data")
    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=1024, chunk_overlap=20),
            Settings.embed_model,
        ],
    )

    # Process documents and add to document store
    nodes = pipeline.run(documents=documents, show_progress=True)
    docstore.add_documents(nodes)
    print("Number of chunks added to document store:", len(nodes))
    
    # Persist the updated docstore
    docstore.persist(persist_path=DOCSTORE_PATH)
    print(f"Docstore persisted to {DOCSTORE_PATH}")

# Global variable to store the query engine
query_engine = None

@cl.on_chat_start
async def start():
    global query_engine
    # Initialize the BM25 retriever and query engine if they haven't been created yet
    if query_engine is None:
        bm25_retriever = BM25Retriever.from_defaults(
            docstore=docstore,
            similarity_top_k=4,
        )
        
        query_engine = RetrieverQueryEngine(
            retriever=bm25_retriever,

        )

    # Send a welcome message to the user
    await cl.Message(
        author="Assistant", content="Hello! I'm an AI assistant using BM25 RAG. How may I help you?"
    ).send()

@cl.on_message
async def handle_message(message: cl.Message):
    global query_engine
    # Check if any files were uploaded
    if message.elements:
        for file in message.elements:
            if file.type == "file":
                # Read the file and process it
                documents = SimpleDirectoryReader(input_files=[file.path]).load_data()

                # Ingest the documents into the pipeline and document store
                pipeline = IngestionPipeline(
                    transformations=[
                        SentenceSplitter(chunk_size=1024, chunk_overlap=20),
                        Settings.embed_model,
                    ],
                )
                nodes = pipeline.run(documents=documents, show_progress=True)
                docstore.add_documents(nodes)
                
                # Update the BM25 retriever and query engine
                bm25_retriever = BM25Retriever.from_defaults(
                    docstore=docstore,
                    similarity_top_k=4,
                )
                query_engine = RetrieverQueryEngine(
                    retriever=bm25_retriever,
                )

                await cl.Message(
                    content=f"Processed {len(nodes)} chunks from the uploaded file."
                ).send()
    res = await query_engine.aquery(message.content)
    await cl.Message(content=str(res), author="Assistant").send()

if __name__ == "__main__":
    import sys
    import subprocess

    parser = argparse.ArgumentParser(description="BM25 RAG Script with ingestion option")
    parser.add_argument('--ingest', action='store_true', help='Ingest documents before starting the chat')
    parser.add_argument('--data_dir', type=str, default="../data", help='Directory containing documents to ingest')
    args = parser.parse_args()

    if args.ingest:
        ingest_documents(args.data_dir)

    # Run the Chainlit app
    subprocess.run(["chainlit", "run", sys.argv[0] ,"-w"], check=True)