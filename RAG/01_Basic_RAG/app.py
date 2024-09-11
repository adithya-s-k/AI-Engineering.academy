import os
import openai
import chainlit as cl
import argparse
from dotenv import load_dotenv
from llama_index.core import (
    Settings,
    StorageContext,
    VectorStoreIndex,
    SimpleDirectoryReader,
    load_index_from_storage,
)
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.core.query_engine.retriever_query_engine import RetrieverQueryEngine
from llama_index.core.callbacks import CallbackManager
from llama_index.core.service_context import ServiceContext

# LlamaIndex vector store import
import qdrant_client
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline

# Load environment variables from .env file
print("Loading Environment variables")
load_dotenv()

# Set OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

# Configure LLM settings
Settings.llm = OpenAI(
    model="gpt-4o-mini",
    temperature=0.1,
    max_tokens=1024,
    streaming=True,
    api_key=OPENAI_API_KEY,
)

# Set embedding model and context window
# Settings.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.embed_model = OpenAIEmbedding(embed_batch_size=10, api_key=OPENAI_API_KEY)
Settings.context_window = 4096
Settings.callback_manager = CallbackManager([cl.LlamaIndexCallbackHandler()])

# Connect to the Vector Database
print("Connecting to Vector Database")
client = qdrant_client.QdrantClient(
    host="localhost",
    port=6333
)

# Initialize the vector store
vector_store = QdrantVectorStore(client=client, collection_name="01_Basic_RAG")

def ingest_documents(data_dir):
    # Load documents from a directory
    documents = SimpleDirectoryReader(data_dir, recursive=True).load_data(
        show_progress=True
    )

    # Ingest data into the vector store
    print("Ingesting Data")
    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(
                chunk_size=1024, chunk_overlap=20
            ),  # Split documents into chunks
            Settings.embed_model,  # Use the embedding model for processing
        ],
        vector_store=vector_store,
    )

    # Ingest directly into the vector database
    nodes = pipeline.run(documents=documents, show_progress=True)
    print("Number of chunks added to vector DB:", len(nodes))

# Global variable to store the index
index = None

@cl.on_chat_start
async def start():
    global index
    # Initialize the index if it hasn't been created yet
    if index is None:
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

    # Initialize service context and query engine on chat start
    query_engine = index.as_query_engine(
        streaming=True,
        similarity_top_k=5,
    )
    cl.user_session.set("query_engine", query_engine)

    # Send a welcome message to the user
    await cl.Message(
        author="Assistant", content="Hello! I'm an AI assistant. How may I help you?"
    ).send()

@cl.on_message
async def handle_message(message: cl.Message):
    global index
    # Check if any files were uploaded
    if message.elements:
        for file in message.elements:
            if file.type == "file":
                # Read the file and process it
                documents = SimpleDirectoryReader(input_files=[file.path]).load_data()

                # Ingest the documents into the pipeline
                pipeline = IngestionPipeline(
                    transformations=[
                        SentenceSplitter(chunk_size=1024, chunk_overlap=20),
                        Settings.embed_model,
                    ],
                    vector_store=vector_store,
                )
                nodes = pipeline.run(documents=documents, show_progress=True)
                
                # Update the index with new documents
                index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
                
                # Update the query engine
                query_engine = index.as_query_engine(
                    streaming=True,
                    similarity_top_k=5,
                )
                cl.user_session.set("query_engine", query_engine)

                await cl.Message(
                    content=f"Processed {len(nodes)} chunks from the uploaded file."
                ).send()
            

    # Retrieve the query engine from the user session
    query_engine = cl.user_session.get("query_engine")  # type: RetrieverQueryEngine

    # Prepare to send a response to the user's message
    msg = cl.Message(content="", author="Assistant")

    # Query the engine with the user's message
    res = await cl.make_async(query_engine.query)(message.content)

    # Stream the response tokens back to the user
    for token in res.response_gen:
        await msg.stream_token(token)
    await msg.send()

if __name__ == "__main__":
    import sys
    import subprocess

    parser = argparse.ArgumentParser(description="RAG Script with ingestion option")
    parser.add_argument("--host", default="localhost", help='Host IP address')
    parser.add_argument("--port", type=int, default=8000, help='Port number')
    parser.add_argument('--ingest', action='store_true', help='Ingest documents before starting the chat')
    parser.add_argument('--data_dir', type=str, default="../data", help='Directory containing documents to ingest')
    args = parser.parse_args()

    if args.ingest:
        ingest_documents(args.data_dir)

    # Run the Chainlit app
    subprocess.run([
        "chainlit", "run", sys.argv[0],
        "--host", args.host,
        "--port", str(args.port)
    ], check=True)