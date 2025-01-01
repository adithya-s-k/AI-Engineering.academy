import os
import openai
import chainlit as cl
import argparse
from dotenv import load_dotenv
from llama_index.core import (
    Settings,
    VectorStoreIndex,
    SimpleDirectoryReader,
)
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.callbacks import CallbackManager
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.postprocessor import LLMRerank

# LlamaIndex vector store import
import qdrant_client
from llama_index.vector_stores.qdrant import QdrantVectorStore

# Load environment variables from .env file
print("Loading Environment variables")
load_dotenv()

# Set OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

# Configure LLM settings
Settings.llm = OpenAI(
    model="gpt-4-1106-preview",
    temperature=0.1,
    max_tokens=1024,
    streaming=True,
    api_key=OPENAI_API_KEY,
)

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
vector_store = QdrantVectorStore(client=client, collection_name="02_ReRanker_RAG")

def ingest_documents(data_dir):
    documents = SimpleDirectoryReader(data_dir, recursive=True).load_data(
        show_progress=True
    )
    print("Ingesting Data")
    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=1024, chunk_overlap=20),
            Settings.embed_model,
        ],
        vector_store=vector_store,
    )
    nodes = pipeline.run(documents=documents, show_progress=True)
    print("Number of chunks added to vector DB:", len(nodes))

# Global variable to store the index
index = None

def get_reranker(rerank_method):
    if rerank_method == "llm":
        return LLMRerank(choice_batch_size=5, top_n=2)
    elif rerank_method == "cohere":
        try:
            from llama_index.postprocessor.cohere_rerank import CohereRerank
        except:
            raise "Cohere reranker package missiong please install : pip install llama-index-postprocessor-cohere-rerank"
        cohere_api_key = os.environ.get("COHERE_API_KEY")
        if not cohere_api_key:
            raise ValueError("COHERE_API_KEY not found in environment variables")
        return CohereRerank(api_key=cohere_api_key, top_n=2)
    elif rerank_method == "colbert":
        try:
            from llama_index.postprocessor.colbert_rerank import ColbertRerank
        except:
            raise "colbertreranker package missiong please install : pip install llama-index-postprocessor-colbert-rerank"
        return ColbertRerank(
            top_n=5,
            model="colbert-ir/colbertv2.0",
            tokenizer="colbert-ir/colbertv2.0",
            keep_retrieval_score=True,
        )
    elif rerank_method == "sentence_transformer":
        from llama_index.core.postprocessor import SentenceTransformerRerank
        return SentenceTransformerRerank(
            model="cross-encoder/ms-marco-MiniLM-L-2-v2", top_n=3
        )
    else:
        return None

@cl.on_chat_start
async def start():
    global index
    if index is None:
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

    rerank_method = cl.user_session.get("rerank_method")
    reranker = get_reranker(rerank_method)

    query_engine = index.as_query_engine(
        streaming=True,
        similarity_top_k=10,
        node_postprocessors=[reranker] if reranker else [],
    )
    cl.user_session.set("query_engine", query_engine)

    await cl.Message(
        author="Assistant", content="Hello! I'm an AI assistant. How may I help you?"
    ).send()

@cl.on_message
async def handle_message(message: cl.Message):
    global index
    if message.elements:
        for file in message.elements:
            if file.type == "file":
                documents = SimpleDirectoryReader(input_files=[file.path]).load_data()
                pipeline = IngestionPipeline(
                    transformations=[
                        SentenceSplitter(chunk_size=1024, chunk_overlap=20),
                        Settings.embed_model,
                    ],
                    vector_store=vector_store,
                )
                nodes = pipeline.run(documents=documents, show_progress=True)
                index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
                
                rerank_method = cl.user_session.get("rerank_method")
                reranker = get_reranker(rerank_method)
                
                query_engine = index.as_query_engine(
                    streaming=True,
                    similarity_top_k=10,
                    node_postprocessors=[reranker] if reranker else [],
                )
                cl.user_session.set("query_engine", query_engine)

                await cl.Message(
                    content=f"Processed {len(nodes)} chunks from the uploaded file."
                ).send()

    query_engine = cl.user_session.get("query_engine")
    msg = cl.Message(content="", author="Assistant")
    res = await cl.make_async(query_engine.query)(message.content)

    for token in res.response_gen:
        await msg.stream_token(token)
    await msg.send()

if __name__ == "__main__":
    import sys
    import subprocess

    parser = argparse.ArgumentParser(description="RAG Script with ingestion and reranking options")
    parser.add_argument("--host", default="localhost", help='Host IP address')
    parser.add_argument("--port", type=int, default=8000, help='Port number')
    parser.add_argument('--ingest', action='store_true', help='Ingest documents before starting the chat')
    parser.add_argument('--data_dir', type=str, default="../data", help='Directory containing documents to ingest')
    parser.add_argument('--rerank', choices=['llm', 'cohere', 'colbert', 'sentence_transformer'], 
                        default=None, help='Choose reranking method')
    args = parser.parse_args()

    if args.ingest:
        ingest_documents(args.data_dir)

    # Set the rerank method in the environment for Chainlit to access
    os.environ["RERANK_METHOD"] = args.rerank if args.rerank else ""

    # Run the Chainlit app with specified host and port
    subprocess.run([
        "chainlit", "run", sys.argv[0],
        "--host", args.host,
        "--port", str(args.port)
    ], check=True)