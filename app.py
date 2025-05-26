import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

def main():
    load_dotenv()
    HF_API_TOKEN = os.getenv("HF_API_TOKEN")
    
    print("Loading PDF from docs/sample.pdf...")
    loader = PyPDFLoader("docs/sample.pdf")
    pages = loader.load()
    total_chars = sum(len(page.page_content) for page in pages)
    print(f"Extracted {total_chars} characters from PDF.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(pages)
    print(f"Split text into {len(docs)} chunks.")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(docs, embeddings)
    retriever = vectordb.as_retriever()

    # Setup HuggingFace pipeline with max_new_tokens to avoid length errors
    hf_pipeline = pipeline(
        "text-generation",
        model="gpt2",
        tokenizer="gpt2",
        max_new_tokens=100,
        temperature=0.7,
    )
    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
    )

    # Interactive chat loop
    while True:
        query = input("\nAsk a question about the document (or type 'exit' to quit): ")
        if query.lower() in ['exit', 'quit']:
            print("Exiting chat. Goodbye!")
            break

        print(f"\n\nAsking: {query}\n\n")
        answer = qa.invoke(query)
        print(f"\n\nAnswer: {answer['result']}\n\n")

if __name__ == "__main__":
    main()
