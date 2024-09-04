# RAG Observability - Arize Phoenix Setup

Welcome to this notebook, where we explore the setup and observation of a Retrieval-Augmented Generation (RAG) pipeline using Llama Index.

## Table of Contents

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Usage](#usage)
4. [Conclusion](#conclusion)

## Introduction

This guide provides a comprehensive walkthrough for configuring the necessary tools and libraries, including embedding models and vector store indexing, to enable efficient document retrieval and query processing. We’ll cover everything from installation and setup to querying and retrieving relevant information, equipping you with the knowledge to harness the power of RAG pipelines for advanced search capabilities.

## Getting Started

To get started with this notebook, you'll need to have a basic understanding of Python and some familiarity with machine learning concepts. Don't worry if you're new to some of these ideas – we'll guide you through each step!

### Prerequisites

- Python 3.7+
- Jupyter Notebook or JupyterLab
- Basic knowledge of Python and machine learning concepts

## Usage

## 1. Setup

### 1.1 Install required packages

To get started with setting up Arize Phoenix, you'll need to install the required packages.

Arize Phoenix is a comprehensive tool designed for observability and monitoring in machine learning and AI systems. It provides functionalities for tracking and analyzing various aspects of machine learning models and data pipelines.

```bash
!pip install arize-phoenix
!pip install openinference-instrumentation-openai
```

These commands will install:

- `arize-phoenix`: A tool for observability in machine learning workflows.
- `openinference-instrumentation-openai`: A package to instrument OpenAI models with observability tools like Arize Phoenix.

### 1.2 Setting up Arize Phoenix

There are 3 ways to do this:

Read more [here.](https://docs.arize.com/phoenix/quickstart)

- Command Line

  ```bash
  python3 -m phoenix.server.main serve
  ```

- Docker

  Launch the phoenix docker image using:

  ```bash
  docker run -p 6006:6006 -p 4317:4317 arizephoenix/phoenix:latest
  ```

  This will expose the Phoenix UI and REST API on localhost:6006 and exposes the gRPC endpoint for spans on localhost:4317.

- Notebook

  ```python
  import phoenix as px
  px.launch_app()
  ```

### 1.3 Import Required Libraries and Configure the Environment

Before proceeding with data processing and evaluation, import the necessary libraries and set up the environment:

```python
import json
import os
from getpass import getpass
import nest_asyncio
import pandas as pd
from tqdm import tqdm
import phoenix as px

# Allows concurrent evaluations in notebook environments
nest_asyncio.apply()

# Set display options for pandas DataFrames to show more content
pd.set_option("display.max_colwidth", 1000)
```

- `json`, `os`: Standard Python libraries for handling JSON data and operating system interactions.

- `getpass`: A utility for securely capturing password input.
  `nest_asyncio`: Allows the usage of asyncio within Jupyter notebooks.

- `pandas` (`pd`): A powerful data manipulation library for Python.

- `tqdm`: Provides progress bars for loops, useful for tracking the progress of data processing.

- `phoenix` (`px`): The phoenix library is part of Arize's observability tools. It provides an interactive UI for exploring data and monitoring machine learning models.

Configure `nest_asyncio` to allow concurrent evaluations in notebook environments and set the maximum column width for pandas DataFrames to ensure better readability.

### 1.4 Launch the Phoenix App

```python
px.launch_app()
```

This function initializes and launches the Phoenix app, which opens in a new tab in your default web browser. It provides an interactive interface for exploring datasets, visualizing model performance, and debugging.

### 1.5 View the Phoenix App Session

Once the Phoenix app is launched, you can use the session object to interact with the app directly in the notebook. Run the following code to launch the Phoenix app and view it in the current session:

```python
# Launch and view the Phoenix app session
(session := px.launch_app()).view()
```

This line launches the Phoenix app and assigns the session to a variable named session, with the `view()` method allowing you to display the Phoenix app directly within the notebook interface, providing a more integrated experience without switching between the browser and the notebook.

### 1.6 Set Up the Endpoint for Traces

To send traces to the Phoenix app for analysis and observability, define the endpoint URL where the Phoenix app is listening for incoming data.

```python
endpoint = "http://127.0.0.1:6006/v1/traces"
```

The `endpoint` variable stores the URL of the Phoenix app's endpoint that listens for incoming traces.

## 2. Trace Open AI

For more integration, [read.](https://docs.arize.com/phoenix/tracing/integrations-tracing)

### 2.1 Install and Import the OpenAI Package

```bash
!pip install openai
import openai
```

`openai`: The Python client library for OpenAI's API. It enables you to make requests to OpenAI's models, including GPT-3 and GPT-4, for various tasks.

### 2.2 Configure the OpenAI API Key

```python
import openai
import os
from getpass import getpass

# Retrieve API key from environment variable or prompt user if not set
if not (openai_api_key := os.getenv("OPENAI_API_KEY")):
    openai_api_key = getpass("🔑 Enter your OpenAI API key: ")

# Set the API key for the OpenAI client
openai.api_key = openai_api_key

# Store the API key in environment variables for future use
os.environ["OPENAI_API_KEY"] = openai_api_key
```

- Retrieve API Key: The code first attempts to get the API key from an environment variable (OPENAI_API_KEY). If the key is not found, it prompts the user to enter it securely using getpass.

- Set API Key: The retrieved or provided API key is then set for the openai client library.

- Store API Key: Finally, the API key is stored in the environment variables to ensure it is available for future use within the session.

### 2.3 Set Up OpenTelemetry for Tracing

To enable tracing for your OpenAI interactions, configure OpenTelemetry with the necessary components.

```python
from opentelemetry import trace as trace_api
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

# Set up the Tracer Provider
tracer_provider = trace_sdk.TracerProvider()

# Define the OTLP Span Exporter with the endpoint
span_exporter = OTLPSpanExporter(endpoint)

# Set up the Span Processor to process and export spans
span_processor = SimpleSpanProcessor(span_exporter)

# Add the Span Processor to the Tracer Provider
tracer_provider.add_span_processor(span_processor)

# Set the global Tracer Provider
trace_api.set_tracer_provider(tracer_provider)
```

**OpenTelemetry Libraries**

In the provided code, several OpenTelemetry libraries are used to set up tracing. Here's an overview of each:

- `opentelemetry`:

  \***\*Purpose\*\***: The core library for OpenTelemetry, providing APIs for tracing and metrics.

  **Usage**: It includes the trace module, which is used to create and manage traces.

- `opentelemetry.exporter.otlp.proto.http.trace_exporter`:

  **Purpose**: Provides the OTLP (OpenTelemetry Protocol) exporter for traces using HTTP.

  **Usage**: The `OTLPSpanExporter` class in this module sends trace data to an OTLP-compatible backend. This exporter is configured with an endpoint where trace data will be sent.

- `opentelemetry.sdk.trace`:

  **Purpose**: Contains the SDK implementations for tracing, including the `TracerProvider`.

  **Usage**:

  - `TracerProvider`: Manages Tracer instances and is responsible for exporting spans (units of work) collected during tracing.

  - `SimpleSpanProcessor`: A span processor that exports spans synchronously, used to process and send trace data to the exporter.

- `opentelemetry.sdk.trace.export`:

  **Purpose**: Provides classes for exporting trace data.

  **Usage**:

  - `SimpleSpanProcessor`: Processes spans and exports them using the specified exporter. It ensures that spans are sent to the backend for analysis.

### 2.4 Instrument OpenAI with OpenInference

To integrate OpenTelemetry with OpenAI and enable tracing for OpenAI model interactions, use the `OpenAIInstrumentor` from the `openinference` library.

```python
from openinference.instrumentation.openai import OpenAIInstrumentor

# Instantiate and apply instrumentation for OpenAI
OpenAIInstrumentor().instrument()
```

- `OpenAIInstrumentor`: A class from the openinference library designed to instrument OpenAI's API calls, enabling tracing and observability.

- `instrument()`: This method configures the OpenAI API client to automatically generate and send trace data to the OpenTelemetry backend. It integrates with the tracing setup you have configured, allowing you to monitor and analyze interactions with OpenAI's models.

By running this code, you ensure that all OpenAI API calls are traced, allowing you to capture detailed insights into model usage and performance.

### 2.5 Make a Request to OpenAI API

To interact with OpenAI’s API and obtain a response, use the following code. This example demonstrates how to create a chat completion using the OpenAI API and print the result:

```python
import openai

# Create an OpenAI client instance
client = openai.OpenAI()

# Make a request to the OpenAI API for a chat completion
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Write a haiku."}],
)

# Print the content of the response
print(response.choices[0].message.content)
```

- `openai.OpenAI()`: Initializes an OpenAI client instance that can be used to interact with the OpenAI API.

- `client.chat.completions.create()`: Sends a request to the OpenAI API to create a chat completion using the specified model.

  - `model="gpt-4o"`: Specifies the model to use for generating completions. Ensure the model name is correct and available in your OpenAI API account.

  - `messages`: A list of message objects representing the conversation history. In this case, it includes a single message from the user asking to "Write a haiku."

`response.choices[0].message.content`: Extracts and prints the content of the completion response generated by the model.

## 3. Trace Llama index

### 3.1 Install and Import the Required Libraries

```bash
!pip install llama-index
!pip install llama-index-core
!pip install llama-index-llms-openai
!pip install openinference-instrumentation-llama-index==2.2.4
!pip install -U llama-index-callbacks-arize-phoenix
!pip install "arize-phoenix[llama-index]"
```

- `llama-index`: Core package for Llama Index functionality.

- `llama-index-core`: Provides core features and utilities for Llama Index.

- `llama-index-llms-openai`: Integration package for Llama Index and OpenAI models.

- `openinference-instrumentation-llama-index==2.2.4`: Provides instrumentation for tracing Llama Index interactions.

- `llama-index-callbacks-arize-phoenix`: Callback integration for Arize Phoenix with Llama Index.

- `arize-phoenix[llama-index]`: Extends Arize Phoenix to support Llama Index tracing.

### 3.2 Retrieve the URL of the Active Phoenix Session

```python
# Retrieve the URL of the active Phoenix session
px.active_session().url
```

Accesses the current active session of the Phoenix app and retrieves its URL, allowing you to view or share the link to the Phoenix interface where you can monitor and analyze trace data.

### 3.3 Set Up Tracing for Llama Index

To instrument Llama Index for tracing with OpenTelemetry, configure the tracer provider and integrate the Llama Index instrumentor.

```python
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

# Set up the Tracer Provider
tracer_provider = trace_sdk.TracerProvider()

# Add Span Processor to the Tracer Provider
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))

# Instrument Llama Index with the Tracer Provider
LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)
```

- `LlamaIndexInstrumentor`: This class from openinference.instrumentation.llama_index instruments Llama Index to support tracing and observability.

- `trace_sdk.TracerProvider()`: Initializes a new Tracer Provider responsible for creating and managing trace data.
  OTLPSpanExporter(endpoint): Configures the OTLP exporter to send trace data to the specified endpoint.
- `SimpleSpanProcessor`: Processes and exports spans synchronously.

- `tracer_provider.add_span_processor`: Adds the span processor to the Tracer Provider.

- `LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)`: Applies the instrumentation to Llama Index, using the provided Tracer Provider for tracing.

### 3.4 Interact with Llama Index Using OpenAI

To perform a completion request with Llama Index using an OpenAI model, use the following code:

```python
from llama_index.llms.openai import OpenAI

# Initialize the OpenAI model
llm = OpenAI(model="gpt-4o-mini")

# Make a completion request
resp = llm.complete("Paul Graham is ")

# Print the response
print(resp)
```

- `from llama_index.llms.openai import OpenAI`: Imports the OpenAI class from the llama_index package, allowing interaction with OpenAI models.

- `OpenAI(model="gpt-4o-mini")`: Initializes an instance of the OpenAI class with the specified model (e.g., gpt-4).

- `llm.complete(...)`: Sends a prompt to the model to generate a completion based on the input text.

### 3.5 Perform a Chat Interaction with Llama Index Using OpenAI

```python
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage

# Initialize the OpenAI model
llm = OpenAI()

# Define the chat messages
messages = [
    ChatMessage(
        role="system", content="You are a pirate with a colorful personality"
    ),
    ChatMessage(role="user", content="What is your name"),
]

# Get the response from the model
resp = llm.chat(messages)
```

- `OpenAI`: A class for interacting with OpenAI models.

- `ChatMessage`: A class to format chat messages.

- `OpenAI()`: Initializes an instance of the OpenAI class.

- `ChatMessage`: Creates chat message objects with a specified role (e.g., "system", "user") and content.

  - `role="system"`: Defines the system message to set the context or personality of the model.
  - `role="user"`: Represents a user message in the chat.

- `llm.chat(messages)`: Sends the defined messages to the model and retrieves the response.

This code sets up a chat with an OpenAI model, specifying system and user messages to guide the interaction.

## 4. Observe RAG Pipeline

### 4.1 Setup an environment for observing a RAG piepline

```bash
!pip install llama-index
!pip install llama-index-vector-stores-qdrant
!pip install llama-index-readers-file
!pip install llama-index-embeddings-fastembed
!pip install llama-index-llms-openai
!pip install -U qdrant_client fastembed
```

- `llama-index`: Core package for Llama Index functionality.

- `llama-index-vector-stores-qdrant`: Integration for using Qdrant as a vector store with Llama Index.

- `llama-index-readers-file`: Provides file reading capabilities for Llama Index.

- `llama-index-embeddings-fastembed`: Adds FastEmbed support for generating embeddings with Llama Index.

- `llama-index-llms-openai`: Integration for using OpenAI models with Llama Index.

- `qdrant_client`: Client library for interacting with Qdrant, a vector search engine.

- `fastembed`: Library for generating embeddings quickly.

### 4.2 Prepare RAG Pipeline with Embeddings and Document Indexing

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.fastembed import FastEmbedEmbedding
# from llama_index.embeddings.openai import OpenAIEmbedding

from llama_index.core.settings import Settings

Settings.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-base-en-v1.5")
# Settings.embed_model = OpenAIEmbedding(embed_batch_size=10)

documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)
```

- `VectorStoreIndex`: A class used to create and manage a vector store index. This index allows for efficient similarity search and retrieval based on vector representations of documents.

- `SimpleDirectoryReader`: A class for loading documents from a specified directory. It reads and preprocesses files from the directory "sample_data" to be used in the indexing process.

- `FastEmbedEmbedding`: Provides an embedding model for generating vector representations of text using the FastEmbed library. The model specified ("BAAI/bge-base-en-v1.5") is used to convert documents into embeddings, which are then used for similarity search within the vector store index.

- `from llama_index.embeddings.openai import OpenAIEmbedding`:

  `OpenAIEmbedding`: (Commented out) Provides an embedding model for generating vector representations using OpenAI’s embeddings. Uncomment this line if you wish to use OpenAI’s model instead of FastEmbed. This model can be configured with parameters like `embed_batch_size` for batch processing.

- `Settings`: A configuration class used to set global settings for embedding models. By assigning the embed_model attribute, you specify which embedding model to use for the RAG pipeline.

- `Settings.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-base-en-v1.5")`
  Configures the Settings class to use the FastEmbed model for generating embeddings. This step is crucial for defining how text data will be represented in the vector store.

- `documents = SimpleDirectoryReader("data").load_data()`
  Loads and preprocesses documents (in this case) from the "data" directory. Please ensure to tweak the directory name according to your project. The `load_data()` method reads all files in the specified directory and prepares them for indexing.

- `index = VectorStoreIndex.from_documents(documents)`
  Creates a VectorStoreIndex from the preprocessed documents. This index allows for efficient querying and retrieval based on the vector representations generated by the embedding model.

### 4.3 Query the Vector Store Index

Once the vector store index is set up, you can use it to perform queries and retrieve relevant information.

```python
query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")
print(response)
```

- `as_query_engine()`: Converts the `VectorStoreIndex` into a query engine. This engine allows you to perform searches and retrieve information based on the vector representations of documents stored in the index.

- `query()`: Executes a query against the vector store index. The query string "What did the author do growing up?" is used to search for relevant documents and retrieve information based on the context provided by the vector embeddings.

Finally, the `response` containing the information retrieved from the vector store index, which is based on the query and the indexed documents is output.

## Conclusion

In this guide, we have set up a Retrieval-Augmented Generation (RAG) pipeline using Llama Index and integrated it with various components to observe its functionality. We began by configuring and installing the necessary libraries, including Llama Index, OpenTelemetry, and various embedding models.

We then proceeded to:

- Initialize and configure the embedding models, using FastEmbed or OpenAI models as needed.
- Load and index documents from a directory to prepare the data for querying.
- Set up a query engine to perform searches and retrieve relevant information based on the indexed documents.

By following these steps, you have successfully prepared a RAG pipeline capable of efficient document retrieval and query processing. This setup enables advanced search and information retrieval capabilities, leveraging the power of vector-based embeddings and indexing.

Feel free to experiment with different configurations and queries to further explore the capabilities of the RAG pipeline. For any questions or additional customization, consult the documentation of the libraries used or seek further guidance.

If you find this guide helpful, please consider giving us a star on GitHub! ⭐

[![GitHub stars](https://img.shields.io/github/stars/adithya-s-k/AI-Engineering.academy.svg?style=social&label=Star&maxAge=2482000)](https://github.com/adithya-s-k/AI-Engineering.academy)

Thank you for following this guide, and happy querying!
