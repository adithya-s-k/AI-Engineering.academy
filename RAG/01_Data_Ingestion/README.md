# Data Chunking for RAG Systems

## Introduction

Data chunking is a crucial step in Retrieval-Augmented Generation (RAG) systems. It involves breaking down large documents into smaller, manageable pieces that can be efficiently indexed, retrieved, and processed. This README provides an overview of various chunking methods that can be used in RAG pipelines.

## Importance of Chunking in RAG

Effective chunking is essential for RAG systems because it:
1. Improves retrieval accuracy by creating coherent, self-contained units of information.
2. Enhances the efficiency of embedding generation and similarity search.
3. Allows for more precise context selection when generating responses.
4. Helps manage token limits in language models and embedding systems.

## Chunking Methods

We have implemented six different chunking methods, each with its own strengths and use cases:

1. RecursiveCharacterTextSplitter
2. TokenTextSplitter
3. KamradtSemanticChunker
4. KamradtModifiedChunker
5. ClusterSemanticChunker
6. LLMSemanticChunker

## Chunking Workflows

### 1. RecursiveCharacterTextSplitter

```mermaid
flowchart TB
    A[Document] --> B[Split by separators]
    B --> C["Priority: <br/>\n\n, \n, ., ?, !, space"]
    C --> D[Merge splits until max length]
    D --> E[Optional: Add chunk overlap]
```

### 2. TokenTextSplitter

```mermaid
flowchart TB
    A[Document] --> B[Tokenize text]
    B --> C[Split by fixed token count]
    C --> D[Ensure splits at token boundaries]
    D --> E[Optional: Add chunk overlap]
```

### 3. KamradtSemanticChunker

```mermaid
flowchart TB
    A[Document] --> B[Split by sentence]
    B --> C[Compute embeddings<br/>for sliding window]
    C --> D[Calculate cosine distances<br/>between consecutive windows]
    D --> E[Find discontinuities<br/>> 95th percentile]
    E --> F[Split at discontinuities]
```

### 4. KamradtModifiedChunker

```mermaid
flowchart TB
    A[Document] --> B[Split by sentence]
    B --> C[Compute embeddings<br/>for sliding window]
    C --> D[Calculate cosine distances<br/>between consecutive windows]
    D --> E[Binary search for<br/>optimal threshold]
    E --> F[Ensure largest chunk<br/>< specified length]
    F --> G[Split at determined<br/>discontinuities]
```

### 5. ClusterSemanticChunker

```mermaid
flowchart TB
    A[Document] --> B[Split into 50-token pieces]
    B --> C[Compute embeddings<br/>for each piece]
    C --> D[Calculate pairwise<br/>cosine similarities]
    D --> E[Use dynamic programming<br/>to maximize similarity]
    E --> F[Ensure chunks <= max length]
    F --> G[Merge pieces into<br/>optimal chunks]
```

### 6. LLMSemanticChunker

```mermaid
flowchart TB
    A[Document] --> B[Split into 50-token pieces]
    B --> C[Surround with<br/><start_chunk_X> tags]
    C --> D[Prompt LLM with tagged text]
    D --> E[LLM returns split indexes]
    E --> F[Process indexes to<br/>create chunks]
    F --> G[Ensure chunks <= max length]
```

## Method Descriptions

1. **RecursiveCharacterTextSplitter**: Splits text based on a hierarchy of separators, prioritizing natural breaks in the document.

2. **TokenTextSplitter**: Splits text into chunks of a fixed number of tokens, ensuring that splits occur at token boundaries.

3. **KamradtSemanticChunker**: Uses sliding window embeddings to identify semantic discontinuities and split the text accordingly.

4. **KamradtModifiedChunker**: An improved version of KamradtSemanticChunker that uses binary search to find an optimal threshold for splitting.

5. **ClusterSemanticChunker**: Splits text into small pieces, computes embeddings, and uses dynamic programming to create optimal chunks based on semantic similarity.

6. **LLMSemanticChunker**: Utilizes a language model to determine appropriate split points in the text.

## Usage

To use these chunking methods in your RAG pipeline:

1. Import the desired chunker from the `chunkers` module.
2. Initialize the chunker with appropriate parameters (e.g., max chunk size, overlap).
3. Pass your document through the chunker to obtain the chunks.

Example:

```python
from chunkers import RecursiveCharacterTextSplitter

chunker = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = chunker.split_text(your_document)
```

## Choosing a Chunking Method

The choice of chunking method depends on your specific use case:

- For simple text splitting, use RecursiveCharacterTextSplitter or TokenTextSplitter.
- For semantic-aware splitting, consider KamradtSemanticChunker or KamradtModifiedChunker.
- For more advanced semantic chunking, use ClusterSemanticChunker or LLMSemanticChunker.

Factors to consider when choosing a method:
- Document structure and content type
- Desired chunk size and overlap
- Computational resources available
- Specific requirements of your retrieval system (e.g., vector vs. keyword-based)

Experiment with different methods to find the one that works best for your documents and retrieval needs.

## Integration with RAG Systems

After chunking, typically you would:
1. Generate embeddings for each chunk (for vector-based retrieval systems).
2. Index the chunks in your chosen retrieval system (e.g., vector database, inverted index).
3. Use the indexed chunks in your retrieval step when answering queries.

## Contributing

We welcome contributions to improve existing chunking methods or add new ones. Please refer to our contributing guidelines for more information.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

``` mermaid
flowchart TB
    Document --> RecursiveCharacterTextSplitter & TokenTextSplitter & KamradtSemanticChunker & KamradtModifiedChunker & ClusterSemanticChunker & LLMSemanticChunker

    subgraph "1. RecursiveCharacterTextSplitter"
        RecursiveCharacterTextSplitter --> RCS1[Split by separators]
        RCS1 --> RCS2["Priority: <br/>\n\n, \n, ., ?, !, space"]
        RCS2 --> RCS3[Merge splits until max length]
        RCS3 --> RCS4[Optional: Add chunk overlap]
    end

    subgraph "2. TokenTextSplitter"
        TokenTextSplitter --> TTS1[Tokenize text]
        TTS1 --> TTS2[Split by fixed token count]
        TTS2 --> TTS3[Ensure splits at token boundaries]
        TTS3 --> TTS4[Optional: Add chunk overlap]
    end

    subgraph "3. KamradtSemanticChunker"
        KamradtSemanticChunker --> KSC1[Split by sentence]
        KSC1 --> KSC2[Compute embeddings<br/>for sliding window]
        KSC2 --> KSC3[Calculate cosine distances<br/>between consecutive windows]
        KSC3 --> KSC4[Find discontinuities<br/>> 95th percentile]
        KSC4 --> KSC5[Split at discontinuities]
    end

    subgraph "4. KamradtModifiedChunker"
        KamradtModifiedChunker --> KMC1[Split by sentence]
        KMC1 --> KMC2[Compute embeddings<br/>for sliding window]
        KMC2 --> KMC3[Calculate cosine distances<br/>between consecutive windows]
        KMC3 --> KMC4[Binary search for<br/>optimal threshold]
        KMC4 --> KMC5[Ensure largest chunk<br/>< specified length]
        KMC5 --> KMC6[Split at determined<br/>discontinuities]
    end

    subgraph "5. ClusterSemanticChunker"
        ClusterSemanticChunker --> CSC1[Split into 50-token pieces]
        CSC1 --> CSC2[Compute embeddings<br/>for each piece]
        CSC2 --> CSC3[Calculate pairwise<br/>cosine similarities]
        CSC3 --> CSC4[Use dynamic programming<br/>to maximize similarity]
        CSC4 --> CSC5[Ensure chunks <= max length]
        CSC5 --> CSC6[Merge pieces into<br/>optimal chunks]
    end

    subgraph "6. LLMSemanticChunker"
        LLMSemanticChunker --> LSC1[Split into 50-token pieces]
        LSC1 --> LSC2[Surround with<br/><start_chunk_X> tags]
        LSC2 --> LSC3[Prompt LLM with tagged text]
        LSC3 --> LSC4[LLM returns split indexes]
        LSC4 --> LSC5[Process indexes to<br/>create chunks]
        LSC5 --> LSC6[Ensure chunks <= max length]
    end

    %% Force diagram to render left-to-right
    RecursiveCharacterTextSplitter ~~~ TokenTextSplitter ~~~ KamradtSemanticChunker ~~~ KamradtModifiedChunker ~~~ ClusterSemanticChunker ~~~ LLMSemanticChunker
```