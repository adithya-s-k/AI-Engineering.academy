```mermaid
flowchart TD
    subgraph "1. Document Processing"
        A[Documents] --> B[Split Text into Chunks]
        B --> C1[Chunk-1]
        B --> C2[Chunk-2]
        B --> C3[Chunk-n]
    end
    
    subgraph "2. Dual Embedding"
        C1 & C2 & C3 --> DEM{{Dense Embedding Model}}
        C1 & C2 & C3 --> SEM{{Sparse Embedding Model}}
        DEM --> DE1[Dense Embedding-1] & DE2[Dense Embedding-2] & DE3[Dense Embedding-n]
        SEM --> SE1[Sparse Embedding-1] & SE2[Sparse Embedding-2] & SE3[Sparse Embedding-n]
    end
    
    subgraph "3. Unified Indexing"
        DE1 & SE1 --> UI1[Unified Index Entry-1]
        DE2 & SE2 --> UI2[Unified Index Entry-2]
        DE3 & SE3 --> UI3[Unified Index Entry-n]
        UI1 & UI2 & UI3 --> UDB[(Unified VectorDB)]
    end
    
    subgraph "4. Query Processing"
        Q[Query] --> QDEM{{Dense Embedding Model}}
        Q --> QSEM{{Sparse Embedding Model}}
        QDEM --> QDE[Query Dense Embedding]
        QSEM --> QSE[Query Sparse Embedding]
    end
    
    subgraph "5. Two-Step Hybrid Retrieval"
        QSE --> SS{Sparse Search}
        SS --> UDB
        UDB -->|Top K1 Sparse Results| SR[Sparse Retrieved Chunks]
        SR --> DS{Dense Search}
        QDE --> DS
        DS -->|Top K2 Dense Results| DR[Final Retrieved Chunks]
    end
    
    subgraph "6. Context Formation"
        DR --> CF[Query + Retrieved Chunks]
    end
    
    subgraph "7. Generation"
        CF --> LLM[LLM]
        LLM --> R[Response]
    end
    
    Q --> CF
```

# Sentence Window Retriever-Based RAG Approach

## Introduction

The Sentence Window Retriever-Based RAG (Retrieval-Augmented Generation) approach is an advanced implementation of the RAG framework, designed to enhance the context-awareness and coherence of AI-generated responses. This method combines the power of large language models with efficient information retrieval techniques, providing a robust solution for generating high-quality, context-rich responses.

### Motivation

Traditional RAG systems often struggle with maintaining coherence across larger contexts or when dealing with information that spans multiple chunks of text. The Sentence Window Retriever-Based approach addresses this limitation by preserving the contextual relationships between chunks during the indexing process and leveraging this information during retrieval and generation.

### Method Details

#### Document Preprocessing and Vector Store Creation

1. Document Splitting: The input document is split into sentences.
2. Chunk Creation: Sentences are grouped into manageable chunks.
3. Embedding: Each chunk is processed through an embedding model to create vector representations.
4. Vector Database Indexing: Chunk IDs, text, and embeddings are stored in a vector database for efficient similarity search.
5. Document Structure Indexing: A separate database stores the relationships between chunks, including references to previous and next k chunks for each chunk.

#### Retrieval-Augmented Generation Workflow

1. Query Processing: The user query is embedded using the same embedding model used for chunks.
2. Similarity Search: The query embedding is used to find the most relevant chunks in the vector database.
3. Context Expansion: For each retrieved chunk, the system fetches the previous and next k chunks using the document structure database.
4. Context Formation: The retrieved chunks and their expanded context are combined with the original query.
5. Generation: The expanded context and query are passed to a large language model to generate a response.

### Flow Chart

The following flow chart illustrates the Sentence Window Retriever-Based RAG approach:

```mermaid
flowchart TD
    subgraph "1. Document Processing"
        A[Document] --> B[Split into Sentences]
        B --> C[Group Sentences into Chunks]
    end

    subgraph "2. Chunk Processing"
        C --> D1[Chunk 1]
        C --> D2[Chunk 2]
        C --> D3[Chunk 3]
        C --> D4[...]
        C --> Dn[Chunk n]
    end

    subgraph "3. Embedding"
        D1 & D2 & D3 & D4 & Dn --> E{Embedding Model}
        E --> F1[Embedding 1]
        E --> F2[Embedding 2]
        E --> F3[Embedding 3]
        E --> F4[...]
        E --> Fn[Embedding n]
    end

    subgraph "4. Indexing"
        F1 & F2 & F3 & F4 & Fn --> G[(Vector Database)]
        G -->|Store| H[Chunk ID]
        G -->|Store| I[Chunk Text]
        G -->|Store| J[Embedding]
    end

    subgraph "5. Document Structure Store"
        C --> K[(Document Structure DB)]
        K -->|Store| L[Chunk ID]
        K -->|Store| M[Previous k Chunk IDs]
        K -->|Store| N[Next k Chunk IDs]
    end

    subgraph "6. Retrieval"
        O[Query] --> P{Embedding Model}
        P --> Q[Query Embedding]
        Q --> R{Similarity Search}
        R --> G
        G --> S[Retrieved Chunks]
    end

    subgraph "7. Context Expansion"
        S --> T{Expand Context}
        T --> K
        K --> U[Previous k Chunks]
        K --> V[Next k Chunks]
        S & U & V --> W[Expanded Context]
    end

    subgraph "8. Generation"
        W --> X[Query + Expanded Context]
        X --> Y[LLM]
        Y --> Z[Response]
    end
```

### Key Features of RAG

- Efficient Retrieval: Utilizes vector similarity search for fast and accurate information retrieval.
- Context Preservation: Maintains document structure and chunk relationships during indexing.
- Flexible Context Window: Allows for adjustable context expansion at retrieval time.
- Scalability: Capable of handling large document collections and diverse query types.

### Benefits of this Approach

1. Improved Coherence: By including surrounding chunks, the system can generate more coherent and contextually accurate responses.
2. Reduced Hallucination: Access to expanded context helps the model ground its responses in retrieved information, reducing the likelihood of generating false or irrelevant content.
3. Efficient Storage: Only stores necessary information in the vector database, optimizing storage usage.
4. Adaptable Context: The size of the context window can be adjusted based on the specific needs of different queries or applications.
5. Preservation of Document Structure: Maintains the original structure and flow of the document, allowing for more nuanced understanding and generation.

### Conclusion

The Sentence Window Retriever-Based RAG approach offers a powerful solution for enhancing the quality and contextual relevance of AI-generated responses. By preserving document structure and allowing for flexible context expansion, this method addresses key limitations of traditional RAG systems. It provides a robust framework for building advanced question-answering, document analysis, and content generation applications.

