# RAPTOR: Recursive Abstractive Processing for Tree Organized Retrieval

## Introduction

RAPTOR (Recursive Abstractive Processing for Tree Organized Retrieval) is an advanced approach to Retrieval-Augmented Generation (RAG) that enhances the traditional RAG pipeline by incorporating hierarchical document structuring and summarization.

### Motivation

Traditional RAG systems often struggle with large document sets and complex queries. RAPTOR addresses these challenges by creating a hierarchical representation of the document corpus, allowing for more nuanced and efficient retrieval.

### Method Details

```mermaid
flowchart TB
    subgraph "1. Document Processing"
        A[Documents] --> B[Split Text into Chunks]
        B --> C1[Chunk-1]
        B --> C2[Chunk-2]
        B --> C3[Chunk-n]
    end

    subgraph "2. Document Embedding"
        EM1{{Embedding Model}}
        C1 & C2 & C3 --> EM1
        EM1 --> D1[Embedding-1] & D2[Embedding-2] & D3[Embedding-3]
    end

    subgraph "3. Clustering"
        D1 & D2 & D3 --> E[Clustering Algorithm]
        E --> F1[Cluster-1] & F2[Cluster-2] & F3[Cluster-m]
    end

    subgraph "4. Summarization"
        F1 --> G1[Summary-1]
        F2 --> G2[Summary-2]
        F3 --> G3[Summary-m]
    end

    subgraph "5. Tree Construction"
        G1 & G2 & G3 --> H[Build Hierarchical Tree]
        H --> I[RAPTOR Tree]
    end

    subgraph "6. Query Processing"
        J[Query] --> EM2{{Embedding Model}}
        EM2 --> K[Query Embedding]
    end

    subgraph "7. Tree Traversal"
        K --> L[Traverse RAPTOR Tree]
        I --> L
        L --> M[Relevant Nodes]
    end

    subgraph "8. Context Formation"
        M --> N[Query + Relevant Summaries/Chunks]
    end

    subgraph "9. Generation"
        N --> O[LLM]
        O --> P[Response]
    end

    J --> N
```

#### Document Preprocessing and Vector Store Creation

1. Documents are split into manageable chunks.
2. Each chunk is embedded using a suitable embedding model.
3. Embeddings are clustered to group similar content.
4. Clusters are summarized to create higher-level abstractions.
5. A hierarchical tree structure (RAPTOR Tree) is built using these summaries and original chunks.

#### Retrieval-Augmented Generation Workflow

1. The user query is embedded using the same embedding model.
2. The RAPTOR Tree is traversed to find relevant nodes (summaries or chunks).
3. Relevant content is combined with the original query to form a context.
4. This context is passed to a Large Language Model (LLM) to generate a response.

### Key Features of RAPTOR

- Hierarchical Document Representation: Creates a tree structure of document content.
- Multi-level Summarization: Provides abstractions at various levels of detail.
- Efficient Retrieval: Utilizes tree traversal for faster and more relevant information retrieval.
- Scalability: Better handles large document sets compared to flat vector stores.

### Benefits of this Approach

1. Improved Context Relevance: The hierarchical structure allows for more nuanced matching of queries to relevant content.
2. Efficient Retrieval: Tree traversal can be more efficient than exhaustive search in large vector spaces.
3. Handling Complex Queries: The multi-level structure is better equipped to handle queries that require information from multiple parts of the corpus.
4. Scalability: Can handle larger document sets more effectively than traditional RAG approaches.

### Conclusion

RAPTOR enhances the RAG pipeline by introducing a hierarchical, summary-based approach to document representation and retrieval. This method promises to improve the quality and efficiency of information retrieval, especially for large and complex document sets, leading to more accurate and contextually relevant responses in AI-powered question-answering systems.