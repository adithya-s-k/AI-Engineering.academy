# 🚀 RAG (Retrieval Augmented Generation) System Implementation Guide

Welcome to the comprehensive guide for implementing RAG systems! This repository provides a structured approach to building and optimizing Retrieval Augmented Generation systems, from basic implementations to advanced techniques.

## 📚 Repository Structure

### Core Modules

#### Fundamentals

1. [**RAG from Scratch**](./00_RAG_from_Scratch/)
   - Complete implementation guide from ground up
   - RAG in 10 lines of code
   - Understanding embeddings and similarity
   - Basic requirements setup

#### Basic Implementation & Evaluation

2. [**Basic RAG Implementation**](./01_Basic_RAG/)

   - Basic server implementation
   - Jupyter notebook tutorials
   - Performance evaluation notebooks
   - Environment setup guide

3. [**BM25 RAG**](./01_BM25_RAG/)

   - BM25 algorithm implementation
   - Application setup
   - Interactive notebook examples

4. [**Data Ingestion**](./01_Data_Ingestion/)

   - Data chunking strategies
   - Embedding generation
   - Batch processing examples
   - Data parsing techniques

5. [**RAG Evaluation**](./01_RAG_Evaluation/)

   - RAGAS metrics implementation
   - Deepeval integration
   - TruLens evaluation
   - Test dataset examples

6. [**RAG Observability**](./01_RAG_Observability/)
   - System monitoring setup
   - Performance tracking
   - Debug tools integration

#### Advanced Techniques

7. [**ReRanker RAG**](./02_ReRanker_RAG/)

   - Result re-ranking implementation
   - Evaluation metrics
   - Performance optimization

8. [**Hybrid RAG**](./03_Hybrid_RAG/)

   - Qdrant hybrid search implementation
   - Multiple retrieval method integration

9. [**Sentence Window RAG**](./04_Sentence_Window_RAG/)

   - Context window optimization
   - Sentence-level retrieval

10. [**Auto Merging RAG**](./05_Auto_Merging_RAG/)

    - Automatic content merging
    - Redundancy elimination

11. [**Advanced Query Processing**](./06_Query_Transformation_RAG/)
    - HyDE (Hypothetical Document Embeddings)
    - Query transformation techniques
    - Query optimization strategies

#### Specialized Implementations

12. [**Self Query RAG**](./07_Self_Query_RAG/)

    - Self-querying mechanisms
    - Query refinement techniques

13. [**RAG Fusion**](./08_RAG_Fusion/)

    - Multiple RAG model integration
    - Result fusion strategies

14. [**RAPTOR**](./09_RAPTOR/)

    - Advanced reasoning implementation
    - Performance optimization

15. [**ColBERT RAG**](./10_ColBERT_RAG/)

    - ColBERT model integration
    - Ragatouille retriever implementation

16. [**Graph RAG**](./11_Graph_RAG/)

    - Graph-based retrieval
    - Knowledge graph integration

17. [**Agnetic RAG**](./12_Agnetic_RAG/)

    - Multi-document agent system
    - Domain-specific implementations

18. [**Vision RAG**](./13_Vision_RAG/)
    - GPT-4V integration
    - Multi-modal retrieval implementation

### 📂 Data Resources

Located in the [`data/`](./data/) directory:

- **Markdown Documents** (`md/`): Processed markdown versions of papers
- **PDF Documents** (`pdf/`): Original research papers and documentation
- **Sample Database** (`sample-lancedb/`): Example database implementation

## 🎯 Implementation Techniques

### ✅ Implemented Features

1. Simple RAG with vector store integration
2. Context enrichment algorithms
3. Multi-faceted filtering systems
4. Fusion retrieval mechanisms
5. Intelligent reranking
6. Query transformation
7. Hierarchical indexing
8. HyDE implementation
9. Dynamic chunk sizing
10. Semantic chunking
11. Context compression
12. Explainable retrieval
13. Graph RAG implementation
14. RAPTOR integration

### 🚧 Upcoming Features

1. Retrieval with feedback loops
2. Adaptive retrieval systems
3. Iterative retrieval mechanisms
4. Ensemble retrieval implementation
5. Multi-modal integration
6. Self RAG optimization
7. Corrective RAG systems

## 🛠️ Tech Stack

- 🦙 **RAG Orchestration:** Llama-index
- 🔍 **Vector Database:** Qdrant
- 👁️ **Observability:** Arize Phoenix
- 📊 **Evaluation:** RAGAS & Deepeval

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines for more information.

## 📚 References

This project builds upon research and implementations from various sources. See our acknowledgments section for detailed credits.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

<div align="center">
  Made with ❤️ for the RAG community
</div>

<!-- # 🚀 RAG (Retrieval Augmented Generation) System Roadmap

Welcome to the RAG System Roadmap! RAG systems are gaining traction in natural language processing by enhancing the quality of AI responses with relevant data retrieval. This roadmap outlines the exciting journey to systematically improve a RAG system using the latest technologies, methods, and strategies.

## Tech Stack

- 🦙 **RAG Orchestration:** Llama-index
- 🔍 **Vector Database:** Qdrant
- 👁️ **Observability:** Arize Phoenix
- 📊 **Evaluation:** RAGAS & Deepeval


### Modules

| ⭐ Module Name | 📝 Description |
|-------------|-------------|
| [00_RAG_from_Scratch](./00_RAG_from_Scratch) | A comprehensive guide or approach to building RAG systems from the ground up, covering all aspects of RAG implementation. |
| [01_Basic_RAG](./01_Basic_RAG) | A fundamental implementation of retrieval-augmented generation, combining a retrieval system with a language model. |
| [01_BM25_RAG](./01_BM25_RAG) | Utilizes the BM25 algorithm for retrieval, effective for keyword-based retrieval in RAG systems. |
| [01_Data_Ingestion](./01_Data_Ingestion) | Focuses on preparing and importing data into the system, including cleaning, formatting, and indexing. |
| [01_RAG_Evaluation](./01_RAG_Evaluation) | Methods and tools to assess RAG model performance, including metrics for retrieval accuracy and generation quality. |
| [01_RAG_Observability](./01_RAG_Observability) | Techniques for monitoring and analyzing RAG system behavior in real-time, tracking performance metrics and identifying bottlenecks. |
| [02_ReRanker_RAG](./02_ReRanker_RAG) | Enhances retrieval accuracy by re-ranking initial results using a separate model or algorithm. |
| [03_Hybrid_RAG](./03_Hybrid_RAG) | Combines multiple retrieval methods to leverage their respective strengths, such as dense and sparse retrieval techniques. |
| [04_Sentence_Window_RAG](./04_Sentence_Window_RAG) | Optimizes retrieval by focusing on relevant sentence contexts instead of entire documents. |
| [05_Auto_Merging_RAG](./05_Auto_Merging_RAG) | Automates the process of combining retrieved documents or passages, reducing redundancy and creating a more coherent context. |
| [06_HyDE_RAG](./06_HyDE_RAG) | Implements Hypothetical Document Embedding to improve query understanding and retrieval relevance. |
| [06_Query_Transformation_RAG](./06_Query_Transformation_RAG) | Applies various techniques to reformulate and expand queries, improving retrieval by addressing potential mismatches. |
| [07_Self_Query_RAG](./07_Self_Query_RAG) | A sophisticated approach where the system generates and refines its own queries based on the initial user input. |
| [08_RAG_Fusion](./08_RAG_Fusion) | Integrates results from multiple RAG models or retrieval methods for more comprehensive and accurate responses. |
| [09_RAPTOR](./09_RAPTOR) | An advanced RAG model that incorporates improved reasoning capabilities. |
| [10_ColBERT_RAG](./10_ColBERT_RAG) | Utilizes the ColBERT model for dense retrieval, enabling efficient and effective retrieval through dense vector representations. |
| [11_Graph_RAG](./11_Graph_RAG) | Leverages graph-based methods for complex information retrieval, capturing relationships and connections in knowledge graphs. |
| [12_Agnetic_RAG](./12_Agnetic_RAG) | A domain-specific approach for tailored RAG implementations, adapting systems to specific use cases. |
### Data

- [data](./data/): Contains datasets and other data files used throughout the project.

## Systematically Improving RAG

To enhance the performance of our RAG system, we will focus on the following areas:

1. **Data Quality**: Ensure high-quality, diverse, and relevant data for training and retrieval.
2. **Embedding Techniques**: Experiment with different embedding models and fine-tuning approaches.
3. **Retrieval Optimization**: Improve the retrieval process using techniques like hybrid search or re-ranking.
4. **Context Window Management**: Optimize the use of context windows for more effective generation.
5. **Prompt Engineering**: Develop and refine prompts to guide the LLM effectively.
6. **Fine-tuning**: Explore domain-specific fine-tuning of the LLM when applicable.
7. **Evaluation and Metrics**: Utilize RAGAS and Deepeval to assess performance and guide improvements.
8. **Observability**: Leverage Arize Phoenix to monitor system behavior and identify areas for optimization.
9. **Iterative Testing**: Continuously test and refine the system based on real-world usage and feedback.
10. **Scalability**: Optimize the system architecture to handle increased load and data volume efficiently.


## List of RAG Techniques

1. ✅**Simple RAG**: Encode document content into a vector store for quick retrieval.
2. ✅**Context Enrichment**: Add surrounding context to each retrieved chunk.
3. ✅**Multi-faceted Filtering**: Apply various filtering techniques to refine results.
4. ✅**Fusion Retrieval**: Combine vector-based and keyword-based retrieval.
5. ✅**Intelligent Reranking**: Reassess and reorder initially retrieved documents.
6. ✅**Query Transformation**: Modify or expand the original query.
7. ✅**Hierarchical Indices**: Use summaries to identify relevant document sections.
8. ✅**Hypothetical Questions**: Transform queries into hypothetical documents (HyDE).
9. ✅**Choose Chunk Size**: Select appropriate fixed size for text chunks.
10. ✅**Semantic Chunking**: Create context-aware segments.
11. ✅**Context Compression**: Compress and extract pertinent parts of documents.
12. ✅**Explainable Retrieval**: Provide explanations for document relevance.
13. **Retrieval w/ Feedback**: Utilize user feedback to fine-tune models.
14. **Adaptive Retrieval**: Use tailored strategies for different query types.
15. **Iterative Retrieval**: Generate follow-up queries to fill information gaps.
16. **Ensemble Retrieval**: Apply different models and use voting mechanisms.
17. ✅**Graph RAG**: Retrieve entities and relationships from a knowledge graph.
18. **Multi-Modal**: Integrate models that understand different data modalities.
19. ✅**RAPTOR**: Use abstractive summarization for hierarchical context.
20. **Self RAG**: Implement multi-step processes for improved responses.
21. **Corrective RAG**: Dynamically evaluate and correct the retrieval process. -->
