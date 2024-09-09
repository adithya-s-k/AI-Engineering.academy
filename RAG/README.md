# 🚀 RAG (Retrieval Augmented Generation) System Roadmap

Welcome to the RAG System Roadmap! RAG systems are gaining traction in natural language processing by enhancing the quality of AI responses with relevant data retrieval. This roadmap outlines the exciting journey to systematically improve a RAG system using the latest technologies, methods, and strategies.

## Tech Stack

- 🦙 **RAG Orchestration:** Llama-index (because who doesn't love a good llama?)
- 🔍 **Vector Database:** Qdrant (for when you need to find a needle in a digital haystack)
- 👁️ **Observability:** Arize Phoenix (rise from the ashes of system errors!)
- 📊 **Evaluation:** RAGAS & Deepeval (because even AIs need report cards)


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
21. **Corrective RAG**: Dynamically evaluate and correct the retrieval process.
