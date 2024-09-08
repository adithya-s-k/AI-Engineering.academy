# RAG (Retrieval Augmented Generation) System

RAG has become increasingly popular in the field of natural language processing and machine learning. This README outlines the tech stack and approach for systematically improving a RAG system.

## Tech Stack

### RAG Orchestration Framework
- **Llama-index**

### Vector Database
- **Qdrant**

### Observability
- **Arize Phoenix**

### Evaluation
- **RAGAS**
- **Deepeval**

### Language Models
- **LLM of Choice:** GPT-4-mini
- **Embedding Model:** text-embedding-3-small

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

## Repository Structure

### Modules

| Module Name | Description |
|-------------|-------------|
| [00_RAG_from_Scratch](./00_RAG_from_Scratch) | A step-by-step guide to building RAG from scratch |
| [01_Basic_RAG](./01_Basic_RAG) | Introduction to a basic RAG setup |
| [01_BM25_RAG](./01_BM25_RAG) | Implementation of RAG using BM25 for retrieval |
| [01_Data_Ingestion](./01_Data_Ingestion) | Guidelines and scripts for data ingestion |
| [01_RAG_Evaluation](./01_RAG_Evaluation) | Methods and tools to evaluate RAG models |
| [01_RAG_Observability](./01_RAG_Observability) | Techniques to monitor RAG model performance |
| [02_ReRanker_RAG](./02_ReRanker_RAG) | Enhancing retrieval accuracy using re-ranking |
| [03_Hybrid_RAG](./03_Hybrid_RAG) | Combining various retrieval methods |
| [04_Sentence_Window_RAG](./04_Sentence_Window_RAG) | Optimizing retrieval with sentence window techniques |
| [05_Auto_Merging_RAG](./05_Auto_Merging_RAG) | Automating the merging of retrieved documents |
| [06_HyDE_RAG](./06_HyDE_RAG) | Implementing Hypothetical Document Embedding |
| [06_Query_Transformation_RAG](./06_Query_Transformation_RAG) | Techniques to transform queries |
| [07_Self_Query_RAG](./07_Self_Query_RAG) | A self-querying approach to improve RAG systems |
| [08_RAG_Fusion](./08_RAG_Fusion) | Integrating multiple RAG models and results |
| [09_RAPTOR](./09_RAPTOR) | Introduction to the RAPTOR model |
| [10_ColBERT_RAG](./10_ColBERT_RAG) | Using ColBERT for effective dense retrieval |
| [11_Graph_RAG](./11_Graph_RAG) | Leveraging graph-based methods for RAG |
| [12_Agnetic_RAG](./12_Agnetic_RAG) | The Agnetic approach for domain-specific RAG |

### Data

- [data](../data/): Contains datasets and other data files used throughout the project.

## Strategies for Enhancing RAG

1. **Simple RAG**: Encode document content into a vector store for quick retrieval.
2. **Context Enrichment**: Add surrounding context to each retrieved chunk.
3. **Multi-faceted Filtering**: Apply various filtering techniques to refine results.
4. **Fusion Retrieval**: Combine vector-based and keyword-based retrieval.
5. **Intelligent Reranking**: Reassess and reorder initially retrieved documents.
6. **Query Transformation**: Modify or expand the original query.
7. **Hierarchical Indices**: Use summaries to identify relevant document sections.
8. **Hypothetical Questions**: Transform queries into hypothetical documents (HyDE).
9. **Choose Chunk Size**: Select appropriate fixed size for text chunks.
10. **Semantic Chunking**: Create context-aware segments.
11. **Context Compression**: Compress and extract pertinent parts of documents.
12. **Explainable Retrieval**: Provide explanations for document relevance.
13. **Retrieval w/ Feedback**: Utilize user feedback to fine-tune models.
14. **Adaptive Retrieval**: Use tailored strategies for different query types.
15. **Iterative Retrieval**: Generate follow-up queries to fill information gaps.
16. **Ensemble Retrieval**: Apply different models and use voting mechanisms.
17. **Graph RAG**: Retrieve entities and relationships from a knowledge graph.
18. **Multi-Modal**: Integrate models that understand different data modalities.
19. **RAPTOR**: Use abstractive summarization for hierarchical context.
20. **Self RAG**: Implement multi-step processes for improved responses.
21. **Corrective RAG**: Dynamically evaluate and correct the retrieval process.

By focusing on these areas and utilizing our chosen tech stack, we aim to create a robust and high-performing RAG system that delivers accurate and relevant results.