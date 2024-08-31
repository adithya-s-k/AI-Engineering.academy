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

By focusing on these areas and utilizing our chosen tech stack, we aim to create a robust and high-performing RAG system that delivers accurate and relevant results.

To create a `README.md` file for the folder structure displayed in the image, I can provide a template that includes links to each folder and describes the overall structure. Here's an example:

## Folder Structure

```plaintext
RAG/
├── 00_Before_RAG
├── 00_Data_Ingestion
├── 00_RAG_from_Scratch
├── 01_Basic_RAG
├── 01_RAG_Evaluation
├── 01_RAG_Observability
├── 02_ReRanker_RAG
├── 03_Sentence_Window_Retriever
├── 04_Parent_Document_Retriever
├── 05_Self_Query_RAG
├── 06_HyDE_RAG
├── 07_RAG_Fusion
├── 08_RAPTOR
├── 09_Graph_RAG
├── 10_Agnetic_RAG
├── data/
└── README.md
```

### Folders

- [00_Before_RAG](./00_Before_RAG/): Preliminary setup and concepts before diving into RAG.
- [00_Data_Ingestion](./00_Data_Ingestion/): Modules for ingesting and preprocessing data.
- [00_RAG_from_Scratch](./00_RAG_from_Scratch/): Building RAG from scratch.
- [01_Basic_RAG](./01_Basic_RAG/): Basic implementation and understanding of RAG.
- [01_RAG_Evaluation](./01_RAG_Evaluation/): Methods and tools for evaluating RAG models.
- [01_RAG_Observability](./01_RAG_Observability/): Enhancing the observability of RAG systems.
- [02_ReRanker_RAG](./02_ReRanker_RAG/): Implementing reranking techniques in RAG.
- [03_Sentence_Window_Retriever](./03_Sentence_Window_Retriever/): Sentence window retriever modules.
- [04_Parent_Document_Retriever](./04_Parent_Document_Retriever/): Parent document retriever for contextual retrieval.
- [05_Self_Query_RAG](./05_Self_Query_RAG/): Self-query retriever integration in RAG.
- [06_HyDE_RAG](./06_HyDE_RAG/): Hybrid Dense-Sparse Retriever for RAG.
- [07_RAG_Fusion](./07_RAG_Fusion/): Fusion techniques for combining different retrieval methods.
- [08_RAPTOR](./08_RAPTOR/): RAPTOR framework integration with RAG.
- [09_Graph_RAG](./09_Graph_RAG/): Graph-based retrieval approaches in RAG.
- [10_Agnetic_RAG](./10_Agnetic_RAG/): Agnetic retriever techniques in RAG.

### Data

- [data](./data/): Contains datasets and other data files used throughout the project.

I'll convert the content from the image into Markdown format for you. Here's a structured version of the 21 strategies for enhancing Retrieval-Augmented Generation (RAG):

---

### 1. Simple RAG
Encodes document content into a vector store, enabling quick retrieval of relevant information to enhance model responses.

### 2. Context Enrichment
Adds surrounding context to each retrieved chunk, improving the coherence and completeness of the returned information.

### 3. Multi-faceted Filtering
Applies various filtering techniques (metadata, similarity thresholds, etc.) to refine and improve the quality of retrieved results.

### 4. Fusion Retrieval
Combines vector-based similarity search with keyword-based retrieval to improve document retrieval.

### 5. Intelligent Reranking
Reassesses and reorders initially retrieved documents to ensure that the most pertinent information is prioritized for subsequent processing.

### 6. Query Transformation
Modifies or expands the original query with query rewriting, step-back prompting, and sub-query decomposition.

### 7. Hierarchical Indices
First identifies relevant document sections through summaries, then drills down to specific details within those sections.

### 8. Hypothetical Questions
HyDE transforms queries into hypothetical documents that contain answers, bridging the gap between query and document distributions in vector space.

### 9. Choose Chunk Size
Selects an appropriate fixed size for text chunks to balance context preservation and retrieval efficiency.

### 10. Semantic Chunking
Unlike traditional methods that split text by fixed character/word counts, semantic chunking creates more meaningful, context-aware segments.

### 11. Context Compression
Compresses and extracts the most pertinent parts of documents in the context of a given query.

### 12. Explainable Retrieval
Not only retrieves relevant documents based on a query but also provides explanations for why each retrieved document is relevant.

### 13. Retrieval w/ Feedback
Utilizes user feedback on the relevance and quality of retrieved documents and generated responses to fine-tune retrieval and ranking models.

### 14. Adaptive Retrieval
Classifies queries into different categories and uses tailored retrieval strategies (factual, analytical, contextual, etc.) for each, considering user context and preferences.

### 15. Iterative Retrieval
Analyzes initial results and generates follow-up queries to fill in gaps or clarify information.

### 16. Ensemble Retrieval
Applies different embedding models or retrieval algorithms and uses voting or weighting mechanisms to determine the final set of retrieved documents.

### 17. Graph RAG
Retrieves entities and their relationships from a knowledge graph relevant to the query, combining with unstructured text for more informative responses.

### 18. Multi-Modal
Integrates models that can retrieve and understand different data modalities, combining insights from text, images, and videos.

### 19. RAPTOR
Uses abstractive summarization to recursively process and organize retrieved documents, organizing the information in a tree structure for hierarchical context.

### 20. Self RAG
Multi-step processes including retrieval decision, document retrieval, relevance evaluation, response generation, and more to improve model responses.

### 21. Corrective RAG
Dynamically evaluates and corrects the retrieval process, combining vector databases, web search, and models to improve response generation.

---

Feel free to copy this Markdown text for use in any platform that supports Markdown formatting.
