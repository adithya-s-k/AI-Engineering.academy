### RAG Comparison

In the landscape of modern information retrieval, the RAG (Retrieval Augmented Generation) approach has gained significant traction across industries. This project endeavors to provide an insightful comparative analysis of various RAG methodologies, shedding light on their respective strengths and limitations.

#### Data Source:

The primary data source considered for this study comprises research papers and medical documents, typically available in PDF format. To facilitate seamless conversion to 

Few Example Datasource
- [Scientific_papers](https://huggingface.co/datasets/scientific_papers)

Markdown, we utilize the Marker-api (https://github.com/adithya-s-k/marker-api), providing a straightforward API interface for PDF-to-Markdown conversion.
**Comparison between Original PDF , Marker API , PyPDF**
![Comparision](./assets/comparision.png)

Converted these PDFs to Markdown and put them for ingestion into the RAG.

#### RAG Techniques Explored:

1. **Naive RAG**
   - *Explanation*: The Naive RAG technique involves a straightforward approach to retrieval and generation, without extensive preprocessing or semantic analysis. It serves as a baseline for comparison against more sophisticated methods.

2. **Semantic Chunking RAG**
   - *Explanation*: Semantic Chunking RAG employs advanced semantic analysis techniques to break down text into meaningful chunks or segments. This method aims to improve relevance and coherence in retrieved information.

3. **Sentence Window Retrieval**
   - *Explanation*: Sentence Window Retrieval focuses on retrieving information based on contextual windows within a document. By considering the surrounding sentences, this technique aims to enhance the relevance of retrieved content.

4. **Auto-Merging Retrieval**
   - *Explanation*: Auto-Merging Retrieval leverages algorithms for automatically merging and synthesizing information from multiple sources. This approach aims to streamline the retrieval process and improve the quality of generated content.

5. **Agentic RAG**
   - *Explanation*: Agentic RAG introduces an agent-based approach to information retrieval and generation. Agents autonomously retrieve and process data, contributing to a more dynamic and adaptive system.

6. **Visual RAG**
   - *Explanation*: Visual RAG integrates visual information, such as images or diagrams, into the retrieval and generation process. By incorporating visual cues, this technique aims to enrich the content and improve user understanding.
   

#### Technology Stack:

- **Orchestration Framework**: Leveraging the Llama index for seamless orchestration.
- **Embedding Generation**:
  - *Textual*: Utilizing Fastembed for textual embedding.
  - *Visual*: Employing CLIP for generating visual embeddings.
- **Language Model (LLM)**: Initial deployment utilizes OpenAI's LLM, with plans to extend compatibility to other open-source LLMs.
- **Evaluation Metrics**: Evaluation conducted using Truera and RAGAS.



#### Project Structure:

- **1_Naive_RAG.ipynb**
- **2_Semantic_Chunking_RAG.ipynb**
- **3_Sentence_Window_Retrieval_RAG.ipynb**
- **4_Auto_Merging_Retrieval_RAG.ipynb**
- **5_Agentic_RAG.ipynb**
- **6_Visual_RAG.ipynb**
- **assets/**
- **data/**: Repository for storing all data files, including PDF documents and Markdown files.

Through this structured approach, we aim to provide a comprehensive understanding of RAG methodologies, facilitating informed decision-making in their application across various domains.