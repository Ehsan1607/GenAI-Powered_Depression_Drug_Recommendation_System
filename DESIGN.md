# **DESIGN.md**

---

## **Introduction**
This project implements a Retrieval-Augmented Generation (RAG)-based system designed for an Insurance Claims team. The system determines the most effective depression treatments by analyzing user reviews from the Kaggle dataset "WebMD Reviews for Psychiatric Drugs." Using Machine Learning (ML) and Natural Language Processing (NLP), the system provides insights and business recommendations through a REST API and Command-Line Interface (CLI).

The primary goals of the system are:
1. Analyze drug characteristics based on real-world reviews.
2. Enable stakeholders to query the system for drug recommendations.
3. Employ scalable and efficient ML techniques for generating accurate responses.

---

## **System Architecture**

### **Components**
1. **Data Preprocessing**:
   - Cleans and filters the dataset for relevant depression-related reviews.
   - Combines metadata (e.g., drug name, condition, ratings) into a contextual representation.

2. **Vector Database**:
   - Uses FAISS for scalable similarity search on transformer-generated embeddings.
   - Stores vectorized representations of drug reviews.

3. **Transformer-Based Embedding Generation**:
   - Uses the `all-MiniLM-L6-v2` model to generate dense embeddings of textual data.

4. **Query Retrieval**:
   - Retrieves the most relevant reviews using FAISS and combines contexts for response generation.

5. **Response Generation**:
   - Uses OpenAI GPT-3.5-turbo to generate concise, context-aware answers.

6. **API and CLI**:
   - RESTful API (`api.py`) serves recommendations programmatically.
   - CLI (`cli.py`) provides an interactive interface for users to query the system.

---

## **System Workflow**

1. **Data Preprocessing**:
   - Input: Raw Kaggle dataset (`webmd_reviews.csv`).
   - Filters depression-related reviews and enriches data by combining metadata into a `combined_text` column.

2. **Prompt-Based Query Analysis**:
   - Validates and analyzes user queries using a prompt-based approach with OpenAI GPT-3.5-turbo.
   - Ensures queries are relevant and meaningful before proceeding.

3. **Vector Store Creation**:
   - Embedding: Generates embeddings using the transformer model.
   - Indexing: Saves embeddings in a FAISS vector database for efficient similarity search.

4. **Query Handling**:
   - **Embedding Generation**: Encodes the validated query into a dense vector.
   - **Retrieval**: Searches the FAISS index for the top `k` relevant reviews.

5. **Prompt-Based RAG Generation**:
   - Combines retrieved contexts.
   - Constructs a tailored prompt for GPT-3.5-turbo, including:
     - User query.
     - Retrieved contexts.
   - Generates a concise and contextually relevant response.

6. **Output Delivery**:
   - RESTful API: Serves responses via the `/recommend` endpoint.
   - CLI: Displays the response directly in the terminal.

---

## **Design Choices**

1. **FAISS for Vector Storage**:
   - Selected for its scalability and efficiency in handling large datasets.
   
2. **SentenceTransformer**:
   - Lightweight and effective for generating embeddings tailored to textual similarity tasks.

3. **GPT-3.5-turbo**:
   - Provides advanced natural language understanding and response generation capabilities.

4. **LLM for Query Validation**:
   - Ensures user queries are meaningful and relevant before processing.

5. **Dual Interface**:
   - Supports both API and CLI for versatility in user interaction.

---

## **Scalability and Optimization**

1. **User Query Filtering**:
   - Validates queries using an LLM to ensure relevance, reducing unnecessary computations.

2. **Vector Search**:
   - FAISS enables efficient similarity search and can scale horizontally by partitioning the index across servers.

3. **Efficient Embedding Model**:
   - The `all-MiniLM-L6-v2` model balances performance and computational efficiency for real-time processing.

4. **Asynchronous API**:
   - FastAPI's asynchronous capabilities allow for concurrent request handling, improving responsiveness under high loads.

---

## **Performance Metrics**

1. **System Accuracy**:
   - Measured by the relevance and quality of GPT-3.5-turbo responses.
   - **Recall**: Percentage of relevant reviews correctly retrieved from the vector database.

2. **Query Latency**:
   - Time taken to process a query and deliver a response.
   - Goal: <1 second for retrieval and <3 seconds end-to-end.

3. **Resource Utilization**:
   - Monitored for efficient memory, CPU, and GPU usage during vector search and embedding generation.

4. **Fallback Rate**:
   - Frequency of fallback mechanisms like:
     - Query clarifications for vague inputs.
     - Spelling corrections.

5. **User Satisfaction**:
   - Evaluated based on the clarity and relevance of responses provided.

---

## **Future Enhancements**

1. **Handling Complex Queries**:
   - Improve parsing of multi-part or ambiguous queries.

2. **Streamlined Clarifications**:
   - Introduce interactive feedback for vague or incomplete queries, helping users refine their questions.

3. **Conversation History**:
   - Add context-awareness by maintaining user conversation history for follow-up queries.

4. **Cache Embeddings**:
   - Cache embeddings of frequently queried items to improve response time for repeated queries.

5. **Model Fine-Tuning**:
   - Fine-tune transformer and GPT models on the dataset to improve embedding accuracy and response quality.

6. **Optimize API Use**:
   - Minimize token usage through prompt optimization and leverage local caching for repeated or simple queries.

---
