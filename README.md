# **GenAI-Powered Depression Drug Recommendation System**

This project leverages Generative AI (GenAI) and a Retrieval-Augmented Generation (RAG) architecture to provide insightful and personalized recommendations for depression drug treatments based on user reviews. The system supports both a REST API and a Command-Line Interface (CLI) for user interaction.

---

## **Prerequisites**

1. **Python**: Ensure Python 3.10 is installed.

2. **Libraries**: Install required Python packages listed in `requirements.txt`.

3. **OpenAI API Key**:

   - Save your OpenAI API key in `config/openai_api_key.txt`.
   - Example:
     ```text
     sk-xxxxxxxxxxxxxxxxxxxx
     ```

4. **Dataset**: The project uses the "WebMD Reviews for Psychiatric Drugs" dataset. Ensure the dataset file (`webmd_reviews.csv`) is located in the `data/` directory.

---

## **Structure**

```plaintext
depression-treatment/
├── requirements.txt         # Project dependencies
├── config/                  # Configuration files
│   └── openai_api_key.txt   # OpenAI API key
├── data/                    # Dataset files
│   ├── webmd_reviews.csv    # Raw dataset
│   ├── cleaned_reviews.csv  # Preprocessed dataset
├── models/                  # Model-related files
│   ├── faiss_index          # FAISS vector index
│   ├── reviews_with_metadata.csv # Metadata for vector search
├── src/                     # Core application code
│   ├── api.py               # REST API implementation
│   ├── cli.py               # Command-Line Interface
│   ├── llm_handler.py       # LLM interaction utility
│   ├── preprocess.py        # Preprocessing script
│   ├── query_retrieval.py   # Query retrieval logic
│   ├── vector_store.py      # FAISS index creation
├── tests/                   # Test files
│   ├── test_api.py          # Tests for the API
│   ├── test_query_retrieval.py # Tests for query retrieval
├── README.md                # Project README file
├── DESIGN.md                # Project design documentation
```

---

## **Set Up the Environment**

1. Clone the repository and navigate to the project directory:

   ```bash
   git clone <repository_url>
   cd GenAI-Powered_Depression_Drug_Recommendation_System
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Ensure the OpenAI API key is configured in `config/openai_api_key.txt`.

4. Verify the dataset file `webmd_reviews.csv` is present in the `data/` directory.

5. Preprocess the dataset:

   ```bash
   python src/preprocess.py
   ```

6. Create the FAISS vector database:

   ```bash
   python src/vector_store.py
   ```

---

## **Run the Application**

### **API**

Start the API by running:

```bash
python src/api.py
```

To test the API, open a separate terminal window and execute the following command:

```bash
curl -X POST "http://127.0.0.1:8000/recommend" -H "Content-Type: application/json" -d "{\"query\": \"Which depression drug works best for women aged 30 to 40?\"}"

```

### **CLI**

Run the Command-Line Interface:

```bash
python src/cli.py
```

- Example interaction:
  ```plaintext
  Your Query or type 'exit' to quit: Which depression drug works best for men?
  Response: Prozac and Lexapro are commonly effective for men with depression.
  ```

---

## **Test the Application**

To run the unit tests you need to use relative import in code files::

	- from src.query_retrieval import query_retrieval
	- from src.llm_handler import call_llm

```bash
python -m unittest discover -s tests
```

Example Output:

```plaintext
...
----------------------------------------------------------------------
Ran 4 tests in 2.345s

OK
```

---

## **Description of Test Files**

1. **`test_api.py`**:

   - Tests the functionality of the `/recommend` API endpoint.
   - Validates response codes, query handling (valid, empty, or invalid inputs), and output correctness.

2. **`test_query_retrieval.py`**:

   - Verifies the query retrieval logic.
   - Ensures the FAISS index is correctly queried and appropriate responses are retrieved for valid inputs.
   - Tests edge cases like empty queries.


