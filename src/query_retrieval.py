import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
#from src.llm_handler import call_llm# Uncomment this if using relative imports
from llm_handler import call_llm  # Centralized LLM interaction utility

def query_retrieval(user_query: str, index_path: str, metadata_path: str) -> str:
    """
    Retrieve relevant context for a user query from a FAISS vector store and metadata.

    Workflow:
    1. Load the FAISS index and metadata file.
    2. Validate the metadata file contains the 'combined_text' column.
    3. Generate an embedding for the user's query.
    4. Retrieve the top 5 most relevant contexts using the FAISS index.
    5. Combine the retrieved contexts and pass them, along with the query, to the LLM.
    6. Return the LLM's response.

    Args:
        user_query (str): The user's query.
        index_path (str): Path to the FAISS index file.
        metadata_path (str): Path to the metadata CSV file.

    Returns:
        str: A response generated using the relevant context.

    Raises:
        Exception: If an error occurs during any step of the process.
    """
    try:
        # Step 1: Load FAISS index and metadata file
        index = faiss.read_index(index_path)
        metadata = pd.read_csv(metadata_path)

        # Step 2: Validate that the metadata contains the 'combined_text' column
        if "combined_text" not in metadata.columns:
            raise ValueError("The metadata file does not contain the 'combined_text' column.")

        # Step 3: Generate embedding for the user's query
        model = SentenceTransformer("all-MiniLM-L6-v2")  # Load a pre-trained SentenceTransformer model
        query_embedding = model.encode([user_query])  # Encode the query into an embedding

        # Step 4: Retrieve the top 5 most relevant contexts
        distances, indices = index.search(query_embedding, k=5)  # Perform a nearest neighbor search
        retrieved_metadata = metadata.iloc[indices[0]]  # Get metadata for the retrieved indices

        # Step 5: Combine the top contexts into a single string
        context = "\n".join(retrieved_metadata["combined_text"].astype(str).tolist())

        # Step 6: Use the context and query to generate a response via the LLM
        prompt = (
            f"You are an expert assistant for depression drug recommendations. Based on the following context, "
            f"answer the user's question concisely and accurately:\n\n"
            f"Context:\n{context}\n\n"
            f"User Query: {user_query}\n\n"
            f"Response:"
        )
        response = call_llm(prompt, model="gpt-3.5-turbo", max_tokens=300, temperature=0.5)

        return response

    except faiss.IOReaderError:
        # Handle errors related to loading the FAISS index
        raise Exception(f"Error: Could not read FAISS index from path {index_path}. Please ensure the index exists.")
    except FileNotFoundError as e:
        # Handle file not found errors for metadata or index files
        raise Exception(f"Error: {e}")
    except ValueError as e:
        # Handle validation errors for the metadata file
        raise Exception(f"Error: {e}")
    except Exception as e:
        # Handle any other unexpected errors
        raise Exception(f"An unexpected error occurred during query retrieval: {e}")
