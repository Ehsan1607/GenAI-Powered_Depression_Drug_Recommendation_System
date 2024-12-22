import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

def create_vector_store(file_path: str, output_index: str, output_metadata: str):
    """
    Create a FAISS vector store for efficient retrieval and save metadata.

    Workflow:
    1. Load the preprocessed dataset.
    2. Combine relevant fields (drug name, condition, demographics, etc.) into a single text for embedding.
    3. Generate embeddings using a pre-trained SentenceTransformer model.
    4. Store embeddings in a FAISS index for fast similarity search.
    5. Save the metadata (original dataset with combined text) to a CSV file.

    Args:
        file_path (str): Path to the cleaned dataset (CSV format).
        output_index (str): Path to save the FAISS index.
        output_metadata (str): Path to save the metadata CSV file.

    Returns:
        None
    """
    # Step 1: Load the cleaned data
    df = pd.read_csv(file_path)

    # Step 2: Combine relevant fields into a single text column for embedding
    df["combined_text"] = (
        "Drug Name: " + df["drug_name"].astype(str) + " | "
        "Condition: " + df["condition"].astype(str) + " | "
        "Gender: " + df["gender"].astype(str) + " | "
        "Age Group: " + df["age"].astype(str) + " | "
        "Time on Drug: " + df["time_on_drug"].astype(str) + " | "
        "Rating Overall: " + df["rating_overall"].astype(str) + " | "
        "Review: " + df["text"]
    )

    # Step 3: Initialize the SentenceTransformer model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Step 4: Generate embeddings using the combined text
    print("Generating embeddings...")
    embeddings = model.encode(df["combined_text"].tolist(), show_progress_bar=True)

    # Step 5: Save embeddings in a FAISS index
    dimension = embeddings.shape[1]  # Determine the dimensionality of the embeddings
    index = faiss.IndexFlatL2(dimension)  # Create a FAISS index for L2 (Euclidean) distance
    index.add(np.array(embeddings))  # Add embeddings to the index
    faiss.write_index(index, output_index)  # Save the index to a file
    print(f"FAISS index saved to {output_index}")

    # Step 6: Save metadata (original dataset + combined_text column)
    df.to_csv(output_metadata, index=False)
    print(f"Metadata saved to {output_metadata}")

if __name__ == "__main__":
    """
    Main execution block:
    - Calls `create_vector_store` to generate the FAISS index and save metadata.
    """
    # Input: Preprocessed dataset
    input_file = "data/cleaned_reviews.csv"
    
    # Output: Paths for FAISS index and metadata file
    index_output = "models/faiss_index"
    metadata_output = "models/reviews_with_metadata.csv"

    # Create the FAISS vector store and save metadata
    create_vector_store(input_file, index_output, metadata_output)
