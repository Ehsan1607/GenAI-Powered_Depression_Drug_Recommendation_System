import unittest
from src.query_retrieval import query_retrieval
import os

class TestQueryRetrieval(unittest.TestCase):
    """
    Unit tests for the query retrieval functionality.

    Tests the integration of FAISS vector search with metadata to ensure
    relevant responses are retrieved for valid user queries.
    """

    @classmethod
    def setUpClass(cls):
        """
        Set up shared test resources.

        Initializes paths for FAISS index and metadata CSV files. Ensures
        these files exist before running tests to avoid runtime errors.
        """
        cls.faiss_index_path = "models/faiss_index"  # Path to the FAISS index
        cls.metadata_path = "models/reviews_with_metadata.csv"  # Path to the metadata CSV

        # Check if required files exist
        assert os.path.exists(cls.faiss_index_path), f"FAISS index not found at {cls.faiss_index_path}."
        assert os.path.exists(cls.metadata_path), f"Metadata file not found at {cls.metadata_path}."

    def test_query_retrieval_success(self):
        """
        Test query retrieval with a valid query.

        Verifies:
        - The function returns a string response.
        - The response is non-empty.
        - The response contains expected drug names such as "Fluoxetine", "Prozac", or "Lexapro".
        """
        query = "Which depression drug works best for women?"  # Example valid query
        response = query_retrieval(query, self.faiss_index_path, self.metadata_path)  # Call the query retrieval function

        # Validate response type and content
        self.assertIsInstance(response, str)  # Check if the response is a string
        self.assertGreater(len(response), 0)  # Ensure the response is non-empty
        self.assertTrue(
            any(word in response.lower() for word in ["fluoxetine", "prozac", "lexapro"]),  # Validate expected drug names
            "Response does not contain expected drug names."
        )

if __name__ == "__main__":
    """
    Main entry point for running the tests.
    """
    unittest.main()
