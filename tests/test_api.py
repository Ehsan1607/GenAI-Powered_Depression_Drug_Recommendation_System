import unittest
from fastapi.testclient import TestClient
from src.api import app

class TestAPI(unittest.TestCase):
    """
    Unit tests for the FastAPI application endpoints.
    """

    @classmethod
    def setUpClass(cls):
        """
        Set up resources shared across all tests in the class.
        
        Initializes the FastAPI test client, which is used to send requests
        to the application and validate its responses.
        """
        cls.client = TestClient(app)

    def test_api_recommend_success(self):
        """
        Test the /recommend endpoint with a valid query.
        
        Verifies:
        - The endpoint returns a status code of 200 (success).
        - The response JSON contains a 'response' key.
        - The 'response' contains at least one of the expected drug names.
        """
        response = self.client.post("/recommend", json={"query": "Which depression drug works best for women?"})
        self.assertEqual(response.status_code, 200)  # Check for successful response
        data = response.json()  # Parse the JSON response
        self.assertIn("response", data)  # Ensure the 'response' key is present
        self.assertGreater(len(data["response"]), 0)  # Ensure the response is not empty
        # Validate that the response contains expected drug names
        self.assertTrue(
            any(word in data["response"].lower() for word in ["fluoxetine", "prozac", "lexapro"]),
            "Response does not contain expected drug names."
        )

    def test_api_recommend_empty_query(self):
        """
        Test the /recommend endpoint with an empty query.
        
        Verifies:
        - The endpoint returns a status code of 200 (success).
        - The response JSON contains a 'response' key.
        - The 'response' provides a proper clarification for empty queries.
        """
        response = self.client.post("/recommend", json={"query": ""})
        self.assertEqual(response.status_code, 200)  # Check for successful response
        data = response.json()  # Parse the JSON response
        self.assertIn("response", data)  # Ensure the 'response' key is present
        # Validate that the response contains a clarification message
        self.assertIn("does not seem related", data["response"])

    def test_api_recommend_invalid_input(self):
        """
        Test the /recommend endpoint with invalid input (missing 'query' key).
        
        Verifies:
        - The endpoint returns a status code of 422 (Unprocessable Entity) for missing required fields.
        """
        response = self.client.post("/recommend", json={})
        self.assertEqual(response.status_code, 422)  # Ensure proper error handling for invalid input

if __name__ == "__main__":
    """
    Main entry point for running the tests.
    """
    unittest.main()
