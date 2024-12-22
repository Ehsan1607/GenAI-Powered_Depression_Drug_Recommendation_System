from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

#from src.query_retrieval import query_retrieval  # Uncomment this if using relative imports
#from src.llm_handler import call_llm            # Uncomment this if using relative imports

from query_retrieval import query_retrieval  # Function to retrieve relevant data from FAISS vector database
from llm_handler import call_llm  # Function to call the GPT-based language model

# Initialize FastAPI application
app = FastAPI()

# Define the structure for incoming API request payloads
class QueryRequest(BaseModel):
    query: str  # User's query string

def analyze_query_with_llm(query: str) -> bool:
    """
    Analyze the user's query using GPT-3.5-turbo to check its relevance to depression drug recommendations.

    Args:
        query (str): The user's query to analyze.

    Returns:
        bool: True if the query is related to depression drug recommendations, False otherwise.
    """
    prompt = (
        f"You are an assistant that determines whether a query is related to depression drug recommendations. "
        f"Only respond with 'Yes' or 'No'. Here are examples of related queries:\n"
        f"- 'Which drug works best for depression in women aged 30 to 40?'\n"
        f"- 'Is Prozac effective for treating anxiety along with depression?'\n"
        f"- 'What are the best-rated drugs for men suffering from depression?'\n\n"
        f"Now analyze the following query:\n"
        f"'{query}'\n"
        f"Is this query related to depression drug recommendations? Respond with 'Yes' or 'No' only."
    )
    # Call the LLM with the constructed prompt
    response = call_llm(prompt, max_tokens=10, temperature=0.2)
    # Return True if the response is "Yes", otherwise False
    return response.lower() == "yes"

@app.post("/recommend")
def recommend(query_request: QueryRequest):
    """
    API endpoint to provide drug recommendations based on the user's query.

    Steps:
        1. Analyze the query for relevance using the LLM.
        2. If relevant, retrieve data using FAISS-based query retrieval.
        3. Return the response or raise an HTTP exception for errors.

    Args:
        query_request (QueryRequest): The incoming query request from the user.

    Returns:
        dict: The response containing the recommendation or a clarification message.
    """
    # Extract and sanitize the user's query
    query = query_request.query.strip()

    # Analyze the query for relevance using the LLM
    is_related = analyze_query_with_llm(query)
    if not is_related:
        # Return clarification if the query is not relevant
        return {"response": "Your query does not seem related to depression drug recommendations. Please rephrase."}

    # Process a valid query
    try:
        # Retrieve relevant results using the FAISS vector database
        result = query_retrieval(query, "models/faiss_index", "models/reviews_with_metadata.csv")
        return {"response": result}
    except Exception as e:
        # Handle errors during query retrieval
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

# Main entry point to run the FastAPI application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
