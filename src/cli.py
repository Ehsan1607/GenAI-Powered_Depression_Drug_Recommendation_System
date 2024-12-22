# Import necessary modules for query retrieval and LLM handling

#from src.query_retrieval import query_retrieval  # Uncomment this if using relative imports
#from src.llm_handler import call_llm            # Uncomment this if using relative imports

from query_retrieval import query_retrieval      # Import for query retrieval logic
from llm_handler import call_llm                # Import for handling LLM queries

def analyze_query_with_llm(query: str) -> bool:
    """
    Analyze the user's query using GPT-3.5-turbo to determine if it is related to depression drug recommendations.

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
    # Call the LLM with the constructed prompt and analyze the response
    response = call_llm(prompt, max_tokens=10, temperature=0.2)
    # Return True if the response is "Yes", otherwise False
    return response.lower() == "yes"

def main():
    """
    Command-Line Interface (CLI) for querying the depression drug recommendation system.

    Steps:
        1. Accepts user input (queries).
        2. Analyzes query relevance using the LLM.
        3. If relevant, retrieves responses using the FAISS vector database.
        4. Displays the result or handles errors gracefully.
    """
    print("Welcome to the Depression Treatment Q&A CLI!")
    print("Type your query below or type 'exit' to quit.\n")

    while True:
        # Get user input
        query = input("Your Query or type 'exit' to quit: ").strip()
        
        # Handle exit command
        if query.lower() == "exit":
            print("Goodbye!")
            break

        # Handle blank queries
        if not query:
            print("Please provide a query. For example: 'Which depression drug works best for women aged 30 to 40?'\n")
            continue

        # Analyze the query for relevance
        try:
            is_related = analyze_query_with_llm(query)
        except Exception as e:
            print(f"An unexpected error occurred during query analysis: {e}\n")
            continue

        # Handle unrelated queries
        if not is_related:
            print("Your query does not seem related to depression drug recommendations.")
            print("Please rephrase your query. Examples of valid queries include:")
            print("- 'Which drug works best for depression in women aged 30 to 40?'\n")
            continue

        # Process valid query
        try:
            response = query_retrieval(query, "models/faiss_index", "models/reviews_with_metadata.csv")
            print(f"Response: {response}\n")
        except Exception as e:
            print(f"An unexpected error occurred: {e}\n")

# Entry point for the CLI application
if __name__ == "__main__":
    main()
