import openai
import os

# Set OpenAI API key from environment variable or config file
def set_openai_api_key():
    """
    Set the OpenAI API key from the environment variable or config file.

    Priority:
    1. Environment variable `OPENAI_API_KEY`.
    2. Config file located at `config/openai_api_key.txt`.

    Raises:
        Exception: If the API key is not found in either source.
    """
    if "OPENAI_API_KEY" in os.environ:
        openai.api_key = os.environ["OPENAI_API_KEY"]
    else:
        # Load API key from the configuration file
        try:
            with open("config/openai_api_key.txt", "r") as f:
                openai.api_key = f.read().strip()
        except FileNotFoundError:
            raise Exception(
                "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable "
                "or create a config/openai_api_key.txt file with your API key."
            )

# Centralized method to call the LLM
def call_llm(prompt: str, model: str = "gpt-3.5-turbo", max_tokens: int = 100, temperature: float = 0.5) -> str:
    """
    Call the OpenAI LLM (Language Model) with the provided prompt and parameters.

    Args:
        prompt (str): The input prompt for the LLM.
        model (str): The model to use (default: "gpt-3.5-turbo").
        max_tokens (int): Maximum number of tokens in the LLM's response.
        temperature (float): The sampling temperature to control randomness (default: 0.7).

    Returns:
        str: The LLM's generated response.

    Raises:
        Exception: If any error occurs during the API call, including:
            - AuthenticationError: When the API key is invalid.
            - OpenAIError: When the API encounters issues.
            - General errors related to unexpected issues.
    """
    try:
        # Ensure the OpenAI API key is set
        set_openai_api_key()

        # Call the OpenAI API with the prompt and parameters
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},  # System message
                {"role": "user", "content": prompt},  # User's input prompt
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )

        # Extract and return the response content
        return response["choices"][0]["message"]["content"].strip()

    except openai.error.AuthenticationError as e:
        # Handle authentication errors
        raise Exception(f"Authentication Error: {e}")
    except openai.error.OpenAIError as e:
        # Handle general OpenAI API errors
        raise Exception(f"OpenAI API Error: {e}")
    except Exception as e:
        # Handle any other unexpected errors
        raise Exception(f"An unexpected error occurred: {e}")
