import pandas as pd

def preprocess_data(file_path: str) -> pd.DataFrame:
    """
    Preprocess the raw WebMD reviews dataset.

    Steps:
    1. Load the dataset from the specified file path.
    2. Filter for reviews related to depression or related conditions.
    3. Remove rows with missing text or overall ratings.
    4. Clean and normalize the text data by converting to lowercase and removing non-alphabetic characters.

    Args:
        file_path (str): The path to the raw dataset file (CSV format).

    Returns:
        pd.DataFrame: A cleaned and preprocessed DataFrame.
    """
    # Load raw data from the CSV file
    df = pd.read_csv(file_path)

    # Filter for rows where the 'condition' column contains relevant conditions
    relevant_conditions = r"(?i)depression|major depressive disorder|bipolar depression|anxiousness associated with depression"
    df = df[df["condition"].str.contains(relevant_conditions, na=False)]

    # Drop rows with missing values in the 'text' or 'rating_overall' columns
    df = df.dropna(subset=["text", "rating_overall"])

    # Clean text data: convert to lowercase and remove non-alphabetic characters
    df["text"] = df["text"].str.lower().str.replace(r"[^a-z\s]", "", regex=True)

    return df

if __name__ == "__main__":
    """
    Main execution block:
    1. Calls the `preprocess_data` function to clean the dataset.
    2. Saves the preprocessed data to a new CSV file.
    """
    # Input file path
    input_file = "data/webmd_reviews.csv"
    
    # Output file path
    output_file = "data/cleaned_reviews.csv"

    # Preprocess the data
    data = preprocess_data(input_file)
    
    # Save the cleaned data to a new CSV file
    data.to_csv(output_file, index=False)
    
    # Print success message
    #print(f"Preprocessed data saved to {output_file}")
