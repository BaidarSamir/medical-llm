# ==============================================================================
# FILE 1: PREPROCESS.PY (FINAL ROBUST VERSION V6 - FIXED & OPTIMIZED)
# ==============================================================================
# PURPOSE:
# 1. Force NLTK to download all required resources, including specific models.
# 2. Perform highly efficient, batch-processed text cleaning.
# 3. Load, combine, clean, and split raw data for model training.
# ==============================================================================

import pandas as pd
import re
import string
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet, stopwords
import os
import zipfile
from sklearn.model_selection import train_test_split
import time

# --- Most Aggressive NLTK Data Handling ---
def setup_nltk_data():
    """
    Forcefully downloads and unzips all required NLTK packages to solve
    environment issues definitively.
    """
    print("--- Force NLTK Setup ---")
    local_nltk_path = os.path.join(os.getcwd(), 'nltk_data')
    os.makedirs(local_nltk_path, exist_ok=True)

    if local_nltk_path not in nltk.data.path:
        nltk.data.path.insert(0, local_nltk_path)
        print(f"Added '{local_nltk_path}' to the front of NLTK's search path.")

    # <<< FIX: Added 'averaged_perceptron_tagger_eng' to the list.
    # This is the specific model the pos_tagger was looking for.
    packages = [
        'punkt',
        'punkt_tab',
        'averaged_perceptron_tagger',
        'averaged_perceptron_tagger_eng', # The missing resource
        'wordnet',
        'stopwords'
    ]

    for package in packages:
        try:
            print(f"--> Attempting to download '{package}'...")
            nltk.download(package, download_dir=local_nltk_path, quiet=False, raise_on_error=True)
        except Exception as e:
            print(f"    - Download failed or package already present for {package}. Error: {e}")

    # Unzipping logic remains the same, as it's robust.
    print("\n--> Attempting to unzip packages...")
    package_zip_files = {
        'averaged_perceptron_tagger': 'taggers/averaged_perceptron_tagger.zip',
        'punkt': 'tokenizers/punkt.zip',
        'stopwords': 'corpora/stopwords.zip',
        'wordnet': 'corpora/wordnet.zip'
    }

    for pkg_name, zip_location in package_zip_files.items():
        full_zip_path = os.path.join(local_nltk_path, zip_location)
        target_dir = os.path.dirname(full_zip_path)
        # Check if the unzipped directory already exists to avoid re-unzipping
        unzipped_dir_name = os.path.splitext(os.path.basename(zip_location))[0]
        unzipped_path = os.path.join(target_dir, unzipped_dir_name)

        if os.path.exists(full_zip_path) and not os.path.exists(unzipped_path):
            print(f"    - Found '{full_zip_path}'. Unzipping...")
            try:
                with zipfile.ZipFile(full_zip_path, 'r') as zip_ref:
                    zip_ref.extractall(target_dir)
                print(f"    - Unzip of '{pkg_name}' complete.")
            except zipfile.BadZipFile:
                print(f"    - WARNING: Bad zip file at '{full_zip_path}'.")
        else:
            print(f"    - Skipping unzip for '{pkg_name}' (already unzipped or zip not found).")

    print("--- NLTK setup process finished. ---")


# Run NLTK setup immediately
setup_nltk_data()

# --- Configuration ---
INPUT_DIRECTORY = '/kaggle/input/symptom-datasetgeneral-and-cardiology'
OUTPUT_DIRECTORY = '/kaggle/working/processed_symptoms'
TRAIN_DATA_FILENAME = 'train_symptoms.csv'
EVAL_DATA_FILENAME = 'eval_symptoms.csv'
EVAL_SPLIT_SIZE = 0.20

# --- Initialize Lemmatizer and Stop Words (once, globally) ---
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


# <<< EFFICIENCY UPGRADE: Helper function now takes a tag, not a word.
def get_wordnet_pos(nltk_tag):
    """Map NLTK POS tag to the format WordNetLemmatizer accepts"""
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN # Default to noun

# <<< EFFICIENCY UPGRADE: The entire preprocessing function is rewritten for speed.
def preprocess_text(text):
    """
    A highly efficient function to clean raw text data by processing in batches.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # 1. Basic cleaning
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))

    # 2. Tokenize the entire text
    tokens = nltk.word_tokenize(text)

    # 3. Filter out stop words and non-alphabetic tokens first
    filtered_tokens = [w for w in tokens if w.isalpha() and w not in stop_words]

    if not filtered_tokens:
        return ""

    # 4. Perform POS tagging ONCE on the list of filtered tokens (THIS IS THE KEY OPTIMIZATION)
    pos_tags = nltk.pos_tag(filtered_tokens)

    # 5. Lemmatize using the corresponding POS tag
    lemmatized_tokens = [
        lemmatizer.lemmatize(word, get_wordnet_pos(tag))
        for word, tag in pos_tags
    ]

    return " ".join(lemmatized_tokens)

def main():
    """Main function to execute the preprocessing and splitting pipeline."""
    print("\n--- Starting Data Preprocessing and Splitting ---")
    start_time = time.time()

    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    print(f"Output will be saved in: '{OUTPUT_DIRECTORY}'")

    if not os.path.isdir(INPUT_DIRECTORY):
        print(f"Error: Input directory not found at '{INPUT_DIRECTORY}'")
        return

    all_csv_files = [f for f in os.listdir(INPUT_DIRECTORY) if f.endswith('.csv')]
    if not all_csv_files:
        print(f"Error: No CSV files found in '{INPUT_DIRECTORY}'.")
        return

    print(f"Found {len(all_csv_files)} CSV file(s): {', '.join(all_csv_files)}")

    df_list = [pd.read_csv(os.path.join(INPUT_DIRECTORY, f)) for f in all_csv_files]
    combined_df = pd.concat(df_list, ignore_index=True)
    print(f"Combined data into a single dataset with {len(combined_df)} total rows.")

    if 'text' not in combined_df.columns or 'department' not in combined_df.columns:
        print("Error: All CSVs must contain 'text' and 'department' columns.")
        return

    print("Cleaning, removing stop words, and lemmatizing text data (using efficient method)...")
    combined_df.dropna(subset=['text'], inplace=True)

    # Apply the new, faster preprocessing function
    combined_df['processed_text'] = combined_df['text'].apply(preprocess_text)

    combined_df = combined_df[combined_df['processed_text'].str.strip() != '']
    combined_df.dropna(subset=['processed_text'], inplace=True)
    print("Text cleaning complete.")

    if combined_df.empty:
        print("Error: No data left after cleaning. Please check CSV content.")
        return

    print("Splitting data into training and evaluation sets...")
    stratify_col = None
    if combined_df['department'].nunique() > 1 and all(combined_df['department'].value_counts() > 1):
        stratify_col = combined_df['department']
    else:
        print("Warning: Cannot stratify split. Performing a regular split.")

    train_df, eval_df = train_test_split(
        combined_df, test_size=EVAL_SPLIT_SIZE, random_state=42, stratify=stratify_col
    )

    print(f"\nTraining set size: {len(train_df)} samples")
    print(f"Evaluation set size: {len(eval_df)} samples")

    train_data_path = os.path.join(OUTPUT_DIRECTORY, TRAIN_DATA_FILENAME)
    eval_data_path = os.path.join(OUTPUT_DIRECTORY, EVAL_DATA_FILENAME)

    print(f"\nSaving data...")
    train_df.to_csv(train_data_path, index=False)
    eval_df.to_csv(eval_data_path, index=False)
    print(f"Saved training data to '{train_data_path}'")
    print(f"Saved evaluation data to '{eval_data_path}'")

    end_time = time.time()
    print(f"\n--- Preprocessing and Splitting Complete in {end_time - start_time:.2f} seconds ---")

if __name__ == '__main__':
    main()

