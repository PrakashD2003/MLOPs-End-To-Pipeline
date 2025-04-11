import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
import yaml

# Ensure that a directory named 'logs' exist in our root folder (if not it creates one)(for storing log file)
log_dir = 'logs'
os.makedirs(log_dir,exist_ok=True)

# Logging Configuration
logger = logging.getLogger('Feature_Engineering') # Created Object of logger with name 'Pre_Proccessing'
logger.setLevel("DEBUG") # Setting level of logger as 'DEBUG' so that we see debug as well as all other levels after 'DEBUG'

# Creating Handlers
console_handler = logging.StreamHandler() # Console(terminal) handeler
file_path = os.path.join(log_dir,'Feature_Engineering.log') # Creating path for log_file
file_handler = logging.FileHandler(file_path,encoding="utf-8") # Creates Log file

# Setting Log Levels for Handlers
console_handler.setLevel("DEBUG")
file_handler.setLevel("DEBUG")

# Creating a Formatter and attaching it to handelers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Adding handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Function for loading the Dataset
def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        df.fillna('',inplace=True)
       
        logger.info("Data successfully loaded and NaN filled from %s", file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error("Failed to parse the CSV file: %s", e)
        raise
    except FileNotFoundError as e:
        logger.error('File not found: %s', e)
        raise
    except Exception as e:
        logger.error("Unexpected error occeured while loading the data: %s", e)
        raise

# Function to apply TF-IDF transformation to the dataset
# This function converts text data into numerical features using TF-IDF (Term Frequency-Inverse Document Frequency).
# It assigns weights to words based on their importance and transforms the dataset into a numerical format.
def apply_tfidf(train_data: pd.DataFrame, test_data: pd.DataFrame, max_features: int) -> tuple:
    """Apply TF-IDF transformation to the dataset."""
    try:
        logger.debug('Tranforming text Data using TDIDF...')
        # Validate that the input data contains the required 'text' and 'target' columns
        if 'text' not in train_data.columns or 'text' not in test_data.columns:
            logger.error("Missing 'text' column in input data.")
            raise KeyError("Column 'text' not found in input data.")
        
        if 'target' not in train_data.columns or 'target' not in test_data.columns:
            logger.error("Missing 'target' column in input data.")
            raise KeyError("Column 'target' not found in input data.")
        
        # Ensure max_features is a positive integer
        if not isinstance(max_features, int) or max_features <= 0:
            logger.error("Invalid max_features: %s. It must be a positive integer.", max_features)
            raise ValueError("max_features must be a positive integer.")

        # Initialize the TF-IDF vectorizer
        # max_features determines the number of most important words to keep
        vectorizer = TfidfVectorizer(max_features=max_features)

        # Extract the text data (features) and target labels
        X_train = train_data['text'].values  # Training text data
        y_train = train_data['target'].values  # Training labels
        X_test = test_data['text'].values  # Testing text data
        y_test = test_data['target'].values  # Testing labels

        # Fit the vectorizer on the training data and transform it into numerical format
        X_train_tfidf = vectorizer.fit_transform(X_train)  # Learn vocabulary & transform training data
        X_test_tfidf = vectorizer.transform(X_test)  # Transform test data using the same vocabulary

        # Convert the transformed TF-IDF matrices into Pandas DataFrames
        train_df = pd.DataFrame(X_train_tfidf.toarray())  # Convert sparse matrix to DataFrame
        train_df['label'] = y_train  # Add the target labels to the DataFrame

        test_df = pd.DataFrame(X_test_tfidf.toarray())  # Convert sparse matrix to DataFrame
        test_df['label'] = y_test  # Add the target labels to the DataFrame

        # Log success message
        logger.info('TF-IDF applied and data transformed successfully.')
        
        # Return the transformed training and testing datasets
        return train_df, test_df

    except Exception as e:
        # Log and raise any error encountered during processing
        logger.error('Error during TF-IDF transformation: %s', e)
        raise

# Function to save processed train and test dataset
def save_data(train_data_df: pd.DataFrame,test_data_df: pd.DataFrame,file_path: str) ->None:
    """Save the proccessed train and test datasets."""
    try:
        # Creating a Directory named 'Proccessed' if not already exist
        logger.debug("Creating directory for saving processed data...")
        proccessed_data_path = os.path.join(file_path,'Processed') # Defining path for 'Processed' directory
        os.makedirs(proccessed_data_path,exist_ok=True)
        logger.info("Successfully Created Directory at: %s",proccessed_data_path)

        # Saving Proccessed Train and Test data as CSV inside 'raw' directory
        logger.debug("Saving the Proccessed train and test datasets...")
        train_data_df.to_csv(os.path.join(proccessed_data_path,"train_tfidf.csv"),index=False)
        test_data_df.to_csv(os.path.join(proccessed_data_path,"test_tfidf.csv"),index=False)
       
        logger.info('Proccessed Training and Test data saved to: %s', proccessed_data_path)
    except Exception as e:
        logger.error('Unexpected error occeured while saving the data: %s', e)
        raise

def main():
    try:
        max_features = 50
        
        logger.debug("Attempting to load training data from: %s", './data/interim/train_processed.csv')
        train_data = load_data('./data/interim/train_processed.csv')
       
        logger.debug("Attempting to load testing data from: %s", './data/interim/test_processed.csv')
        test_data = load_data('./data/interim/test_processed.csv')
        

        train_df, test_df = apply_tfidf(train_data, test_data, max_features)

        save_data(train_df,test_df,file_path='./data')
       
    except Exception as e:
        logger.error('Unexpected error occured while the feature engineering process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()