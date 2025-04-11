import pandas as pd
import os
from sklearn.model_selection import train_test_split
import logging

# Ensure that a directory named 'logs' exist in our root folder (if not it creates one)(for storing log file)
log_dir = 'logs'
os.makedirs(log_dir,exist_ok=True)

# Logging Configuration
logger = logging.getLogger('Data_Ingestion') # Created Object of logger with name 'Data_Ingetion'
logger.setLevel('DEBUG') # Setting level of logger as 'DEBUG' so that we see debug as well as all other levels after 'debug'

# Creating Handelers
console_handler = logging.StreamHandler() # Console(terminal) handeler
log_file_path = os.path.join(log_dir,'Data_Ingetion_logs.log') # Creating path for log_file
file_handler = logging.FileHandler(log_file_path, encoding="utf-8") # Creates Log file

# Setting Log Levels for Handelers
console_handler.setLevel('DEBUG')
file_handler.setLevel('DEBUG')

# Creating a Formatter and attaching it to handelers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Adding handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Function for loading the Dataset
def load_data(data_url: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        logger.debug("Attempting to load data from: %s", data_url)
        
        df = pd.read_csv(data_url)
       
        logger.info("Data successfully loaded from %s", data_url)
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

# Function for Preprocessing the Dataset
def preprocessing_data (df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data."""
    try:
        logger.debug("Starting data preprocessing...")
       
        # Removing unnecssary columns
        df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)
       
        # Renaming Columns
        df.rename(columns={'v1':'target','v2':'text'},inplace=True)
        
        logger.info('Data PreProcessing Completed')
        return df  # Return the modified DataFrame
    except KeyError as e:
        logger.error('Missing Colunm in the dataframe: %s', e)
        raise
    except Exception as e:
        logger.error("Unexpected error occeured while preprocessing: %s", e)
        raise

# Function to save processed train and test dataset
def save_data(train_data: pd.DataFrame,test_data: pd.DataFrame,data_path):
    """Save the train and test datasets."""
    try:
        # Creating a Directory named 'raw' if not already exist
        logger.debug("Creating directory for saving processed data...")
        raw_data_path = os.path.join(data_path,'raw') # Defining path for 'raw' directory
        os.makedirs(raw_data_path,exist_ok=True)
        logger.info("Successfully Created Directory at: %s",raw_data_path)

        # Saving Train and Test data as CSV inside 'raw' directory
        logger.debug("Saving train and test datasets...")
        train_data.to_csv(os.path.join(raw_data_path,"train.csv"),index=False)
        test_data.to_csv(os.path.join(raw_data_path,"test.csv"),index=False)
       
        logger.info('Training and Test data saved to: %s', raw_data_path)
    except Exception as e:
        logger.error('Unexpected error occeured while saving the data: %s', e)
        raise
        
# Define the main function to execute the data processing pipeline
def main():
    try:
        # Set the test dataset size for splitting
        test_size = 0.20
        
        # Define the URL of the dataset (CSV file)
        data_url = "https://raw.githubusercontent.com/PrakashD2003/DATASETS/refs/heads/main/spam.csv"
        
        # Load the dataset from the provided URL
        df = load_data(data_url=data_url)
        
        # Preprocess the dataset (e.g., cleaning, feature extraction, transformation)
        final_df = preprocessing_data(df)
        
        # Split the dataset into training and testing sets
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=2)
        
        # Save the train and test data to the specified directory
        save_data(train_data, test_data, data_path='./data')
    
    # Handle any unexpected exceptions that may occur during execution
    except Exception as e:
        logger.error('Failed to complete the data ingestion process: %s', e)
        print(f"Error: {e}")

# Ensure that the main function runs only when the script is executed directly
if __name__ == '__main__':
    main()



