import os
import re
import string
import binascii
import time
import logging
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """
    A class to load, preprocess, and generate shingles from text files within a specified directory.
    
    This class handles the following tasks:
    - Loading text documents from individual files in a directory.
    - Performing basic text preprocessing, including normalization and punctuation removal.
    - Generating shingles from the preprocessed text for use in MinHash algorithms to estimate document similarity.
    
    Attributes:
        directory (str): The path to the directory containing text files.
        shingle_size (int): The number of words in each shingle.
        documents_df (pd.DataFrame): DataFrame to store document information and shingles.
    
    Methods:
        load_documents(): Loads text from files and stores them in a DataFrame.
        preprocess_text(text): Normalizes whitespace, removes punctuation, and lowercases the text.
        generate_shingles(text): Generates shingles from preprocessed text.
        process_documents(): Processes documents to generate shingles and stores them in the DataFrame.
        get_processed_documents(): Returns the DataFrame with processed document information and shingles.
    """
    def __init__(self, directory, shingle_size=5):
        """
        Initializes the DataLoader with the path to a directory containing text files.
        
        Parameters:
            directory (str): The path to the directory containing the document files.
            shingle_size (int): The number of words in each shingle.
        """
        self.directory = directory
        self.shingle_size = shingle_size
        self.documents_df = pd.DataFrame(columns=['DocID', 'DocName', 'DocText', 'WordCount'])

    def load_documents(self):
        """
        Loads text documents from the specified directory and stores them in a DataFrame.

        """
        temp_data = [] 
        for doc_id, filename in enumerate(os.listdir(self.directory), start=1):
            if filename.endswith(".txt"):
                filepath = os.path.join(self.directory, filename)
                with open(filepath, 'r', encoding='utf-8') as file:
                    text = file.read()
                    word_count = len(text.split())
                    temp_data.append({'DocID': doc_id, 'DocName': filename, 'DocText': text, 'WordCount': word_count})
        self.documents_df = pd.concat([self.documents_df, pd.DataFrame(temp_data)], ignore_index=True)

    def preprocess_text(self, text):
        """
        Processes text by normalizing whitespace, removing punctuation, and converting to lowercase.
        
        Parameters:
            text (str): The text content of a document.
        
        Returns:
            str: Processed text ready for generating shingles.
        """
        text = re.sub(r"\[(.*?)\]", "", text)  # Remove [+XYZ chars] in content
        text = re.sub(r'\s+', ' ', text).translate(str.maketrans('', '', string.punctuation)).lower()
        return text

    def generate_shingles(self, text):
        """
        Generates shingles from the text of a document
        
        Parameters:
            text (str): The text content of a document preprocessed for shingling.
        
        Returns:
            set: A set of shingles represented as CRC32 hashed values.
        """
        words = text.split()
        shingles = set()
        for i in range(len(words) - self.shingle_size + 1):
            shingle = ' '.join(words[i:i+self.shingle_size])
            crc = binascii.crc32(shingle.encode('utf-8')) & 0xffffffff
            shingles.add(crc)
        return shingles
    
    def proccess_documents(self):
        """
        Preprocesses text for shingling and generates shingles for each document, 
        storing them in a DataFrame

        Returns:
            DataFrame: DataFrame containing DocID, DocName, DocText, WordCount, and Shingles.
        """
        self.load_documents()
        self.documents_df['ProcessedText'] = self.documents_df['DocText'].apply(self.preprocess_text)
        
        start_time = time.time()
        self.documents_df['Shingles'] = self.documents_df['ProcessedText'].apply(self.generate_shingles)
        elapsed_time = time.time() - start_time
       
        logger.info("Shingling done in %.2f seconds", elapsed_time)
        self.documents_df.drop(columns=['ProcessedText'], inplace=True)
        return self.documents_df
    
    def get_proccessed_documents(self):
        """
        Returns the DataFrame containing document information and sets of shingles, 
        that represent documents
        
        Returns:
            DataFrame: DataFrame containing DocID, DocName, DocText, WordCount, and Shingles.
        """
        self.proccess_documents()
        return self.documents_df