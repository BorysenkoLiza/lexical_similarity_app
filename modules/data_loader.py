import os
import re
import string
import binascii

class DataLoader:
    """
    A class to load, preprocess, and generate shingles from text files within a specified directory.
    
    This class handles the loading of text documents from individual files, performs basic text preprocessing,
    and generates shingles which are used in MinHash algorithms for estimating document similarity.
    
    Attributes:
        directory (str): Directory path where the text files are stored.
        doc_counter (int): Counter to assign a unique document ID to each document.
        docs_as_sets (dict): Dictionary to store document ID and corresponding shingles.
        shingle_size (int): The number of words in each shingle.
    
    Methods:
        load_documents(): Iterator that loads text from files and assigns unique document IDs.
        basic_preprocess(text): Normalizes whitespace, removes punctuation, and lowercases the text.
        generate_shingles(text): Generates shingles from preprocessed text.
        get_docs_as_sets_and_texts(): Processes documents to generate and retrieve shingles for each.
    """
    def __init__(self, directory, shingle_size=3):
        """
        Initializes the DataLoader with the path to a directory containing text files.
        
        Parameters:
            directory (str): The path to the directory containing the document files.
            shingle_size (int): The number of words in each shingle.
        """
        self.directory = directory
        self.doc_counter = 0  # Initialize a counter for docID assignment
        self.docs_as_sets = {}  # Dictionary to store document ID and corresponding shingles
        self.shingle_size = shingle_size

    def load_documents(self):
        """
        Loads and preprocesses each text file in the specified directory, assigning a unique document ID to each.
        
        Yields:
            tuple: A tuple containing the document ID and text content for each document.
        """
        for filename in os.listdir(self.directory):
            if filename.endswith(".txt"):  # Ensures only text files are processed
                filepath = os.path.join(self.directory, filename)
                with open(filepath, 'r', encoding='utf-8') as file:
                    text = file.read()
                    self.doc_counter += 1  # Increment for each new document
                    yield (self.doc_counter, text)

    def basic_preprocess(self, text):
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
        Generates shingles from the text of a document using the configured shingle size.
        
        Parameters:
            text (str): The text content of a document preprocessed for shingling.
        
        Returns:
            set: A set of unique shingles represented as CRC32 hashed values.
        """
        words = text.split()
        shingles = set()
        for i in range(len(words) - self.shingle_size + 1):
            shingle = ' '.join(words[i:i+self.shingle_size])
            crc = binascii.crc32(shingle.encode('utf-8')) & 0xffffffff
            shingles.add(crc)
        return shingles
    
    
    def get_docs_as_sets_and_texts(self):
        """
        Preprocesses text for shingling and generates shingles for each document, 
        storing them in docs_as_sets and also returns the preprocessed texts for clustering.
        
        Returns:
            tuple: A tuple containing a dictionary with document IDs as keys and sets of shingles,
                   and a list of tuples with document IDs and preprocessed texts for clustering.
        """
        texts = []
        for docID, text in self.load_documents():
            processed_text = self.basic_preprocess(text)
            shingles = self.generate_shingles(processed_text)
            self.docs_as_sets[docID] = shingles
            texts.append((docID, processed_text))
        return self.docs_as_sets, texts
