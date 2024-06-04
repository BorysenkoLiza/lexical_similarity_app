import sys
import os
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from gensim.models.keyedvectors import KeyedVectors

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from modules.semantic_clusterer import SemanticClusterer

class TestSemanticClusterer(unittest.TestCase):

    @patch('modules.semantic_clusterer.KeyedVectors.load_word2vec_format')
    def setUp(self, mock_load):
        # Mock the Word2Vec model
        self.mock_model = MagicMock(spec=KeyedVectors)
        self.mock_model.vector_size = 300
        mock_load.return_value = self.mock_model
        
        # Initialize the SemanticClusterer with the mock model
        self.clusterer = SemanticClusterer(vector_size=300, min_cluster=2, batch_size=100, seed=42)
        
        # Sample documents
        self.documents = [
            "Hello world",
            "Another document",
            "Hello again",
            "More text data",
            "Document clustering",
            "Text analysis"
        ]
        
        # Mock Word2Vec responses
        self.mock_model.__getitem__.side_effect = lambda x: np.random.rand(300) if x in ["hello", "world", "another", "document", "again", "more", "text", "data", "clustering", "analysis"] else None

    def test_clean_text(self):
        raw_text = "Hello world! This is a test. [some text] ... more text."
        expected_tokens = ["hello", "world", "test", "text","text"]
        cleaned_tokens = self.clusterer.clean_text(raw_text)
        self.assertEqual(cleaned_tokens, expected_tokens)

    def test_vectorize_docs(self):
        vectors = self.clusterer.vectorize_docs(self.documents)
        print(vectors)
        self.assertEqual(vectors.shape, (len(self.documents), 300))


if __name__ == "__main__":
    unittest.main()