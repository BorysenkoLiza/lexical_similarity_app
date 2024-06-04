class JaccardSimilarityCalculator:
    """
    A class to calculate Jaccard similarities between sets of items (e.g., shingles).
    
    Attributes:
        docs_as_sets (dict): A dictionary with document IDs as keys and sets of items (e.g., shingles) as values.
    """
    
    def __init__(self, docs_as_sets):
        """
        Initializes the JaccardSimilarityCalculator with a dictionary of documents and their associated sets.
        
        Parameters:
            docs_as_sets (dict): Dictionary with document IDs as keys and sets of items as values.
        """
        self.docs_as_sets = docs_as_sets

    def calculate_jaccard(self):
        """
        Calculates the Jaccard similarity between all pairs of documents and returns a list of tuples with document pairs and their similarities.
        
        Returns:
            list: A list of tuples, where each tuple contains two document IDs and their Jaccard similarity.
        """
        doc_ids = list(self.docs_as_sets.keys())
        similarities = []
        num_docs = len(doc_ids)
        
        for i in range(num_docs):
            set1 = self.docs_as_sets[doc_ids[i]]
            for j in range(i + 1, num_docs):
                set2 = self.docs_as_sets[doc_ids[j]]
                intersection = len(set1.intersection(set2))
                union = len(set1.union(set2))
                jaccard_similarity = intersection / union if union != 0 else 0
                similarities.append((doc_ids[i], doc_ids[j], jaccard_similarity))
        
        return similarities