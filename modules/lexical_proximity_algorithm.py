import random
import time
import sympy
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LexicalProximityAlgorithm:
    """
    This class implements the MinHash algorithm to estimate the similarity between documents based on sets of shingles.
    It is designed to compute MinHash signatures for each document and to evaluate pairwise similarities between
    documents using these signatures. A similarity threshold is used to identify and report pairs of documents that
    are considered similar based on their Jaccard similarity estimate.

    The algorithm is particularly useful for large datasets where direct computation of Jaccard similarities would be
    computationally expensive. It provides an efficient probabilistic approach to detect similar documents in a large corpus.

    Attributes:
        docs_as_sets (dict): A dictionary mapping each document ID to its corresponding set of shingles.
        num_hashes (int): The number of hash functions used to compute MinHash signatures, impacting the accuracy of similarity estimates.
        similarity_threshold (float): The minimum similarity score required to consider two documents as similar.
        num_docs (int): The total number of documents being analyzed.
        max_shingle (int): The maximum value for shingle encoding, used in hash function calculations.
        next_prime (int): A prime number larger than max_shingle, used to ensure a good distribution for hash functions.
        coeff_a (list): Random coefficients 'a' used in the hash functions.
        coeff_b (list): Random coefficients 'b' used in the hash functions.
    
    Methods:
        _pick_random_coeffs(): Generates a list of unique random coefficients for the hash functions.
        generate_minhash_signatures(): Computes MinHash signatures for each document using the random hash functions.
        calculate_similarities(signatures): Calculates and returns the pairwise similarities between documents based on their MinHash signatures.
    """

    def __init__(self, docs_as_sets, num_hashes=100):
            """
            Initializes the LexicalProximityAlgorithm with a dictionary of documents and their shingle sets.
            
            Parameters:
                docs_as_sets (dict): Dictionary with document IDs as keys and sets of shingles as values.
                num_hashes (int): Number of hash functions to use in the MinHash algorithm.
                similarity_threshold (float): The threshold for considering documents as similar.
            """
            self.docs_as_sets = docs_as_sets
            self.num_hashes = num_hashes
            self.num_docs = len(docs_as_sets)
            self.max_shingle = 2**32 - 1
            self.next_prime = self.find_next_prime(self.max_shingle)
            self.coeff_a = self.pick_random_coeffs(self.max_shingle)
            self.coeff_b = self.pick_random_coeffs(self.max_shingle)
            logger.info("LexicalProximityAlgorithm initialized with %d documents", self.num_docs)
    
    def find_next_prime(self, n):
        """
        Finds the next prime number greater than a given number n.
        """
        return sympy.nextprime(n)
    
    def pick_random_coeffs(self, max_shingle):
        """
        Helper method to generate a list of unique random coefficients for the hash functions used in MinHash.
        Our random hash function will take the form of: 
        h(x) = (a*x + b) % c
        Where 'x' is the input value, 'a' and 'b' are random coefficients, 
        and 'c' is a prime number just greater than maxShingleID.
        Returns:
            list: A list of unique random integers.
        """
        rand_list = []
        while len(rand_list) < self.num_hashes:
            rand_index = random.randint(0, max_shingle)
            if rand_index not in rand_list:
                rand_list.append(rand_index)
        return rand_list

    def generate_minhash_signatures(self):
        """
        Generates MinHash signatures for all documents based on their shingle sets.
        
        Returns:
            dict: A dictionary with document IDs as keys and their MinHash signatures as values.
        """
        start_time = time.time()
        signatures = {}
        for doc_id in self.docs_as_sets:
            shingleIDSet = self.docs_as_sets[doc_id]
            signature = []
            for i in range(0,self.num_hashes):
                min_hash = self.next_prime + 1
                for shingleID in shingleIDSet:
                    hash_code = (self.coeff_a[i] * shingleID + self.coeff_b [i]) % self.next_prime
                    if hash_code < min_hash:
                        min_hash = hash_code
                signature.append(min_hash)
            signatures[doc_id] = signature
        elapsed_time = time.time() - start_time
        logger.info("MinHash signatures generated in %.2f seconds", elapsed_time)
        return signatures

    def calculate_similarities(self, signatures):
        """
        Calculates similarities between all pairs of documents based on their MinHash signatures.
        
        Returns:
            list: A list of tuples (doc_id1, doc_id2, similarity) for document pairs with similarities.
        """
        start_time = time.time()
        doc_ids = list(signatures.keys())
        similarities = []
        for i in range(self.num_docs):
            doc_id1 = doc_ids[i]
            sig1 = signatures[doc_id1]
            for j in range(i + 1, self.num_docs):
                doc_id2 = doc_ids[j]
                sig2 = signatures[doc_id2]
                count = 0
                for k in range(0, self.num_hashes):
                    count += (sig1[k] == sig2[k])
                sim = count / self.num_hashes
                similarities.append((doc_id1, doc_id2, sim))
        elapsed_time = time.time() - start_time
        logger.info("Similarities calculated in %.2f seconds", elapsed_time)
        return similarities