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

    Attributes:
        docs_as_sets (dict): Dictionary mapping document IDs to sets of shingles.
        num_hashes (int): Number of hash functions used in MinHash.
        num_docs (int): Number of documents.
        max_shingle (int): Maximum value for shingle encoding.
        next_prime (int): Prime number larger than max_shingle for hash functions.
        coeff_a (list): List of random coefficients 'a' for hash functions.
        coeff_b (list): List of random coefficients 'b' for hash functions.
    
    Methods:
        find_next_prime(n): Finds the next prime number greater than n.
        pick_random_coeffs(max_shingle): Generates unique random coefficients for hash functions.
        generate_minhash_signatures(): Computes MinHash signatures for each document.
        calculate_similarities(signatures): Calculates pairwise similarities between documents.
    """

    def __init__(self, docs_as_sets, num_hashes=100):
            """
            Initializes the LexicalProximityAlgorithm
            
            Parameters:
                docs_as_sets (dict): Dictionary with document IDs as keys and sets of shingles as values.
                num_hashes (int): Number of hash functions to use in the MinHash algorithm.
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
        Parameters:
            n (int): The number to find the next prime for.

        Returns:
            int: The next prime number greater than n.
        """
        return sympy.nextprime(n)
    
    def pick_random_coeffs(self, max_shingle):
        """
        Generates a list of unique random coefficients for the hash functions h(x) = (a*x + b) % c

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
        
        Parameters:
            signatures (dict): Dictionary with document IDs as keys and MinHash signatures as values.

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