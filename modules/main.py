from data_loader import DataLoader
from semantic_clusterer import SemanticClusterer
from lexical_proximity_algorithm import LexicalProximityAlgorithm
from jaccard import JaccardSimilarityCalculator
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Path to the directory containing the text documents
directory_path = "D:/uni/4 курс/2 семестр/Диплом/datasets/corpus/corpus/wikipedia_documents_test/machined/data5"
shingle_size = 3  # Shingle size for generating shingles

# Initialize DataLoader and load documents as sets of shingles
logger.info("Initializing DataLoader and loading documents...")
loader = DataLoader(directory_path, shingle_size)
docs_as_sets, documents = loader.get_docs_as_sets_and_texts()

# Initialize the SemanticClusterer
logger.info("Initializing SemanticClusterer and processing documents...")
clusterer = SemanticClusterer()
labels, silhouette = clusterer.process_documents(documents)
logger.info("Cluster Labels: %s", labels)
logger.info("Silhouette Score: %f", silhouette)

# Grouping documents by clusters
clusters = {}
for (doc_id, doc_text), label in zip(documents, labels):
    if label not in clusters:
        clusters[label] = []
    clusters[label].append(doc_id)  # Store only doc_id for later reference

# Debugging: Check if clusters are populated
logger.info(f"Number of clusters: {len(clusters)}")
for label, doc_ids in clusters.items():
    logger.info(f"Cluster {label} has {len(doc_ids)} documents")

#Similarity for docs in each cluster
if not clusters:
    logger.warning("No clusters found. Please check the clustering process.")
else:
    for label, doc_ids in clusters.items():
        logger.info(f"Running MinHash for Cluster {label}")
        try:
            cluster_docs_as_sets = {doc_id: docs_as_sets[doc_id] for doc_id in doc_ids}  # Filter docs_as_sets for this cluster
        except KeyError as e:
            logger.error(f"KeyError: {e} - Check if doc_id is present in docs_as_sets")
            continue
        minhash = LexicalProximityAlgorithm(cluster_docs_as_sets, num_hashes=100, similarity_threshold=0.5)
        signatures = minhash.generate_minhash_signatures()
        similarities = minhash.calculate_similarities(signatures)

        for doc1, doc2, similarity in similarities:
            if similarity > 0.1:  # Adjust threshold as needed
                logger.info(f"Cluster {label}: Document {doc1} is similar to Document {doc2} with similarity {similarity:.8f}")
cluster_top_terms = clusterer.get_top_terms_per_cluster()

'''# Jaccard similarity for comparison
logger.info("Calculating Jaccard similarities for comparison...")
similarity_calculator = JaccardSimilarityCalculator(docs_as_sets)

# Calculate Jaccard similarities
jaccard_similarities = similarity_calculator.calculate_jaccard()

# Print the results
for doc_pair in jaccard_similarities:
    if doc_pair[2] > 0.1:
        logger.info(f"Jaccard similarity between {doc_pair[0]} and {doc_pair[1]}: {doc_pair[2]:.8f}")'''