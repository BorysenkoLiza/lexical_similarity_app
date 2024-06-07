'''from data_loader import DataLoader
from semantic_clusterer import SemanticClusterer
from lexical_proximity_algorithm import LexicalProximityAlgorithm
from jaccard import JaccardSimilarityCalculator
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Path to the directory containing the text documents
directory_path = "D:/uni/4 курс/2 семестр/Диплом/git_thesis/bbc_data"
shingle_size = 5  # Shingle size for generating shingles

# Initialize DataLoader and load documents as sets of shingles
logger.info("Initializing DataLoader and loading documents...")
loader = DataLoader(directory_path, shingle_size)
documents_df = loader.get_proccessed_documents()

# Initialize the SemanticClusterer
logger.info("Initializing SemanticClusterer and processing documents...")
clusterer = SemanticClusterer()
documents_df, silhouette = clusterer.process_documents(documents_df,train_new_model=False)
logger.info("Silhouette Score: %f", silhouette)

# Grouping documents by clusters
cluster_groups = documents_df.groupby('Cluster')

# Debugging: Check if clusters are populated
logger.info(f"Number of clusters: {len(cluster_groups)}")
for label, group in cluster_groups:
    logger.info(f"Cluster {label} has {len(group)} documents")

# Similarity for docs in each cluster using DataFrame
for label, group in cluster_groups:
    logger.info(f"Running MinHash for Cluster {label}")
    cluster_docs_df = group[['DocID', 'Shingles']].copy()  # Ensure a copy to work with

    # Convert DataFrame to dictionary format: {DocID: shingles_set, ...}
    docs_as_sets = {row['DocID']: row['Shingles'] for index, row in cluster_docs_df.iterrows()}
    try:
        minhash = LexicalProximityAlgorithm(docs_as_sets, num_hashes=100)
        signatures = minhash.generate_minhash_signatures()
        similarities = minhash.calculate_similarities(signatures)
        similarity_calculator = JaccardSimilarityCalculator(docs_as_sets)
        jaccard_similarities = similarity_calculator.calculate_jaccard()

        for doc1, doc2, similarity in similarities:
            jaccard_similarity = next((jsim for (jdoc1, jdoc2, jsim) in jaccard_similarities if {jdoc1, jdoc2} == {doc1, doc2}), 0)
            if similarity > 0.1:  # Adjust threshold as needed
                logger.info(f"Cluster {label}: Document {doc1} is similar to Document {doc2} with similarity {similarity:.8f} and Jaccard similarity {jaccard_similarity:.8f}")
    except KeyError as e:
        logger.error(f"KeyError: {e} - Check if DocID is present in docs_as_sets")
cluster_top_terms = clusterer.get_top_terms_per_cluster()'''
