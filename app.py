from flask import Flask, request, render_template, redirect, session, url_for
import os
import zipfile
import logging
import shutil
import tempfile
from werkzeug.utils import secure_filename
from modules.data_loader import DataLoader
from modules.semantic_clusterer import SemanticClusterer
from modules.lexical_proximity_algorithm import LexicalProximityAlgorithm
from modules.jaccard import JaccardSimilarityCalculator
import secrets  # To generate a secret key
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'zip'}
app.secret_key = secrets.token_hex(16)  # Generate a unique secret key

# Global store for DataFrames
data_store = {}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'zipFile' not in request.files:
        return redirect(request.url)
    file = request.files['zipFile']
    num_clusters = request.form.get('numClusters', type=int, default=10)
    shingle_size = request.form.get('shingleSize', type=int, default=3)
    num_hashes = request.form.get('numHashes', type=int, default=100)
    vector_size = request.form.get('vectorSize', type=int, default=300)
    similarity_threshold = request.form.get('similarityThreshold', type=float, default=0.5)

    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        zip_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(zip_path)
        extracted_folder = process_zip(zip_path)
        result_id = process_documents(extracted_folder, num_clusters, shingle_size, num_hashes, vector_size, similarity_threshold)
        logger.info(f"Redirecting to results with result_id: {result_id}")
        session['result_id'] = result_id  # Store result_id in session
        return redirect(url_for('results'))
    return render_template('index.html')

def process_zip(zip_path):
    # Create a temporary directory for extraction
    temp_extract_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_extract_dir)
    # Move all .txt files to the final extraction directory
    extract_folder = os.path.join(app.config['UPLOAD_FOLDER'], os.path.splitext(os.path.basename(zip_path))[0])
    os.makedirs(extract_folder, exist_ok=True)
    for root, dirs, files in os.walk(temp_extract_dir):
        for file in files:
            if file.endswith('.txt'):
                shutil.move(os.path.join(root, file), os.path.join(extract_folder, file))
    shutil.rmtree(temp_extract_dir)
    return extract_folder

def process_documents(extract_folder, num_clusters, shingle_size, num_hashes, vector_size, similarity_threshold):
    # Initialize DataLoader and load documents as sets of shingles
    logger.info("Initializing DataLoader and loading documents...")
    loader = DataLoader(extract_folder, shingle_size)
    documents_df = loader.get_docs_as_sets_and_texts()

    # Initialize the SemanticClusterer
    logger.info("Initializing SemanticClusterer and processing documents...")
    clusterer = SemanticClusterer(vector_size=vector_size, min_cluster=num_clusters)
    documents_df, silhouette = clusterer.process_documents(documents_df, train_new_model=False)
    logger.info("Silhouette Score: %f", silhouette)

    # Grouping documents by clusters
    cluster_groups = documents_df.groupby('Cluster')

    # Similarity for docs in each cluster
    cluster_similarities = {}
    clusters = {}
    logger.info(f"Number of clusters: {len(cluster_groups)}")

    for label, group in cluster_groups:
        logger.info(f"Running MinHash for Cluster {label}")
        cluster_docs_df = group[['DocID','DocName','Shingles']].copy() 
        #Convert DataFrame to dictionary format: {DocID: shingles_set, ...}
        docs_as_sets = {row['DocID']: row['Shingles'] for index, row in cluster_docs_df.iterrows()}
        clusters[label] = [{'DocID': row['DocID'], 'DocName': row['DocName']} for index, row in cluster_docs_df.iterrows()]  # Store both document IDs and names for the cluster
        try:
            minhash = LexicalProximityAlgorithm(docs_as_sets, num_hashes=num_hashes)
            signatures = minhash.generate_minhash_signatures()
            similarities = minhash.calculate_similarities(signatures)
            similar_pairs = sorted(similarities, key=lambda x: x[2], reverse=True)[:5] #top five similar pairs for cluster
            cluster_similarities[label] = similar_pairs
            #similarity_calculator = JaccardSimilarityCalculator(docs_as_sets)
            #jaccard_similarities = similarity_calculator.calculate_jaccard()
            for doc1, doc2, similarity in similarities:
                if similarity > similarity_threshold:  # Adjust threshold as needed
                    logger.info(f"Cluster {label}: Document {doc1} is similar to Document {doc2} with similarity {similarity:.8f}")
        except KeyError as e:
            logger.error(f"KeyError: {e} - Check if DocID is present in docs_as_sets")

    cluster_top_terms = clusterer.get_top_terms_per_cluster()
    # Store DataFrame and results in the global store
    result_id = secrets.token_hex(8)
    logger.info(f"Storing results with result_id: {result_id}")
    data_store[result_id] = {
        'clusters': clusters,
        'cluster_top_terms': cluster_top_terms,
        'cluster_similarities': cluster_similarities,
        'documents_df': documents_df
    }

    return result_id

@app.route('/results')
def results():
    result_id = session.get('result_id')
    logger.info(f"Fetching results for result_id: {result_id}")
    if result_id not in data_store:
        logger.error(f"Results not found for result_id: {result_id}")
        return "Results not found", 404

    results = data_store[result_id]
    return render_template('results.html', clusters=results['clusters'], cluster_top_terms=results['cluster_top_terms'], cluster_similarities=results['cluster_similarities'])

if __name__ == '__main__':
    app.run(debug=True)