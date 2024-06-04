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
import secrets  # To generate a secret key

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'zip'}
app.secret_key = secrets.token_hex(16)  # Generate a unique secret key

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
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        zip_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(zip_path)
        extracted_folder = process_zip(zip_path)
        process_documents(extracted_folder, num_clusters, shingle_size)
        return redirect(url_for('results'))
    return render_template('index.html')

def process_zip(zip_path):
    # Create a temporary directory for extraction
    temp_extract_dir = tempfile.mkdtemp()

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_extract_dir)
    
    # Create the final extraction directory
    extract_folder = os.path.join(app.config['UPLOAD_FOLDER'], os.path.splitext(os.path.basename(zip_path))[0])
    os.makedirs(extract_folder, exist_ok=True)

    # Move all .txt files to the final extraction directory
    for root, dirs, files in os.walk(temp_extract_dir):
        for file in files:
            if file.endswith('.txt'):
                full_path = os.path.join(root, file)
                new_path = os.path.join(extract_folder, file)
                shutil.move(full_path, new_path)

    # Cleanup the temporary directory
    shutil.rmtree(temp_extract_dir)
    
    return extract_folder

def process_documents(extract_folder, num_clusters, shingle_size):
    # Initialize DataLoader and load documents as sets of shingles
    logger.info("Initializing DataLoader and loading documents...")
    loader = DataLoader(extract_folder, shingle_size)
    docs_as_sets, documents = loader.get_docs_as_sets_and_texts()

    # Initialize the SemanticClusterer
    logger.info("Initializing SemanticClusterer and processing documents...")
    clusterer = SemanticClusterer(num_clusters)  # Assuming SemanticClusterer accepts num_clusters
    labels, silhouette = clusterer.process_documents([doc[1] for doc in documents])
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

    # Similarity for docs in each cluster
    cluster_similarities = {}
    if not clusters:
        logger.warning("No clusters found. Please check the clustering process.")
    else:
        for label, doc_ids in clusters.items():
            logger.info(f"Running MinHash for Cluster {label}")
            try:
                cluster_docs_as_sets = {doc_id: docs_as_sets[doc_id] for doc_id in doc_ids}
            except KeyError as e:
                logger.error(f"KeyError: {e} - Check if doc_id is present in docs_as_sets")
                continue
            minhash = LexicalProximityAlgorithm(cluster_docs_as_sets, num_hashes=100, similarity_threshold=0.5)
            signatures = minhash.generate_minhash_signatures()
            similarities = minhash.calculate_similarities(signatures)

            similar_pairs = sorted(similarities, key=lambda x: x[2], reverse=True)[:5] #top five similar pairs for cluster
            cluster_similarities[label] = similar_pairs

            for doc1, doc2, similarity in similarities:
                if similarity > 0.1:  # Adjust threshold as needed
                    logger.info(f"Cluster {label}: Document {doc1} is similar to Document {doc2} with similarity {similarity:.8f}")
    cluster_top_terms = clusterer.get_top_terms_per_cluster()
    session['clusters'] = {int(k): v for k, v in clusters.items()}  # Convert keys to int
    session['cluster_top_terms'] = {int(k): v for k, v in cluster_top_terms.items()}  # Convert keys to int
    session['cluster_similarities'] = {int(k): v for k, v in cluster_similarities.items()}  # Convert keys to int
    session['silhouette'] = float(silhouette)

@app.route('/results')
def results():
    clusters = session.get('clusters', {})
    cluster_top_terms = session.get('cluster_top_terms', {})
    cluster_similarities = session.get('cluster_similarities', {})
    silhouette = session.get('silhouette', 0)
    return render_template('results.html', clusters=clusters, cluster_top_terms=cluster_top_terms, cluster_similarities=cluster_similarities, silhouette=silhouette)

if __name__ == '__main__':
    app.run(debug=True)