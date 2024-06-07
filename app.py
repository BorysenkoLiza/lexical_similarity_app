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
import secrets

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'zip'}
app.secret_key = secrets.token_hex(16)  # Generate a unique secret key

# Global store for DataFrames
data_store = {}

def allowed_file(filename):
    """
    Check if the uploaded file is allowed (i.e., is a zip file).
    
    Parameters:
        filename (str): Name of the file to check.
    
    Returns:
        bool: True if the file is allowed, False otherwise.
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET'])
def index():
    """
    Render the index page.
    
    Returns:
        Response: Rendered HTML of the index page.
    """
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    """
    Handle the file upload, process the uploaded zip file, and redirect to results.
    
    Returns:
        Response: Redirect to results page or render index page with an error.
    """
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
    """
    Extract the uploaded zip file to a temporary directory and move text files to the final extraction directory.
    
    Parameters:
        zip_path (str): Path to the uploaded zip file.
    
    Returns:
        str: Path to the final extraction directory.
    """
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
    """
    Process the documents in the extracted folder, perform clustering, and calculate similarities.
    
    Parameters:
        extract_folder (str): Path to the folder containing extracted text files.
        num_clusters (int): Number of clusters for document clustering.
        shingle_size (int): Shingle size for document shingling.
        num_hashes (int): Number of hash functions for MinHash.
        vector_size (int): Vector size for Word2Vec model.
        similarity_threshold (float): Similarity threshold for considering documents as similar.
    
    Returns:
        str: Unique result ID for the processed documents.
    """
    # Initialize DataLoader and load documents as sets of shingles
    logger.info("Initializing DataLoader and loading documents...")
    loader = DataLoader(extract_folder, shingle_size)
    documents_df = loader.get_proccessed_documents()

    # Initialize the SemanticClusterer
    logger.info("Initializing SemanticClusterer and processing documents...")
    clusterer = SemanticClusterer(vector_size=vector_size, num_cluster=num_clusters)
    documents_df, silhouette,cluster_top_terms = clusterer.process_documents(documents_df)
    logger.info("Silhouette Score: %f", silhouette)

    # Grouping documents by clusters
    cluster_groups = documents_df.groupby('Cluster')

    # Similarity for docs in each cluster
    cluster_similarities = {}
    clusters = {}
    word_counts = {}
    logger.info(f"Number of clusters: {len(cluster_groups)}")

    for label, group in cluster_groups:
        logger.info(f"Running MinHash for Cluster {label}")
        cluster_docs_df = group[['DocID', 'DocName', 'Shingles', 'WordCount']].copy()

        #Convert DataFrame to dictionary format: {DocID: shingles_set, ...}
        docs_as_sets = {row['DocID']: row['Shingles'] for index, row in cluster_docs_df.iterrows()}
        clusters[label] = [{'DocID': row['DocID'], 'DocName': row['DocName'], 'WordCount': row['WordCount']} for index, row in cluster_docs_df.iterrows()] 
        for index, row in cluster_docs_df.iterrows():
            word_counts[row['DocID']] = row['WordCount'] # Store both document IDs and names for the cluster
        try:
            minhash = LexicalProximityAlgorithm(docs_as_sets, num_hashes=num_hashes)
            signatures = minhash.generate_minhash_signatures()
            similarities = minhash.calculate_similarities(signatures)
            similar_pairs = sorted([pair for pair in similarities if pair[2] >= similarity_threshold], key=lambda x: x[2], reverse=True)
            cluster_similarities[label] = similar_pairs
        except KeyError as e:
            logger.error(f"KeyError: {e} - Check if DocID is present in docs_as_sets")

    # Store DataFrame and results in the global store
    result_id = secrets.token_hex(8)
    logger.info(f"Storing results with result_id: {result_id}")
    data_store[result_id] = {
        'clusters': clusters,
        'cluster_top_terms': cluster_top_terms,
        'cluster_similarities': cluster_similarities,
        'documents_df': documents_df,
        'word_counts': word_counts 
    }
    # Clean up the uploads folder
    clean_uploads_folder()

    return result_id

def clean_uploads_folder():
    """
    Remove all files from the uploads folder.
    """
    folder = app.config['UPLOAD_FOLDER']
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            logger.error(f'Failed to delete {file_path}. Reason: {e}')

@app.route('/results')
def results():
    """
    Display the results page with the processed documents and their clusters.
    
    Returns:
        Response: Rendered HTML of the results page.
    """
    result_id = session.get('result_id')
    logger.info(f"Fetching results for result_id: {result_id}")
    if result_id not in data_store:
        logger.error(f"Results not found for result_id: {result_id}")
        return "Results not found", 404

    results = data_store[result_id]
    similar_docs = session.get('similar_docs', None)
    file_name = session.get('file_name', None)
    doc_names = {row['DocID']: row['DocName'] for index, row in results['documents_df'].iterrows()}
    word_counts = results['word_counts']
    total_similar_pairs = sum(len(pairs) for pairs in results['cluster_similarities'].values())
    return render_template('results.html', clusters=results['clusters'], 
                           cluster_top_terms=results['cluster_top_terms'], 
                           cluster_similarities=results['cluster_similarities'], 
                           similar_docs=similar_docs, 
                           file_name=file_name,
                           doc_names=doc_names,
                           word_counts=word_counts,
                           total_similar_pairs=total_similar_pairs)

@app.route('/find_similar', methods=['POST'])
def find_similar():
    """
    Find similar documents for a given document name within its cluster and redirect to the results page.
    
    Returns:
        Response: Redirect to the results page or error message if document not found.
    """
    result_id = session.get('result_id')
    logger.info(f"Finding similar documents for result_id: {result_id}")
    if result_id not in data_store:
        logger.error(f"Results not found for result_id: {result_id}")
        return "Results not found", 404

    file_name = request.form.get('fileName')
    results = data_store[result_id]
    documents_df = results['documents_df']

    # Find the document ID and cluster for the given file name
    doc_info = documents_df[documents_df['DocName'] == file_name]
    if doc_info.empty:
        logger.error(f"Document with name {file_name} not found")
        return f"Document with name {file_name} not found", 404

    doc_id = doc_info.iloc[0]['DocID']
    cluster_id = doc_info.iloc[0]['Cluster']

    logger.info(f"Document {file_name} (ID: {doc_id}) found in cluster {cluster_id}")

    # Find similar documents within the cluster
    similar_docs = []
    for doc1, doc2, similarity in results['cluster_similarities'][cluster_id]:
        if (doc1 == doc_id or doc2 == doc_id) and (doc1 != doc2):
            other_doc_id = doc1 if doc1 != doc_id else doc2
            similar_docs.append((other_doc_id, similarity))

    session['similar_docs'] = similar_docs
    session['file_name'] = file_name
    logger.info(f"Document {file_name} (ID: {doc_id}) has {len(similar_docs)} similar documents")
    return redirect(url_for('results'))

if __name__ == '__main__':
    app.run(debug=True)