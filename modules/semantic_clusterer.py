import re
import string
import logging


import nltk
import numpy as np
import pandas as pd

from gensim.models import Word2Vec, KeyedVectors

from nltk import word_tokenize
from nltk.corpus import stopwords as nltk_stopwords

from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score, silhouette_samples

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

nltk.download("stopwords")
nltk.download("punkt")

class SemanticClusterer:
    """
    A class to perform document clustering using Word2Vec vectors and MiniBatchKMeans.
    
    This class loads a pretrained Word2Vec model, preprocesses text documents, vectorizes them, and then clusters the vectors
    using the MiniBatchKMeans algorithm. It also evaluates the clustering quality using silhouette scores.
    
    Attributes:
        vector_size (int): Dimension of the Word2Vec vectors.
        min_cluster (int): Minimum number of clusters to use in KMeans.
        batch_size (int): Batch size for MiniBatchKMeans.
        seed (int): Seed for random number generation.
        stopwords (set): A set of stopwords to use.
        km (MiniBatchKMeans): Instance of the MiniBatchKMeans clustering algorithm.
        model (KeyedVectors): The pretrained Word2Vec model.
    
    Methods:
        clean_text(text): Pre-process text and generate tokens.
        vectorize_docs(documents): Convert tokenized documents into vectors using the Word2Vec model.
        cluster_documents(vectors): Cluster document vectors using MiniBatchKMeans.
        get_top_terms_per_cluster(): Return the top terms for each cluster based on the cluster centroids.
        process_documents(raw_documents): Process documents from text to clusters.
    """
    def __init__(self, vector_size=300, min_cluster=10, batch_size=500, seed=42):
        """
        Initialize the SemanticClusterer with the specified parameters and load the pretrained Word2Vec model.
        
        Parameters:
            vector_size (int, optional): Dimension of the Word2Vec vectors. Defaults to 300.
            min_cluster (int, optional): Minimum number of clusters to use in KMeans. Defaults to 10.
            batch_size (int, optional): Batch size for MiniBatchKMeans. Defaults to 500.
            seed (int, optional): Seed for random number generation. Defaults to 42.
        """
        self.vector_size = vector_size
        self.min_cluster = min_cluster
        self.batch_size = batch_size
        self.seed = seed
        self.stopwords = set(nltk_stopwords.words('english'))
        np.random.seed(seed)
        self.km = None
        self.model = KeyedVectors.load_word2vec_format(
            'D:/uni/4 курс/2 семестр/Диплом/git_thesis/Lexical_proximity/modules/GoogleNews-vectors-negative300.bin',
            binary=True,
            limit=80000
        )
        logger.info("Word2Vec model loaded successfully.")


    def clean_text(self, text):
        """
        Pre-process text and generate tokens.
        
        Parameters:
            text (str): Text to tokenize.
        
        Returns:
            list: Tokenized text.
        """
        text = str(text).lower() 
        text = re.sub(r"\[(.*?)\]", r"\1", text)  # Remove brackets but keep the text inside
        text = re.sub(r"\s+", " ", text)  # Remove multiple spaces in content
        text = re.sub(r"\w+…|…", "", text)  # Remove ellipsis (and last word)
        text = re.sub(r"(?<=\w)-(?=\w)", " ", text)  # Replace dash between words   
        text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)  # Remove punctuation
        tokens = word_tokenize(text)  # Get tokens from text
        tokens = [t for t in tokens if t not in self.stopwords]  # Remove stopwords
        tokens = ["" if t.isdigit() else t for t in tokens]  # Remove digits
        tokens = [t for t in tokens if len(t) > 1]  # Remove short tokens
        return tokens
    
    def tokenize_documents(self, documents):
        tokenized_docs = [self.clean_text(doc) for doc in documents]
        return tokenized_docs

    def vectorize_docs(self, tokenized_documents, strategy="average"):
        """
        Convert tokenized documents into vectors using the Word2Vec model.
        
        Parameters:
            tokenized_documents (list of list of str): Tokenized documents.
            strategy (str): Aggregation strategy ("average", or "min-max").
        
        Returns:
            list: List of document vectors.
        """
        features = []
        size_output = self.model.vector_size
        embedding_dict = self.model

        if strategy == "min-max":
            size_output *= 2

        if hasattr(self.model, "wv"):
            embedding_dict = self.model.wv

        for tokens in tokenized_documents:
            zero_vector = np.zeros(size_output)
            vectors = []
            for token in tokens:
                if token in embedding_dict:
                    try:
                        vectors.append(embedding_dict[token])
                    except KeyError:
                        continue
            if vectors:
                vectors = np.asarray(vectors)
                if strategy == "min-max":
                    min_vec = vectors.min(axis=0)
                    max_vec = vectors.max(axis=0)
                    features.append(np.concatenate((min_vec, max_vec)))
                elif strategy == "average":
                    avg_vec = vectors.mean(axis=0)
                    features.append(avg_vec)
                else:
                    raise ValueError(f"Aggregation strategy {strategy} does not exist!")
            else:
                features.append(zero_vector)
        return features
    
    def train_word2vec_model(self, tokenized_documents):
        """
        Train a Word2Vec model on the provided tokenized documents.
        
        Parameters:
            tokenized_documents (list of list of str): Tokenized documents.
        
        Returns:
            Word2Vec: Trained Word2Vec model.
        """
        model = Word2Vec(sentences=tokenized_documents, vector_size=self.vector_size, workers=1, seed=self.seed)
        self.model = model.wv
        logger.info("Word2Vec model trained successfully.")
        return model

    def cluster_documents(self, vectors):
        """
        Cluster document vectors using MiniBatchKMeans.
        
        Parameters:
            vectors (np.array): Document vectors.
        
        Returns:
            tuple: Cluster labels and silhouette score.
        """
        self.km = MiniBatchKMeans(n_clusters=self.min_cluster, batch_size=self.batch_size, random_state=self.seed)
        labels = self.km.fit_predict(vectors)
        silhouette_avg = silhouette_score(vectors, labels)
        sample_silhouette_values = silhouette_samples(vectors, labels)
        
        # Print silhouette values for each cluster
        logger.info("Documents clusterized successfully.")
        self.print_silhouette_scores(labels, sample_silhouette_values)
        return labels, silhouette_avg
    
    def print_silhouette_scores(self, labels, sample_silhouette_values):
        """
        Print the silhouette scores for each cluster.
        
        Parameters:
            labels (np.array): Cluster labels for each document.
            sample_silhouette_values (np.array): Silhouette values for each document.
        """
        for i in range(self.min_cluster):
            cluster_silhouette_values = sample_silhouette_values[labels == i]
            logger.info(f"Cluster {i}:")
            logger.info(f"    Number of samples: {cluster_silhouette_values.shape[0]}")
            logger.info(f"    Mean Silhouette Value: {cluster_silhouette_values.mean():.4f}")
            logger.info(f"    Min Silhouette Value: {cluster_silhouette_values.min():.4f}")
            logger.info(f"    Max Silhouette Value: {cluster_silhouette_values.max():.4f}")
    
    def get_top_terms_per_cluster(self):
        """
        Get the top terms for each cluster based on the cluster centroids.
        """
        logger.info("Top terms per cluster (based on centroids):")
        top_terms = {}
        for i in range(self.min_cluster):
            tokens_per_cluster = []
            most_representative = self.model.similar_by_vector(self.km.cluster_centers_[i], topn=10)
            for t in most_representative:
                tokens_per_cluster.append(t[0])
            top_terms[i] = tokens_per_cluster
            logger.info(f"Cluster {i} top terms: {tokens_per_cluster}")  # Log top terms for each cluster
        return top_terms

    def process_documents(self, documents_df, train_new_model=False):
        """
        Processes documents from DataFrame to clusters.
        
        Parameters:
            documents_df (pandas.DataFrame): DataFrame with columns 'DocID' and 'DocText'.
        
        Returns:
            pandas.DataFrame: DataFrame with additional columns for tokenized texts, vectors, and cluster labels.
        """
        documents_df['Tokens'] = documents_df['DocText'].apply(self.clean_text)
        if train_new_model:
            self.train_word2vec_model(documents_df['Tokens'].tolist())
        # Get document vectors
        document_vectors = self.vectorize_docs(documents_df['Tokens'].tolist())
        labels, silhouette = self.cluster_documents(document_vectors)
        documents_df['Cluster'] = labels
        return documents_df, silhouette