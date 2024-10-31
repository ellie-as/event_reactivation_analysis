# imports:
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import numpy as np
from rank_bm25 import BM25Okapi

# Load models
embedding_models_dict = {
    'all-mpnet-base-v2': SentenceTransformer('all-mpnet-base-v2'),
    'multi-qa-mpnet-base-dot-v1': SentenceTransformer('multi-qa-mpnet-base-dot-v1'),
    'multi-qa-distilbert-cos-v1': SentenceTransformer('multi-qa-distilbert-cos-v1')
}

# Metrics for existing methods
distance_metrics_dict = {
    'BoW_vector_tfidf': 'cosine',
    'BoW_vector_count': 'cosine',
    'embedding_all-mpnet-base-v2': 'dot',
    'embedding_multi-qa-mpnet-base-dot-v1': 'dot',
    'embedding_multi-qa-distilbert-cos-v1': 'dot'
}

# Load and process data into event-level paragraphs
def load_data_to_dict(file_path):
    events_dict = {}
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        for line in f:
            line = line.strip()
            if not line or '\t' not in line:
                continue  # Skip empty or incorrectly formatted lines

            event_id_full, text = line.split('\t', 1)
            main_event_id = event_id_full.split('.')[0]

            if main_event_id in events_dict:
                events_dict[main_event_id] += " " + text
            else:
                events_dict[main_event_id] = text

    return events_dict

# Normalize by lower triangle values (excluding diagonal)
def normalize_heatmap(matrix):
    lower_triangle_values = matrix[np.tril_indices_from(matrix, k=-1)]
    vmin = lower_triangle_values.min()
    vmax = lower_triangle_values.max()
    return vmin, vmax

# Process events data
def process_events_data(path_to_txt_file):
    # Load raw text
    raw_events_dict = load_data_to_dict(path_to_txt_file)

    # Generate embeddings and BoW vectors
    embeddings_dict = get_embeddings(raw_events_dict)
    BoW_vectors_dict = get_BoW_vectors(raw_events_dict)

    # Combine embeddings and BoW vectors into events_dict for similarity heatmaps
    processed_events_dict = {
        event_id: {**embeddings_dict[event_id], **BoW_vectors_dict[event_id]}
        for event_id in raw_events_dict.keys()
    }

    # Produce similarity heatmaps using processed embeddings and vectors
    create_and_save_heatmaps(processed_events_dict)

    # Produce Jaccard and BM25 heatmaps with raw text
    create_and_save_jaccard_heatmap(raw_events_dict)
    create_and_save_bm25_heatmap(raw_events_dict)

    # Pickle processed events_dict for further use
    with open('processed_events_dict.pkl', 'wb') as f:
        pickle.dump(processed_events_dict, f)

# Generate embeddings
def get_embeddings(events_dict, models=['all-mpnet-base-v2', 'multi-qa-mpnet-base-dot-v1', 'multi-qa-distilbert-cos-v1']):
    embeddings_dict = {}
    for event_id, paragraph in events_dict.items():
        embeddings_dict[event_id] = {}
        for model in models:
            emb = get_embedding(paragraph, model)
            embeddings_dict[event_id][f'embedding_{model}'] = emb
    return embeddings_dict

# Generate BoW vectors
def get_BoW_vectors(events_dict):
    event_ids = list(events_dict.keys())
    paragraphs = [events_dict[event_id] for event_id in event_ids]

    tfidf_vectorizer = TfidfVectorizer()
    count_vectorizer = CountVectorizer()

    tfidf_matrix = tfidf_vectorizer.fit_transform(paragraphs).toarray()
    count_matrix = count_vectorizer.fit_transform(paragraphs).toarray()

    BoW_vectors_dict = {}
    for idx, event_id in enumerate(event_ids):
        BoW_vectors_dict[event_id] = {
            'BoW_vector_tfidf': tfidf_matrix[idx],
            'BoW_vector_count': count_matrix[idx]
        }

    return BoW_vectors_dict

# Get embedding with sentence transformers library
def get_embedding(txt, model_name):
    model = embedding_models_dict[model_name]
    embedding = model.encode(txt)
    return embedding

# Create and save similarity heatmaps for embeddings and BoW vectors
def create_and_save_heatmaps(events_dict, zero_out=True, normalize_by_lower_triangle=True):
    for representation, metric in distance_metrics_dict.items():
        vectors = np.array([data[representation] for data in events_dict.values()])

        # Calculate pairwise similarities based on metric
        if metric == 'dot':
            pairwise_similarities = util.dot_score(vectors, vectors).cpu().numpy()
        else:
            pairwise_distances = cdist(vectors, vectors, metric=metric)
            pairwise_similarities = 1 - pairwise_distances

        if zero_out:
            pairwise_similarities[np.triu_indices_from(pairwise_similarities, k=1)] = 0

        if normalize_by_lower_triangle:
            vmin, vmax = normalize_heatmap(pairwise_similarities)
        else:
            vmin, vmax = None, None

        plt.figure(figsize=(10, 8))
        sns.heatmap(pairwise_similarities, cmap="viridis", vmin=vmin, vmax=vmax,
                    xticklabels=events_dict.keys(), yticklabels=events_dict.keys())
        plt.title(f"Pairwise similarity heatmap - {representation}")
        plt.savefig(f'{representation}_event_level_similarity_heatmap.png')
        plt.close()

# Create and save Jaccard similarity heatmap with zero_out option
def create_and_save_jaccard_heatmap(events_dict, zero_out=True, normalize_by_lower_triangle=True):
    event_ids = list(events_dict.keys())
    paragraphs = [events_dict[event_id] if isinstance(events_dict[event_id], str) else events_dict[event_id].get('text', '') for event_id in event_ids]

    count_vectorizer = CountVectorizer(binary=True)
    binary_matrix = count_vectorizer.fit_transform(paragraphs).toarray()

    jaccard_similarities = np.zeros((len(event_ids), len(event_ids)))
    for i in range(len(event_ids)):
        for j in range(len(event_ids)):
            intersection = np.sum(np.minimum(binary_matrix[i], binary_matrix[j]))
            union = np.sum(np.maximum(binary_matrix[i], binary_matrix[j]))
            jaccard_similarities[i, j] = intersection / union if union != 0 else 0

    if zero_out:
        jaccard_similarities[np.triu_indices_from(jaccard_similarities, k=1)] = 0

    if normalize_by_lower_triangle:
        vmin, vmax = normalize_heatmap(jaccard_similarities)
    else:
        vmin, vmax = None, None

    plt.figure(figsize=(10, 8))
    sns.heatmap(jaccard_similarities, cmap="viridis", vmin=vmin, vmax=vmax,
                xticklabels=event_ids, yticklabels=event_ids)
    plt.title("Pairwise Jaccard Similarity Heatmap")
    plt.savefig('jaccard_similarity_heatmap.png')
    plt.close()

# Create and save BM25 similarity heatmap with zero_out option
def create_and_save_bm25_heatmap(events_dict, zero_out=True, normalize_by_lower_triangle=True):
    event_ids = list(events_dict.keys())
    tokenized_paragraphs = [events_dict[event_id].split() if isinstance(events_dict[event_id], str) else events_dict[event_id].get('text', '').split() for event_id in event_ids]

    bm25 = BM25Okapi(tokenized_paragraphs)
    bm25_similarities = np.zeros((len(event_ids), len(event_ids)))
    for i, query in enumerate(tokenized_paragraphs):
        scores = bm25.get_scores(query)
        bm25_similarities[i] = scores

    if zero_out:
        bm25_similarities[np.triu_indices_from(bm25_similarities, k=1)] = 0

    if normalize_by_lower_triangle:
        vmin, vmax = normalize_heatmap(bm25_similarities)
    else:
        vmin, vmax = None, None

    plt.figure(figsize=(10, 8))
    sns.heatmap(bm25_similarities, cmap="viridis", vmin=vmin, vmax=vmax,
                xticklabels=event_ids, yticklabels=event_ids)
    plt.title("Pairwise BM25 Similarity Heatmap")
    plt.savefig('bm25_similarity_heatmap.png')
    plt.close()
