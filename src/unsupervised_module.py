from EDA import get_optimal_p
import nltk
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from nltk.stem import WordNetLemmatizer
from nltk.chunk import RegexpParser
from nltk import RegexpParser
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('wordnet', quiet=True)

# !! LLM-aided code block
def sentence_clustering(d, I=None, args=None):
    """
    Splits a document into sentences using NLTK, embeds them into vectors,
    and clusters them using Hierarchical Agglomerative Clustering.
    Uses Silhouette Score to dynamically select the optimal number of clusters.
    @input d: document dictionary 
    @input I: optional index 
    @input args: clustering arguments (encoder, model_name, n_clusters, plot_dendrogram)
    @output C: clustering solution containing cluster assignments, sentences, embeddings, 
               the number of clusters found, and the silhouette score.
    """
    if args is None:
        args = {}
        
    # 1. Extract base clustering arguments 
    encoder = args.get('encoder') 
    model_name = args.get('model_name', 'BAAI/bge-small-en-v1.5') 
    n_clusters = args.get('n_clusters', None) 
    show_plots = args.get('plot_dendrogram', False)
    
    # Extract Sentences and Embeddings 
    sentences = []
    embeddings = []
    
    doc_id = d.get('doc_id')

    # SCENARIO A: It was provided an indexing I
    if I is not None and 'semantic' in I and doc_id is not None:
        # Loop through the global index to find sentences belonging to this specific document
        for idx, global_id in enumerate(I['ids']):
            if str(global_id).startswith(f"{doc_id}_sent_"):
                sentences.append(I['texts'][idx])
                # Grab the matching pre-computed embedding
                if model_name in I['semantic']:
                    embeddings.append(I['semantic'][model_name][idx])
        
        # Convert the list of embeddings to a numpy array for the clustering math
        if embeddings:
            embeddings = np.array(embeddings)

    # SCENARIO B: It was not provided an indexing I
    if len(sentences) == 0 or len(embeddings) == 0:
        text = d.get('text', '')
        sentences = nltk.sent_tokenize(text)
        
        if len(sentences) == 0:
            return {"clusters": {}, "sentences": [], "embeddings": [], "num_clusters_found": 0, "silhouette_score": None}
            
        embeddings = encoder.encode(sentences)
        embeddings = np.array(embeddings)

    if len(sentences) < 3:
        return {"clusters": {0: list(range(len(sentences)))}, 
                "sentences": sentences, 
                "embeddings": embeddings, 
                "num_clusters_found": 1, 
                "silhouette_score": None}

    # Restrictions on the size of the number of clusters
    # Silhouette requires at least 2 clusters, and max clusters must be < n_samples
    min_allowed = min(3, len(sentences) - 1)
    max_allowed = min(7, len(sentences) - 1)
    
    # Variables to track the best clustering solution
    best_k = None
    best_labels = None
    best_sil_score = -2.0 # Silhouette goes down to -1, so -2 is a safe starting minimum
    sil_scores_for_plot = [] 
    
    # Set up the Clustering Model dynamically
    if n_clusters is not None:
        actual_n_clusters = min(n_clusters, len(sentences) - 1)
        clustering_model = AgglomerativeClustering(n_clusters=actual_n_clusters, 
                                                   metric='cosine', 
                                                   linkage='average')
        best_labels = clustering_model.fit_predict(embeddings)
        best_k = clustering_model.n_clusters_
        
        if len(set(best_labels)) > 1:
            best_sil_score = silhouette_score(embeddings, best_labels, metric='cosine')
        else:
            best_sil_score = None
            
    else:
        # Dynamically find the best k using Silhouette Score
        for k in range(min_allowed, max_allowed + 1):
            temp_model = AgglomerativeClustering(n_clusters=k, metric='cosine', linkage='average')
            temp_labels = temp_model.fit_predict(embeddings)
            
            # Calculate silhouette 
            score = silhouette_score(embeddings, temp_labels, metric='cosine')
            sil_scores_for_plot.append((k, score))
            
            if score > best_sil_score:
                best_sil_score = score
                best_k = k
                best_labels = temp_labels

    # Format the Clustering Solution (C)
    C_clusters = {}
    for idx, label in enumerate(best_labels):
        clean_label = int(label) 
        
        if clean_label not in C_clusters:
            C_clusters[clean_label] = []
        C_clusters[clean_label].append(idx)
        
    # Plot the Dendrogram and Silhouette Curve for EDA
    if show_plots and len(sentences) > 1:
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        category = d.get('category', 'unknown').upper()
        
        # Plot 1: Dendrogram
        Z = linkage(embeddings, method='average', metric='cosine')
        ax1.set_title(f"Dendrogram - {category}")
        dendrogram(Z, labels=[f"S{i}" for i in range(len(sentences))], leaf_rotation=90, ax=ax1)
        ax1.set_ylabel("Cosine Distance")
        ax1.set_xlabel("Sentence Index")
        
        # Plot 2: Silhouette Curve 
        if n_clusters is None and len(sil_scores_for_plot) > 0:
            k_vals, s_vals = zip(*sil_scores_for_plot)
            ax2.plot(k_vals, s_vals, marker='o', linestyle='-', color='steelblue', linewidth=2)
            ax2.axvline(x=best_k, color='red', linestyle='--', label=f'Best k={best_k}')
            ax2.set_title(f"Silhouette Scores by Cluster Count")
            ax2.set_xlabel("Number of Clusters (k)")
            ax2.set_ylabel("Silhouette Score")
            ax2.set_xticks(range(min_allowed, max_allowed + 1))
            ax2.grid(axis='y', linestyle='--', alpha=0.7)
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, "Silhouette search disabled\n(n_clusters was fixed)", 
                     horizontalalignment='center', verticalalignment='center')
            ax2.axis('off')
            
        plt.tight_layout()
        plt.show()

    # Final output format
    C = {"clusters": C_clusters,
         "sentences": sentences,
         "embeddings": embeddings,
         "num_clusters_found": best_k,
         "silhouette_score": best_sil_score}
    
    return C

# !! LLM-aided code block
def summarization_clustering(d, C, I=None, args=None):
    """
    Generates an extractive summary by finding the prototype sentence 
    (the one closest to the centroid) for each cluster and ordering them chronologically.
    @input d: document dictionary
    @input C: clustering solution C containing 'clusters', 'sentences', 'embeddings'
    @input I: optional index
    @input args: optional arguments
    @output: The extractive summary
    """
    if not C or not C.get("clusters") or not C.get("sentences"):
        return ""

    clusters = C["clusters"]
    sentences = C["sentences"]
    embeddings = C["embeddings"]
    
    prototype_indices = []
    
    # Iterate through each cluster 
    for cluster_id, sentence_indices in clusters.items():
        if not sentence_indices:
            continue
            
        # If the cluster only has one sentence, it wins by default
        if len(sentence_indices) == 1:
            prototype_indices.append(sentence_indices[0])
            continue
            
        # Grab the mathematical vectors for just the sentences in this specific cluster
        safe_indices = np.array(sentence_indices)
        cluster_embeddings = embeddings[safe_indices]
        
        # Calculate the mathematical center (Centroid) of this cluster
        centroid = np.mean(cluster_embeddings, axis=0).reshape(1, -1)
        
        # Measure Cosine Distances from the centroid to each sentence
        distances = cosine_distances(centroid, cluster_embeddings)[0]
        
        # Find the local index of the sentence with the smallest distance
        closest_local_idx = np.argmin(distances)
        
        # Map that local index back to the document's original sentence index
        winning_idx = sentence_indices[closest_local_idx]
        prototype_indices.append(winning_idx)
        
    # Order the chosen sentences chronologically
    prototype_indices.sort()
    
    # Extract the actual text for the prototype sentence indices
    summary_sentences = [sentences[idx] for idx in prototype_indices]
    
    # Join them together into one clean paragraph separated by spaces
    final_summary = " ".join(summary_sentences)
    
    return final_summary

# !! LLM-aided code block
def keyword_extraction(d, p=None, I=None, args=None):
    """
    Extracts the top 'p' keywords from a document by clustering candidate 
    noun phrases and selecting the prototype keyword (centroid) of each cluster.
    @input d: document dictionary 
    @input p: number of keywords to extract 
    @input I: optional index
    @input args: arguments dictionary containing the 'encoder'
    @output: A list of 'p' prototype keywords ordered by relevance to the document
    """
    if args is None:
        args = {}
        
    # Retrieve the encoder from the Index I
    encoder = args.get('encoder')
    if encoder is None and I is not None:
        encoder_name = args.get('selected_encoder', 'BAAI/bge-small-en-v1.5')
        encoder = I.get('models', {}).get(encoder_name)
    
    text = d.get('text', '')
    if not text or encoder is None:
        print("Warning: Missing text or encoder. Ensure I['models'] is populated.")
        return []

    # Determine p
    if p is None:
        p = get_optimal_p(d)
        
    # Candidate Generation (Noun Phrase Chunking)
    tokens = nltk.word_tokenize(text.lower())
    tags = nltk.pos_tag(tokens)
    
    grammar = "NP: {<JJ.*>*<NN.*>+}"
    cp = RegexpParser(grammar)
    tree = cp.parse(tags)
    
    lemmatizer = WordNetLemmatizer()
    unique_candidates = []
    seen_lemmas = set()
    
    for subtree in tree.subtrees():
        if subtree.label() == 'NP':
            phrase_words = [word for word, tag in subtree.leaves() if word.isalpha()]
            
            # Filter for quality: no empty phrases, and usually keywords are 1-2 words
            if not phrase_words or len(phrase_words) > 2:
                continue
                
            phrase_words[-1] = lemmatizer.lemmatize(phrase_words[-1])
            lemma_phrase = " ".join(phrase_words)
            
            if lemma_phrase not in seen_lemmas:
                seen_lemmas.add(lemma_phrase)
                unique_candidates.append(lemma_phrase)
    
    if not unique_candidates:
        return []
        
    # Representation (Semantic Encoding)
    # We encode the full document to find 'global' relevance later
    doc_embedding = encoder.encode([text])[0].reshape(1, -1)
    # Encode our candidate noun phrases
    word_embeddings = encoder.encode(unique_candidates)
    
    actual_p = min(p, len(unique_candidates))
    if actual_p <= 1:
        return unique_candidates[:actual_p]

    # Clustering 
    # Grouping synonyms or related concepts to avoid redundant keywords
    clustering_model = AgglomerativeClustering(n_clusters=actual_p,
                                               metric='cosine', 
                                               linkage='average')
    labels = clustering_model.fit_predict(word_embeddings)
    
    # Prototype Selection 
    prototypes = []
    prototype_embeddings = []
    
    for cluster_id in range(actual_p):
        indices = np.where(labels == cluster_id)[0]
        cluster_embs = word_embeddings[indices]
        
        # Calculate the mathematical mean of the cluster
        centroid = np.mean(cluster_embs, axis=0).reshape(1, -1)
        
        # Find the actual candidate closest to the centroid 
        dists_to_centroid = cosine_distances(centroid, cluster_embs)[0]
        closest_idx_in_cluster = np.argmin(dists_to_centroid)
        
        original_idx = indices[closest_idx_in_cluster]
        prototypes.append(unique_candidates[original_idx])
        prototype_embeddings.append(word_embeddings[original_idx])
        
    # Final Ranking (Ordering by Relevance to Document)
    prototype_embeddings = np.array(prototype_embeddings)
    dists_to_doc = cosine_distances(doc_embedding, prototype_embeddings)[0]
    
    # Argsort gives indices of sorted distances (shorter distance = more relevant)
    sorted_indices = np.argsort(dists_to_doc)
    final_keywords = [prototypes[idx] for idx in sorted_indices]
    
    return final_keywords
