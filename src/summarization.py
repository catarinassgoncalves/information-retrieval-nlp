import nltk
from transformers import pipeline
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import random
from sentence_transformers import SentenceTransformer
from indexing_module import process_text

# !! LLM-aided code block
def compute_lexical_scores(d, I, args):
    """
    Computes the Okapi BM25 scores for each sentence against the entire document.
    """
    # BM25 Parameters
    k1 = args.get("bm25_k1", 1.2)
    b = args.get("bm25_b", 0.75)
    avgdl = I['avg_length']
    
    # Extract tools from our index
    vectorizer = I['lexical']['vectorizer']
    doc_id = d['doc_id']
    
    # Indices that correspond to this specific document's sentences
    sent_indices = [i for i, sid in enumerate(I['ids']) if sid.startswith(f"{doc_id}_sent_")]
    
    # Prepare the "Query" (The whole document text)
    query_terms = process_text(d['text'])
    unique_query_terms = set(query_terms) # Remove duplicates to speed up the math
    
    scores = {}
    # Score every sentence
    for idx in sent_indices:
        sent_id = I['ids'][idx]
        sent_text = I['texts'][idx]
        sent_len = I['lengths'][idx]
        
        # Tokenize the sentence to get the raw terms
        sent_terms = process_text(sent_text)
        score = 0.0
        
        for term in unique_query_terms:
            # If the query term isn't in the sentence, BM25 adds 0, so skip it
            if term not in sent_terms:
                continue
                
            # Calculate Term Frequency (TF) in the sentence
            tf = sent_terms.count(term)
            
            # Use the pre-calculated IDF directly from Scikit-Learn
            if term in vectorizer.vocabulary_:
                term_idx = vectorizer.vocabulary_[term]
                idf = vectorizer.idf_[term_idx]
            else:
                idf = 0.0 
            
            # Okapi BM25 Formula
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * (sent_len / avgdl))
            score += idf * (numerator / denominator)
            
        scores[sent_id] = score
        
    return scores

# !! LLM-aided code block
def compute_semantic_scores(d, I, args):
    """
    Computes cosine similarity between sentence embeddings and the document embedding.
    """
    # Identify the model we want to use
    encoder_name = args.get('selected_encoder', 'BAAI/bge-small-en-v1.5')
    doc_id = d['doc_id']
    
    # Grab the giant matrix of all embeddings for this model
    all_embeddings = I['semantic'][encoder_name]
    
    # Find the row indices for the sentences belonging to this specific document
    sent_indices = [i for i, sid in enumerate(I['ids']) if sid.startswith(f"{doc_id}_sent_")]
    if not sent_indices:
        return {}
        
    # Extract just the vectors for this document's sentences
    sentence_vectors = all_embeddings[sent_indices]
    
    # Compute the Document Vector (Average of its sentences)
    doc_vector = np.mean(sentence_vectors, axis=0).reshape(1, -1)
    
    # Compute Cosine Similarity for all sentences
    sim_scores = cosine_similarity(sentence_vectors, doc_vector)
    
    # Sscores
    scores = {}
    for local_idx, global_idx in enumerate(sent_indices):
        sent_id = I['ids'][global_idx]
        # sim_scores[local_idx][0] grabs the specific float value
        scores[sent_id] = float(sim_scores[local_idx][0])
        
    return scores

def summarization_baseline(d, p):
    """
    Extracts the first 'p' sentences of document 'd'.
    Returns a list of tuples (position, score) to match other summarizers.
    """
    # Split text into sentences using NLTK
    sentences = nltk.sent_tokenize(d['text'])
    
    # Figure out how many sentences to grab (protects against super short articles)
    num_sentences = min(p, len(sentences))

    return [(i, 1.0) for i in range(num_sentences)]

# !! LLM-aided code block
def extractive_summarization(d, p, I, args):
    """
    Scores sentences using the requested IR model and extracts the top 'p'.
    Returns a list of tuples: (sentence_position, score).
    """
    # Route to the correct scoring engine: 'lexical' or 'semantic'
    model_family = args.get('model_family', 'lexical') 
    
    if model_family == 'lexical':
        # BM25
        scores = compute_lexical_scores(d, I, args)
        
    elif model_family == 'semantic':
        # Cosine Similarity
        scores = compute_semantic_scores(d, I, args)
        
    else:
        return [] 
    
    if not scores:
        return []

    # Rank sentences by their scores 
    ranked_sentences = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    # Extract the top 'p' highest-scoring sentences
    top_p_sentences = ranked_sentences[:p]
    
    # Chronological Reordering
    def get_chronological_index(item):
        sent_id = item[0]
        return int(sent_id.split('_sent_')[-1])
        
    top_p_chronological = sorted(top_p_sentences, key=get_chronological_index)

    # Format output precisely as requested: list of (position, score)
    result = []
    for sent_id, score in top_p_chronological:
        # sent_id looks like 'bbc_business_001_sent_5' we just want '5'
        pos = int(sent_id.split('_sent_')[-1])
        
        # Append the tuple (position, score)
        result.append((pos, score))
        
    return result

# !! LLM-aided code block
def mmr_summarization(d, p, I, args):
    """
    Pulls the scores from the existing engines but applying the MMR loop.
    Returns a list of tuples: (sentence_position, mmr_score).
    """
    model_family = args.get('model_family', 'lexical')
    if model_family == 'lexical':
        relevance_scores = compute_lexical_scores(d, I, args)
    elif model_family == 'semantic':
        relevance_scores = compute_semantic_scores(d, I, args)
    else:
        return [] 
    
    if not relevance_scores:
        return []

    # MMR Setup
    lambda_param = args.get('mmr_lambda', 0.5)
    encoder_name = args.get('selected_encoder', 'BAAI/bge-small-en-v1.5')
    all_embeddings = I['semantic'][encoder_name]

    # Normalize relevance scores to a [0.0, 1.0] scale 
    max_score = max(relevance_scores.values()) if relevance_scores else 1.0
    if max_score == 0: max_score = 1.0
    normalized_scores = {k: (v / max_score) for k, v in relevance_scores.items()}

    unselected_sentences = list(relevance_scores.keys())
    selected_sentences = []

    # MMR Selection Loop
    while len(selected_sentences) < p and unselected_sentences:
        if not selected_sentences:
            # First sentence: purely based on highest relevance
            best_sent = max(unselected_sentences, key=lambda x: normalized_scores[x])
            best_score = (1 - lambda_param) * normalized_scores[best_sent]
            selected_sentences.append((best_sent, best_score))
            unselected_sentences.remove(best_sent)
        else:
            best_mmr_score = -float('inf')
            best_sent = None

            # Get the semantic vectors of sentences already in our summary
            selected_ids = [item[0] for item in selected_sentences]
            selected_indices = [I['ids'].index(s) for s in selected_ids]
            selected_vectors = all_embeddings[selected_indices]

            for sent_id in unselected_sentences:
                # Get the vector for the candidate sentence
                sent_idx = I['ids'].index(sent_id)
                sent_vector = all_embeddings[sent_idx].reshape(1, -1)

                # Calculate redundancy (Cosine Similarity to already selected sentences)
                sims = cosine_similarity(sent_vector, selected_vectors)
                sum_sim = np.sum(sims)

                # The core MMR mathematical trade-off
                mmr_score = ((1 - lambda_param) * normalized_scores[sent_id]) - (lambda_param * sum_sim)
                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_sent = sent_id

            selected_sentences.append((best_sent, best_mmr_score))
            unselected_sentences.remove(best_sent)

    # Chronological Reordering 
    def get_chronological_index(item):
        sent_id = item[0]
        return int(sent_id.split('_sent_')[-1])
        
    top_p_chronological = sorted(selected_sentences, key=get_chronological_index)
    
    result = []
    for sent_id, score in top_p_chronological:
        pos = int(sent_id.split('_sent_')[-1])
        result.append((pos, score))
        
    return result

# Function that plots the lambda varying in a random document per topic
# !! LLM-aided code block
plot_encoder = SentenceTransformer('BAAI/bge-small-en-v1.5')
def varying_lambda_plot(D, I, mode='single', sample_size=10):
    """
    Plots the impact of varying lambda using Semantic MMR and Semantic Redundancy.
    """
    plt.figure(figsize=(10, 6))
    lambda_values = np.arange(0.0, 1.1, 0.1)
    
    # Picking one random document to analyse the impact of lambda on redundancy
    if mode == 'single':
        doc = random.choice(D)
        all_sentences = nltk.sent_tokenize(doc['text'])
        redundancy_scores = []
        
        for lam in lambda_values:
            # Semantic arguments
            mmr_args = {'summary_mode': 'mmr', 'model_family': 'semantic', 'selected_encoder': 'BAAI/bge-small-en-v1.5', 'mmr_lambda': float(lam)}                    
            raw_output = summarization(doc, p=5, l=float(lam), I=I, args=mmr_args)
            selected_texts = [all_sentences[pos] for pos, score in raw_output]
            
            if len(selected_texts) > 1:
                # Calculate Semantic Redundancy
                embeddings = plot_encoder.encode(selected_texts)
                sim_matrix = cosine_similarity(embeddings)
                upper_tri = sim_matrix[np.triu_indices(len(selected_texts), k=1)]
                redundancy_scores.append(np.mean(upper_tri))
            else:
                redundancy_scores.append(0.0)
                
        plt.plot(lambda_values, redundancy_scores, marker='o', linewidth=2.5, markersize=8, color='darkorange', label=f"Doc: {doc['doc_id']}")
        plt.title('Impact of Varying Lambda on Semantic Redundancy', fontsize=14, fontweight='bold')

    # Picking a sample of documents per category to analyse the impact of lambda on redundancy across topics
    elif mode == 'category':
        docs_by_topic = {}
        for doc in D:
            topic = doc.get('category', 'unknown')
            if topic not in docs_by_topic:
                docs_by_topic[topic] = []
            docs_by_topic[topic].append(doc)
            
        for category, docs in docs_by_topic.items():
            sampled_docs = random.sample(docs, min(sample_size, len(docs)))
            category_redundancy = []
            
            for lam in lambda_values:
                lam_redundancy_sum = 0.0
                for doc in sampled_docs:
                    all_sentences = nltk.sent_tokenize(doc['text'])
                    
                    # Semantic arguments
                    mmr_args = {'summary_mode': 'mmr', 'model_family': 'semantic', 'selected_encoder': 'BAAI/bge-small-en-v1.5', 'mmr_lambda': float(lam)}                    
                    raw_output = summarization(doc, p=5, l=float(lam), I=I, args=mmr_args)
                    selected_texts = [all_sentences[pos] for pos, score in raw_output]
                    
                    if len(selected_texts) > 1:
                        # Calculate Semantic Redundancy
                        embeddings = plot_encoder.encode(selected_texts)
                        sim_matrix = cosine_similarity(embeddings)
                        upper_tri = sim_matrix[np.triu_indices(len(selected_texts), k=1)]
                        lam_redundancy_sum += np.mean(upper_tri)
                        
                category_redundancy.append(lam_redundancy_sum / len(sampled_docs))
                
            plt.plot(lambda_values, category_redundancy, marker='o', linewidth=2, markersize=6, label=f"Category: {category}")
        plt.title(f'Average Semantic Redundancy per Category (n={sample_size})', fontsize=14, fontweight='bold')

    plt.xlabel('Lambda Value', fontsize=12)
    plt.ylabel('Redundancy (Semantic Cosine Similarity)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(lambda_values)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

# !! LLM-aided code block
def abstractive_summarization(d, args):
    """
    Generates an abstractive summary of the document using a pre-trained LLM.
    Expected args:
      - 'summary_generator': A pre-loaded Hugging Face pipeline (highly recommended to avoid reloading).
      - 'decoder_llm_summary': Name of the model (default: TinyLlama).
      - 'prompt_type': The style of prompting ('prompt_1' for chat, 'prompt_2' for TL;DR).
      - 'max_summary_length': Max new tokens to generate.    
    Returns: A single string containing the generated summary text.
    """
    doc_text = d['text']
    short_text = doc_text[:2000] 
    
    llm_name = args.get("decoder_llm_summary", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    prompt_type = args.get("prompt_type", "prompt_1")
    # We could have used 'l' here instead of adding a new 'max_summary_length' argument
    max_length = args.get("max_summary_length", 100)
    
    # Load the generator FIRST so we can access its tokenizer for templates
    generator = args.get("summary_generator")
    if generator is None:
        print(f"Loading {llm_name} on the fly... (This might take a moment)")
        generator = pipeline("text-generation", model=llm_name)
    
    # Prompt Engineering using Lab 3's apply_chat_template
    if prompt_type == "prompt_1" and "Chat" in llm_name:
        # Profile 1: Chat Models (Using proper dictionary messages)
        messages = [
            {"role": "system", "content": "You are a professional journalist. Write a concise summary of the provided article."},
            {"role": "user", "content": f"Article: {short_text}"}
        ]
        prompt = generator.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
    elif prompt_type == "prompt_2":
        # Profile 2: The classic "TL;DR" continuation
        prompt = f"{short_text}\n\nTL;DR:"
        
    else:
        prompt = f"Summarize this: {short_text}\nSummary:"

    # Text Generation
    try:
        output = generator(prompt, 
                           max_new_tokens=max_length, 
                           num_return_sequences=1, 
                           return_full_text=False, 
                           temperature=0.2,
                           do_sample=True)
        
        generated_summary = output[0]['generated_text'].strip()
        return generated_summary
        
    except Exception as e:
        print(f"Error generating summary: {e}")
        return ""

# !! LLM-aided code block
def rrf_summarization(d, p, I, args):
    """
    Combines lexical (BM25) and semantic (LLM) rankings using Reciprocal Rank Fusion (RRF).
    Returns a list of tuples: [(position, rrf_score), ...]
    """

    # Get scores from lexical model
    lexical_args = args.copy()
    lexical_args['model_family'] = 'lexical'
    lexical_scores = compute_lexical_scores(d, I, lexical_args)

    # Get scores from semantic model
    semantic_args = args.copy()
    semantic_args['model_family'] = 'semantic'
    semantic_scores = compute_semantic_scores(d, I, semantic_args)
    
    # If something fails, return an empty list (standardized format)
    if not lexical_scores or not semantic_scores:
        return []
        
    # Convert scores into Ranks 
    lexical_ranked = sorted(lexical_scores.keys(), key=lambda x: lexical_scores[x], reverse=True)
    semantic_ranked = sorted(semantic_scores.keys(), key=lambda x: semantic_scores[x], reverse=True)
    
    lexical_ranks = {sent_id: rank + 1 for rank, sent_id in enumerate(lexical_ranked)}
    semantic_ranks = {sent_id: rank + 1 for rank, sent_id in enumerate(semantic_ranked)}
    
    # Calculate RRF Scores 
    mu = args.get('rrf_mu', 5.0) 
    rrf_scores = {}
    
    # Iterate through all unique sentences in the document
    all_sentences = set(lexical_ranks.keys()).union(set(semantic_ranks.keys()))

    for sent_id in all_sentences:
        rrf_score = 0.0
        
        # Add lexical RRF contribution
        if sent_id in lexical_ranks:
            rrf_score += 1.0 / (mu + lexical_ranks[sent_id])
            
        # Add semantic RRF contribution
        if sent_id in semantic_ranks:
            rrf_score += 1.0 / (mu + semantic_ranks[sent_id])
            
        rrf_scores[sent_id] = rrf_score
        
    # Select top 'p' sentences based on their combined RRF score
    top_rrf_sentences = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:p]
    
    # Chronological Reordering 
    def get_chronological_index(sent_id):
        return int(sent_id.split('_sent_')[-1])
        
    top_p_chronological = sorted(top_rrf_sentences, key=get_chronological_index)
    
    output = []
    for sent_id in top_p_chronological:
        pos = get_chronological_index(sent_id)
        score = rrf_scores[sent_id]
        output.append((pos, score))
        
    return output

def summarization(d, p, l, I, args):
    """
    Routes to the correct type of summarization
    """
    # Determine the summarization mode (default to standard extractive)
    summary_mode = args.get('summary_mode', 'extractive')
    
    if summary_mode == 'baseline':
        return summarization_baseline(d, p)
        
    elif summary_mode == 'extractive':
        return extractive_summarization(d, p, I, args)
        
    elif summary_mode == 'mmr':
        return mmr_summarization(d, p, I, args)
        
    elif summary_mode == 'rrf':
        return rrf_summarization(d, p, I, args)
        
    elif summary_mode == 'abstractive':
        return abstractive_summarization(d, args)
        
    else:
        return f"Error: Unknown summary_mode '{summary_mode}'"