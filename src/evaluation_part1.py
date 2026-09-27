import nltk
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from rouge_score import rouge_scorer

# !! LLM-aided code block
def build_extractive_r_set(D_articles, R_summaries, p_sentences=3, encoder_name="all-MiniLM-L6-v2"):
    """
    Translates BBC text summaries into the 'R_set' of sentence IDs for evaluation.
    """
    print("Building R_set (Ground Truth IDs) using Semantic Matching...")
    encoder = SentenceTransformer(encoder_name)
    R_set_extractive = {}
    
    # Map doc_ids to their original article text for quick lookup
    articles_dict = {item['doc_id']: item['text'] for item in D_articles}
    
    for item in R_summaries:
        doc_id = item['doc_id']
        bbc_summary_text = item['text']
        original_article_text = articles_dict.get(doc_id, "")
        
        if not original_article_text:
            continue 
            
        # Split original article into sentences
        article_sentences = nltk.sent_tokenize(original_article_text)
        
        if len(article_sentences) <= p_sentences:
            R_set_extractive[doc_id] = [f"{doc_id}_sent_{i}" for i in range(len(article_sentences))]
            continue
            
        # Semantic comparison: Which original sentences match the summary text?
        sentence_embeddings = encoder.encode(article_sentences)
        summary_embedding = encoder.encode([bbc_summary_text])
        similarities = cosine_similarity(sentence_embeddings, summary_embedding).flatten()
        
        # Take the top p matches and keep them in chronological order
        top_indices = sorted(similarities.argsort()[-p_sentences:])
        
        # IMPROVEMENT: Constructing the FULL, robust ID to prevent cross-document collisions
        R_set_extractive[doc_id] = [f"{doc_id}_sent_{i}" for i in top_indices]
        
    print("R_set successfully built!")
    return R_set_extractive


# !! LLM-aided code block
# Computes the F_{\beta}-measure
def compute_f_beta(precision, recall, beta=1.0):
    """
    !! Computes the F-beta measure.
    beta > 1 favors recall. beta < 1 favors precision. beta=1 is standard F1.
    """
    if precision == 0 and recall == 0:
        return 0.0

    numerator = (1 + beta**2) * precision * recall
    denominator = (beta**2 * precision) + recall
    return numerator / denominator


# 11-Point Interpolated precisions and the respective PR-Curve
def compute_11_point_interpolated_pr(recalls, precisions):

    standard_recalls = np.linspace(0.0, 1.0, 11) # Creates an array of 11 perfectly spaced numbers between 0.0 and 1.0
    interpolated_precisions = [] # List to store the 11 precision scores
    
    for std_r in standard_recalls:
        valid_precisions = [p for r, p in zip(recalls, precisions) if r >= std_r]
        if valid_precisions:
            interpolated_precisions.append(max(valid_precisions))
        else:
            interpolated_precisions.append(0.0)
            
    return interpolated_precisions


def plot_11_point_pr_curve(interpolated_precisions):
    
    standard_recalls = np.linspace(0.0, 1.0, 11)
    plt.figure(figsize=(8, 6))
    
    plt.plot(standard_recalls, interpolated_precisions, marker='o', linestyle='-', color='#1f77b4', linewidth=2)
    
    plt.title("11-Point Interpolated Precision-Recall Curve", fontsize=14, fontweight='bold')
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Interpolated Precision", fontsize=12)
    
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xticks(standard_recalls)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

# !! LLM-aided code block
def evaluate_extractive_document(S_ranked, R_set, doc_id, category=None): 
    
    # Category-specific beta mapping
    beta_weights = {
        "business": 0.5,      # Favor Precision
        "politics": 0.5,      
        "tech": 0.5,          
        "entertainment": 2.0, # Favor Recall
        "sport": 2.0          
    }
    
    beta = beta_weights.get(category.lower(), 1.0) if category else 1.0

    if S_ranked and isinstance(S_ranked[0], tuple):
        S_ranked_sorted = sorted(S_ranked, key=lambda x: x[1], reverse=True)
        # IMPROVEMENT: Reconstructing the full ID to match the newly robust R_set
        S_ids = [f"{doc_id}_sent_{pos}" for pos, score in S_ranked_sorted]
    else:
        # IMPROVEMENT: Handles both raw ints and pre-formatted strings securely
        S_ids = [f"{doc_id}_sent_{item}" if isinstance(item, int) else item for item in S_ranked]

    S_set_flat = set(S_ids)
    R_set_flat = set(R_set)

    # Precision, Recall, F_{beta}-measure
    intersection = S_set_flat.intersection(R_set_flat)
    precision = len(intersection) / len(S_set_flat) if S_set_flat else 0.0
    recall = len(intersection) / len(R_set_flat) if R_set_flat else 0.0
    f_beta = compute_f_beta(precision, recall, beta)

    # PR-curves, Average uninterpolated Precision
    precisions = []
    recalls = []
    hits = 0
    sum_precisions = 0.0 

    for k, item_id in enumerate(S_ids, start=1):
        if item_id in R_set_flat:
            hits += 1
            p_at_k = hits / k
            sum_precisions += p_at_k 
            
        p_at_k_general = hits / k
        r_at_k_general = hits / len(R_set_flat) if len(R_set_flat) > 0 else 0.0
        
        precisions.append(p_at_k_general)
        recalls.append(r_at_k_general)

    ap = sum_precisions / len(R_set_flat) if len(R_set_flat) > 0 else 0.0
    interpolated_11_pt_p = compute_11_point_interpolated_pr(recalls, precisions)
    
    return {
        "precision": precision,
        "recall": recall,
        "f_beta": f_beta,
        "applied_beta": beta,  
        "ap": ap,
        "pr_curve": (recalls, precisions), 
        "interpolated_11_pt_p": interpolated_11_pt_p
    }

def evaluate_abstractive_document(S_text, R_text, encoder):
    vec_S = encoder.encode(S_text[:2000]).reshape(1, -1)
    vec_R = encoder.encode(R_text[:2000]).reshape(1, -1)
    
    similarity = cosine_similarity(vec_S, vec_R)[0][0]
    
    # IMPROVEMENT: Added ROUGE scoring for a complete abstractive evaluation profile
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(R_text, S_text)
    
    return {"semantic_similarity": float(similarity),
            "rouge1": rouge_scores['rouge1'].fmeasure,
            "rouge2": rouge_scores['rouge2'].fmeasure,
            "rougeL": rouge_scores['rougeL'].fmeasure}

# !! LLM-aided code block
def evaluation(S_set, R_set, args):
    
    mode = args.get("eval_mode", "extractive")
    doc_metrics_dict = {} 

    if mode == "extractive":
        doc_categories = args.get("doc_categories", {}) 
        
        for doc_id in S_set.keys():
            S_ranked = S_set.get(doc_id, [])
            
            if isinstance(S_ranked, str):
                print(f"Warning: Document {doc_id} returned a stitched string, but eval_mode is 'extractive'. Skipping.")
                continue

            R_truth = R_set.get(doc_id, [])
            doc_category = doc_categories.get(doc_id, "unknown")
            
            doc_metrics_dict[doc_id] = evaluate_extractive_document(S_ranked, R_truth, doc_id=doc_id, category=doc_category)

        # Macro Averages
        num_docs = len(doc_metrics_dict)
        if num_docs > 0:
            all_11_point_curves = [d["interpolated_11_pt_p"] for d in doc_metrics_dict.values()]
            macro_11_pt_curve = np.mean(all_11_point_curves, axis=0).tolist()

            macro_metrics = {
                "Macro_Precision": sum(d["precision"] for d in doc_metrics_dict.values()) / num_docs,
                "Macro_Recall": sum(d["recall"] for d in doc_metrics_dict.values()) / num_docs,
                "Macro_F_beta": sum(d["f_beta"] for d in doc_metrics_dict.values()) / num_docs, 
                "Macro_AP": sum(d["ap"] for d in doc_metrics_dict.values()) / num_docs,
                "Macro_11_Point_PR": macro_11_pt_curve,
                "Standard_11_Recalls": np.linspace(0.0, 1.0, 11).tolist()
            }
        else:
            macro_metrics = {}
        
    elif mode == "abstractive":
        eval_encoder_name = args.get("eval_encoder", "all-mpnet-base-v2")
        print(f"Loading independent evaluator LLM: {eval_encoder_name}...")
        eval_encoder = SentenceTransformer(eval_encoder_name)

        for doc_id in S_set.keys():
            S_text = S_set.get(doc_id, "")
            
            if not isinstance(S_text, str):
                print(f"Warning: Document {doc_id} returned a list, but eval_mode is 'abstractive'. Ensure it is string text. Skipping.")
                continue

            R_truth = R_set.get(doc_id, "")
            if not R_truth:
                continue
            
            metrics = evaluate_abstractive_document(S_text, R_truth, eval_encoder)
            doc_metrics_dict[doc_id] = metrics

        num_docs = len(doc_metrics_dict)
        if num_docs > 0:
            macro_metrics = {"Average_Semantic_Similarity": sum(d["semantic_similarity"] for d in doc_metrics_dict.values()) / num_docs,
                             "Average_ROUGE-1": sum(d["rouge1"] for d in doc_metrics_dict.values()) / num_docs,
                             "Average_ROUGE-2": sum(d["rouge2"] for d in doc_metrics_dict.values()) / num_docs,
                             "Average_ROUGE-L": sum(d["rougeL"] for d in doc_metrics_dict.values()) / num_docs,}
        else:
            macro_metrics = {}

    else:
        raise ValueError("Invalid eval_mode. Choose 'extractive' or 'abstractive'.")

    results = {"macro_averages": macro_metrics,
               "document_scores": doc_metrics_dict}

    if mode == "extractive" and "Macro_11_Point_PR" in macro_metrics:
        plot_11_point_pr_curve(macro_metrics["Macro_11_Point_PR"])

    return results
