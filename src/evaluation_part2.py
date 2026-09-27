from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from rouge_score import rouge_scorer
import re
import nltk

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


def evaluate_extractive_document_from_text(S_text, R_text, category=None): 
    """
    Evaluates extractive summaries by tokenizing back into sentences for exact-match 
    Precision/Recall (Project I), AND computes ROUGE scores (Project II expansion).
    """
    # Category-specific beta mapping
    beta_weights = {"business": 0.5,      # Favor Precision
                    "politics": 0.5,      
                    "tech": 0.5,          
                    "entertainment": 2.0, # Favor Recall
                    "sport": 2.0}    
    beta = beta_weights.get(category.lower(), 1.0) if category else 1.0



    s_sentences = [s.strip() for s in nltk.sent_tokenize(S_text) if s.strip()]

    r_sentences_raw = re.split(r'(?<=[.?!])\s*(?=[A-Z])', R_text.replace('\n', ' '))
    r_sentences = [r.strip() for r in r_sentences_raw if r.strip()]

    
# 1. Precision Hits: Does the generated sentence exist in the truth text?
    hits = 0
    for s in s_sentences:
        if s in R_text: 
            hits += 1


    # 3. Calculate mathematically bounded Precision and Recall
    precision = hits / len(s_sentences) if len(s_sentences) > 0 else 0.0
    recall = hits / len(r_sentences) if len(r_sentences) > 0 else 0.0
    
    f_beta = compute_f_beta(precision, recall, beta)

    # ROUGE Expansion
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(R_text, S_text)

    return { "precision": precision, 
            "recall": recall, 
            "f_beta": f_beta,
            "rouge1": rouge_scores['rouge1'].fmeasure,
            "rouge2": rouge_scores['rouge2'].fmeasure,
            "rougeL": rouge_scores['rougeL'].fmeasure}

def evaluate_abstractive_document(S_text, R_text, encoder):
    vec_S = encoder.encode(S_text[:2000]).reshape(1, -1)
    vec_R = encoder.encode(R_text[:2000]).reshape(1, -1)
    
    similarity = cosine_similarity(vec_S, vec_R)[0][0]
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(R_text, S_text)
    
    return {"semantic_similarity": float(similarity),
            "rouge1": rouge_scores['rouge1'].fmeasure,
            "rouge2": rouge_scores['rouge2'].fmeasure,
            "rougeL": rouge_scores['rougeL'].fmeasure}


# !! LLM-aided code block
def evaluation(S_set, R_set, args):
    """
    Main evaluation function. 
    Expects S_set and R_set to contain plain text strings for BOTH extractive and abstractive modes.
    """
    mode = args.get("eval_mode", "extractive")
    doc_metrics_dict = {} 

    if mode == "extractive":
        doc_categories = args.get("doc_categories", {}) 
        
        for doc_id in S_set.keys():
            S_text = S_set.get(doc_id, "")
            R_truth_text = R_set.get(doc_id, "")
            
            if not isinstance(S_text, str):
                print(f"Warning: Document {doc_id} returned a list, but eval_mode expects text. Skipping.")
                continue

            if not R_truth_text:
                continue

            doc_category = doc_categories.get(doc_id, "unknown")
            
            # Pass the text to our new evaluator
            metrics = evaluate_extractive_document_from_text(S_text, R_truth_text, category=doc_category)
            doc_metrics_dict[doc_id] = metrics

        # Macro Averages
        num_docs = len(doc_metrics_dict)
        if num_docs > 0:
            macro_metrics = {"Macro_Precision": sum(d["precision"] for d in doc_metrics_dict.values()) / num_docs,
                             "Macro_Recall": sum(d["recall"] for d in doc_metrics_dict.values()) / num_docs,
                             "Macro_F_beta": sum(d["f_beta"] for d in doc_metrics_dict.values()) / num_docs,
                             "Average_ROUGE-1": sum(d["rouge1"] for d in doc_metrics_dict.values()) / num_docs,
                             "Average_ROUGE-2": sum(d["rouge2"] for d in doc_metrics_dict.values()) / num_docs,
                             "Average_ROUGE-L": sum(d["rougeL"] for d in doc_metrics_dict.values()) / num_docs}
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

    return results