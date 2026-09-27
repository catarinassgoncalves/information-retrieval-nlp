from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def evaluate_keyword_agreement(tfidf_keywords, llm_keywords):
    """
    Compares two lists of keywords and returns both Lexical and Semantic agreement scores.
    """
    # Clean the keywords: lowercase, strip whitespace
    list1 = [str(k).strip().lower() for k in tfidf_keywords]
    list2 = [str(k).strip().lower() for k in llm_keywords]


    # 1. JACCARD SIMILARITY (Lexical Agreement)
    set1, set2 = set(list1), set(list2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    jaccard_score = intersection / union if union > 0 else 0.0

    # 2. COSINE SIMILARITY (Semantic Agreement)
    theme1 = " ".join(list1)
    theme2 = " ".join(list2)
    
    embeddings = embedding_model.encode([theme1, theme2])
    
    # Calculate the angle between the two vectors
    cos_sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    
    cos_sim = max(0.0, min(1.0, float(cos_sim)))

    return round(jaccard_score, 4), round(cos_sim, 4)



