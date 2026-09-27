# !! LLM-aided code block
def get_optimal_p(d):
    """
    Analyzes a document's length and category to determine 
    the optimal number of keywords (p) to extract.
    """
    text = d.get('text', '')
    category = d.get('category', '').lower()
    
    # 1. Count the rough number of words
    word_count = len(text.split())
    
    # 2. Set a baseline 'p' based on how long the document is
    if word_count < 100:
        base_p = 3       
    elif word_count < 300:
        base_p = 4       
    elif word_count < 600:
        base_p = 6       
    else:
        base_p = 8       
        
    # 3. Adjust based on the category
    dense_categories = [ 'politics', 'tech']
    simple_categories = ['sport', 'entertainment', 'business']
    
    if category in dense_categories:
        base_p += 1  
    elif category in simple_categories:
        base_p = base_p  
        
    final_p = max(2, min(base_p, 10))
    
    return final_p
