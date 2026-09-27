import numpy as np
from transformers import pipeline

# !! LLM-aided code block
def keyword_extraction(d, p, I, args):
    """
    Extracts the top 'p' keywords from a given document using either 
    extractive (TF-IDF) or abstractive (LLM generation) methods.
    """

    # Determine which mode to run 
    mode = args.get("keyword_mode", "extractive")
    
    # Ensure the provided index is for document-level data
    if I.get('level') != 'document':
        print("Warning: Keyword extraction requires a document-level index!")
        return []
    
    # 1. EXTRACTIVE MODE 
    if mode == "extractive":
        # Find the row index of the current document in the TF-IDF matrix
        try:
            doc_idx = I['ids'].index(d['doc_id']) 
        except ValueError:
            print(f"Error: Document ID '{d['doc_id']}' not found in the index.")
            return []
        
        # Access the pre-trained vectorizer and pre-computed matrix from the index
        vectorizer = I['lexical']['vectorizer'] 
        tfidf_matrix = I['lexical']['matrix'] 
        
        # Retrieve the specific TF-IDF score vector for this document
        doc_vector = tfidf_matrix[doc_idx]
        
        # Get the mapping of column indices to actual text words/phrases
        feature_names = vectorizer.get_feature_names_out()
        
        # Sort the vector from highest TF-IDF score to lowest, getting the column indices
        sorted_indices = doc_vector.toarray().flatten().argsort()[::-1]
        
        # Top 'p' corresponding words or noun phrases
        top_p_keywords = [feature_names[i].replace("_", " ") for i in sorted_indices[:p]]        
        
        return top_p_keywords

    
    # 2. ABSTRACTIVE MODE 
    elif mode == "abstractive":
        # Attempt to grab the document's full text from the index 
        try:
            doc_idx = I['ids'].index(d['doc_id']) 
            doc_text = I['texts'][doc_idx] 
        except (ValueError, KeyError):
            doc_text = d.get('text', str(d)) 
            
        default_llms = ["TinyLlama/TinyLlama-1.1B-Chat-v1.0", "Qwen/Qwen2.5-0.5B-Instruct"]
        llm_names = args.get("decoder_llm_keywords", default_llms)
        
        combined_results = {}
        
        # Iterate through each specified LLM to generate keywords
        for llm_name in llm_names:
            print(f"  -> Running internal extraction with: {llm_name}")
            
            task_type = "text-generation"
            
            # Base generation parameters (max length and omitting the prompt in the output)
            pipeline_kwargs = {"max_new_tokens": 50, "return_full_text": False}
            
            # --- MODEL-SPECIFIC PROMPT ENGINEERING ---
            
            # A. Qwen Model Formatting (Requires strict ChatML tags and deterministic generation)
            if "qwen" in llm_name.lower():
                prompt = (
                    f"<|im_start|>system\nYou are an expert data extractor. You always return exactly the requested number of keywords, formatted strictly as a comma-separated list with no other text.<|im_end|>\n"
                    f"<|im_start|>user\nExtract exactly {p} distinct and important keywords from the text below. "
                    f"You MUST return exactly {p} keywords separated by commas. Do not stop at just one word.\n\n"
                    f"Text: {doc_text[:1200]}<|im_end|>\n"
                    f"<|im_start|>assistant\n"
                )
                pipeline_kwargs["do_sample"] = False 
                pipeline_kwargs["max_new_tokens"] = 75 
                
            # B. TinyLlama / Fallback Formatting (Standard text completion)
            else:
                prompt = f"Extract exactly {p} core keywords from the text below. Return ONLY a comma-separated list of words.\n\nText: {doc_text[:1200]}\n\nKeywords:"
                pipeline_kwargs["num_return_sequences"] = 1
                
            # --- LLM INFERENCE ---
            try:
                # Initialize the HuggingFace pipeline 
                generator = pipeline(task_type, model=llm_name, device_map="auto")
                
                # Generate the text
                output = generator(prompt, **pipeline_kwargs)
                
                # Extract the generated string from the pipeline output format
                generated_text = output[0]['generated_text'] if isinstance(output, list) else output
                
                # --- POST-PROCESSING ---
                # Split by comma, strip whitespace, make lowercase, and remove rogue periods
                keywords = [kw.strip().lower().replace('.', '') for kw in generated_text.split(',')]
                
                
                final_keywords = [k for k in keywords if k][:p]
                
                combined_results[llm_name] = final_keywords
                
            except Exception as e:
                print(f"Error generating keywords for {llm_name}: {e}")
                combined_results[llm_name] = []
                
        return combined_results
    