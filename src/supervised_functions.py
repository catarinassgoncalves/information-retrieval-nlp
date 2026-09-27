import transformers
from datasets import Dataset
import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import evaluate
import numpy as np
rouge_metric = evaluate.load("rouge")
import re
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from tqdm.auto import tqdm
from sklearn.ensemble import RandomForestClassifier
from rouge_score import rouge_scorer
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score, confusion_matrix

# This function is based on Lab 4
def fine_tuning(M, Dtrain, Rtrain, args):
    """
    Fine-tunes a pretrained Seq2Seq LLM for summarization.
    """
    # Initialize Tokenizer and Model
    tokenizer = transformers.AutoTokenizer.from_pretrained(M) 
    model = transformers.AutoModelForSeq2SeqLM.from_pretrained(M)
    
    d_texts = [doc['text'] for doc in Dtrain]
    r_texts = [ref['text'] for ref in Rtrain]

    # Create the raw Hugging Face Dataset
    raw_dataset = Dataset.from_dict({"document": d_texts, "summary": r_texts})
    
    # LLM-aided code block
    # Tokenization mapping function (Hugging Face)
    def tokenize_function(examples):
        # Tokenize the input documents
        model_inputs = tokenizer(examples["document"], 
                                 max_length = 512, 
                                 truncation = True, 
                                 padding = "max_length")
        
        # Tokenize the target summaries
        labels = tokenizer(text_target = examples["summary"], 
                           max_length = 150, 
                           truncation = True, 
                           padding = "max_length")
        
        # Assign the tokenized summaries as the "labels" for the model to learn from
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    # Tokenize the data and training-validation split
    tokenized_data = raw_dataset.map(tokenize_function, batched=True)
    split = tokenized_data.train_test_split(test_size=0.1, seed=42)
    # tokenized_train, tokenized_validation = split['train'], split['test']

    # In case of OOM
    tokenized_train = split['train']
    val_size = min(100, len(split['test']))
    tokenized_validation = split['test'].select(range(val_size))

    # Metrics function for ROUGE (question 1)
    # !! LLM-aided code block
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        # Decode generated tokens to text
        # Replace -100 in labels (used to ignore padding) before decoding
        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # ROUGE expects newlines after each sentence
        decoded_preds = ["\n".join(pred.strip().split()) for pred in decoded_preds]
        decoded_labels = ["\n".join(label.strip().split()) for label in decoded_labels]
        
        # Compute ROUGE
        result = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        return {k: round(v * 100, 4) for k, v in result.items()}
    
    # Training Arguments
    training_args = transformers.Seq2SeqTrainingArguments(output_dir = "results",
                                                          learning_rate=args.get("learning_rate", 2e-5),
                                                          per_device_train_batch_size=args.get("batch_size", 4),
                                                          per_device_eval_batch_size=args.get("batch_size", 4),
                                                          num_train_epochs=args.get("epochs", 5),
                                                          weight_decay = 0.01,
                                                          eval_strategy = "epoch", # Evaluate at the end of each epoch
                                                          save_strategy = "epoch", # Save at the end of each epoch
                                                          predict_with_generate = True, # Generate text so ROUGE can be computed
                                                          load_best_model_at_end = True) # Keeps the model with the best validation score

    # Trainer
    trainer = transformers.Seq2SeqTrainer(model = model,
                                          args = training_args,
                                          train_dataset = tokenized_train,
                                          eval_dataset = tokenized_validation,
                                          processing_class = tokenizer, 
                                          data_collator = transformers.DataCollatorForSeq2Seq(tokenizer = tokenizer, model = model),
                                          compute_metrics=compute_metrics)

    # Execute Fine-tuning
    print("Starting fine-tuning...")
    trainer.train() 
    
    # Save the Model
    save_path = args.get("save_path", f"fine_tuned_{M.replace('/', '_')}")
    trainer.save_model(save_path) 
    print(f"Model saved to {save_path}")

    # Save the ROUGE scores for question 1
    history_path = os.path.join(save_path, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(trainer.state.log_history, f)
    
    return model

# Helper function to generate summaries for a given model
# !! LLM-aided code block
def fine_tuning_summaries(model_path, docs):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    
    # Use GPU if available, otherwise CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    summaries = []
    # tqdm adds a nice progress bar so you know it hasn't frozen!
    for doc in tqdm(docs, desc=f"Generating with {model_path}"):
        inputs = tokenizer(doc['text'], return_tensors="pt", max_length=512, truncation=True).to(device)
        outputs = model.generate(**inputs, max_new_tokens=150)
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        summaries.append(summary)
    return summaries

def feature_extraction(s, d, I, args):
    """
    Extracts 4 distinct features from a sentence for extractive summarization.
    If args['ablation'] is False:
        @output: A numpy array of [length, position, tfidf, bert_sim]
    If args['ablation'] is True:
        @output: A numpy array of [tfidf, bert_sim]
    """
    features = []
    
    # Pre-calculate length and ensure it's at least 1 to avoid division by zero errors later
    s_words = s.split()
    s_length = max(1, len(s_words)) 
    
    # Check if we are running the ablation study
    ablation_mode = args.get('ablation', False)
    if not ablation_mode:
        # FEATURE 1: Sentence Length
        features.append(len(s_words))
        
        # FEATURE 2: Relative Position
        # !! LLM-aided code block
        doc_sentences = [sent.strip() for sent in re.split(r'(?<=[.!?]) +', d) if sent.strip()]
        try:
            pos = doc_sentences.index(s.strip())
            rel_pos = pos / max(1, len(doc_sentences) - 1)
        except ValueError:
            rel_pos = 0.5 
        features.append(rel_pos)
    
    # FEATURE 3: TF-IDF Score
    vectorizer = I['lexical']['vectorizer']
    sparse_vector = vectorizer.transform([s])
    
    # Sum the TF-IDF weights and normalize by sentence length
    tfidf_score = sparse_vector.sum() / s_length
    features.append(tfidf_score)

    # FEATURE 4: BERT Similarity
    encoder = args['encoder']
    s_embedding = encoder.encode([s], show_progress_bar=False)
    d_embedding = encoder.encode([d], show_progress_bar=False)
    similarity = cosine_similarity(s_embedding, d_embedding)[0][0]
    features.append(float(similarity))

    return np.array(features)

#!! LLM-aided code block
def training(D_train, R_train, args):
    """
    Trains an extractive summarization classifier.
    Handles feature extraction, label generation, and model fitting.
    Label generation: 1 for sentences that are part of the ground-truth summary, 0 otherwise.
    args: - 'I': The hybrid index from the indexing module
          - 'X_train': (optional) pre-computed feature matrix to save time
          - 'classifier': (optional) a scikit-learn classifier instance (default is RandomForest)
    """
    I = args['I']
    
    # FEATURE EXTRACTION (X_train)
    # If we already computed it we use it (save time). Otherwise, run the extraction loop
    if 'X_train' in args:
        print("Using pre-computed X_train to save time...")
        X_train = args['X_train']
    else:
        print("Extracting features for all training sentences (this may take a while)...")
        X_train = []
        doc_lookup = {str(doc['doc_id']): doc['text'] for doc in D_train}
        
        for k in tqdm(range(len(I['texts'])), desc="Extracting Features"):
            sentence = I['texts'][k]
            sent_id = I['ids'][k]
            doc_id = sent_id.split('_sent_')[0] 
            document_text = doc_lookup[doc_id]
            
            # Call your Function 2!
            features = feature_extraction(sentence, document_text, I, args)
            X_train.append(features)
            
        X_train = np.array(X_train)

    # LABEL GENERATION (y_train)
    ref_lookup = {str(r['doc_id']): r['text'] for r in R_train}
    scorer = rouge_scorer.RougeScorer(['rouge2'], use_stemmer=True)
    
    y_train = np.zeros(len(X_train))
    doc_to_indices = {}
    
    for k, sent_id in enumerate(I['ids']):
        doc_id = sent_id.split('_sent_')[0]
        if doc_id not in doc_to_indices:
            doc_to_indices[doc_id] = []
        doc_to_indices[doc_id].append(k)
        
    for doc_id, indices in doc_to_indices.items():
        reference_summary = ref_lookup.get(doc_id, "")
        
        ref_sentences = nltk.sent_tokenize(reference_summary)
        target_k = max(1, len(ref_sentences))
        
        scores = []
        for idx in indices:
            sentence_text = I['texts'][idx]
            score = scorer.score(reference_summary, sentence_text)['rouge2'].fmeasure
            scores.append((idx, score))
            
        # Sort by highest ROUGE score
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Label exactly as many sentences as the human summary had!
        for i in range(min(target_k, len(scores))):
            best_idx = scores[i][0]
            y_train[best_idx] = 1

    print(f"Labeling complete. Found {int(sum(y_train))} summary sentences out of {len(y_train)} total.")
    
    # MODEL TRAINING 
    print("Fitting the classification model ...")
    M = args.get('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1))
    M.fit(X_train, y_train)
    
    print("Model training complete!")
    return M

# !! LLM-aided code block
def summarization_supervised(d, M, p, l, args):
    """
    Generates an extractive summary of document d using the trained classifier M.
    Limits the summary to a maximum of p sentences.
    
    Returns:
        A string containing the chronological extractive summary.
    """
    I = args['I']
    sentences = nltk.sent_tokenize(d)
    
    if not sentences:
        return ""
        
    # Feature Extraction
    X_test = np.array([feature_extraction(s, d, I, args) for s in sentences])
    
    # Score the Sentences
    prob = M.predict_proba(X_test)[:, 1]
    
    # Rank the sentences (highest probability first)
    ranked_indices = np.argsort(prob)[::-1]
    
    # Enforce Limits
    selected_indices = []
    
    for idx in ranked_indices:
        sent = sentences[idx]
        # Store the index 
        selected_indices.append(idx)
        # Stop if sentence limit 'p' is reached
        if p is not None and len(selected_indices) >= p:
            break
            
    # Sort chronologically by original sentence index
    selected_indices.sort()
    
    # Reconstruct the summary text
    summary = " ".join([sentences[idx] for idx in selected_indices])
    
    return summary

# !! LLM-aided code block
def supervised_evaluation(D_test, R_test, M, args):
    """
    Evaluates the classifier M using confusion-based scores.
    """
    I = args['I']
    y_true = []
    X_test = []

    scorer = rouge_scorer.RougeScorer(['rouge2'], use_stemmer=True)
    
    print("Extracting features and ground-truth labels for evaluation...")
    # Loop through documents and references together
    for doc, ref in zip(D_test, R_test):
        d_text = doc['text']
        r_text = ref['text']
        
        sentences = nltk.sent_tokenize(d_text)
        
        if not sentences:
            continue

        # ROUGE-2 soft labeling (same as training logic)
        ref_sentences = nltk.sent_tokenize(r_text)
        target_k = max(1, len(ref_sentences))
        
        scored = []
        for s in sentences:
            score = scorer.score(r_text, s)['rouge2'].fmeasure
            scored.append(score)
        
        threshold_indices = sorted(range(len(scored)), 
                                   key=lambda i: scored[i], reverse=True)[:target_k]
        
        for i, s in enumerate(sentences):
            features = feature_extraction(s, d_text, I, args)
            X_test.append(features)
            y_true.append(1 if i in threshold_indices else 0)
                
    X_test = np.array(X_test)
    y_true = np.array(y_true)
    
    # Generate predictions
    y_pred = M.predict(X_test)
    y_proba = M.predict_proba(X_test)[:, 1] # Probabilities for AUC

    print(f"\n--- Prediction Results ---")
    print(f"Total sentences tested: {len(y_pred)}")
    print(f"Sentences actually in ground truth (y_true = 1): {sum(y_true)}")
    print(f"Sentences model PREDICTED as summary (y_pred = 1): {sum(y_pred)}")
    print(f"----------------------------\n")
    
    # Calculate confusion-based scores
    scores = {'precision': precision_score(y_true, y_pred, zero_division=0),
              'recall': recall_score(y_true, y_pred, zero_division=0),
              'f1_score': f1_score(y_true, y_pred, zero_division=0),
              'auc': roc_auc_score(y_true, y_proba) if len(set(y_true)) > 1 else float('nan')}
    
    print(confusion_matrix(y_true, y_pred))

    return scores