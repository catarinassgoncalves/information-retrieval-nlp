import spacy
import re
import nltk
import time
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

nlp = spacy.load("en_core_web_sm")

def process_text(text):
    """
    Clean text and extract individual words and normalized noun phrases.
    Prevents double-counting of words in phrases and filters out single-letter artifacts.
    """
    # Lowercase and remove weird whitespace/special characters
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)

    # Process the text through spaCy
    doc = nlp(text)
    processed_tokens = []
    
    # Track word indices already used in a phrase to avoid double-counting
    chunk_token_indices = set() 

    # Extract and Normalize Noun Phrases
    for chunk in doc.noun_chunks:
        # Collect tokens in the chunk that aren't stop words and are > 1 char
        valid_tokens = [token for token in chunk 
                        if not token.is_stop and token.is_alpha and len(token.lemma_) > 1]
        
        # EDA CONSTRAINT: Only keep phrases with 1 or 2 valid words
        if len(valid_tokens) > 0 and len(valid_tokens) <= 2:
            # Replace spaces with underscores to create a single cohesive token
            clean_chunk = "_".join([token.lemma_.lower() for token in valid_tokens])
            processed_tokens.append(clean_chunk)
            
            # Record indices so we skip these in the individual words loop
            for token in valid_tokens:
                chunk_token_indices.add(token.i)

    # Extract Remaining Individual Words
    for token in doc:
        # Only process if word wasn't part of a noun phrase above
        if token.i not in chunk_token_indices:
            # Standard cleaning + length filter
            if not token.is_stop and not token.is_punct and token.is_alpha and len(token.lemma_) > 1:
                processed_tokens.append(token.lemma_)

    return processed_tokens

# !! LLM-aided code block
def indexing(D, args):
    """
    Builds a hybrid index combining Scikit-Learn's sparse matrices with multiple dense LLM embeddings.
    """
    start_time = time.time()

    level = args.get('level', 'document')
    # List of encoders used to build the semantic index
    encoder_models = args.get('encoder_models', ['BAAI/bge-small-en-v1.5'])

    corpus_texts = []
    corpus_ids = []
    corpus_lengths = []

    # Prepare the corpus
    for doc in D:
        if level == 'document':
            corpus_texts.append(doc['text'])
            corpus_ids.append(doc['doc_id'])
            corpus_lengths.append(len(doc['text'].split()))

        elif level == 'sentence':
            sentences = nltk.sent_tokenize(doc['text'])
            for i, sent in enumerate(sentences):
                corpus_texts.append(sent)
                corpus_ids.append(f"{doc['doc_id']}_sent_{i}")
                corpus_lengths.append(len(sent.split()))

    # Lexical Component - Sparse Matrix
    print(f"Building Lexical Index (TF-IDF) for {len(corpus_texts)} items...")
    vectorizer = TfidfVectorizer(tokenizer=process_text, token_pattern=None)
    tfidf_matrix = vectorizer.fit_transform(corpus_texts)

    # Semantic Component - Dense LLM Embeddings
    semantic_index = {}
    for model_name in encoder_models:
        print(f"Building Semantic Index using LLM: {model_name}...")
        encoder = SentenceTransformer(model_name)
        # Store embeddings in a dictionary keyed by the model name
        semantic_index[model_name] = encoder.encode(corpus_texts, show_progress_bar=True)

    # Return
    I = {'level': level,
         'ids': corpus_ids,
         'texts': corpus_texts, # Store the raw text for easy retrieval later
         'lengths': corpus_lengths,
         'avg_length': sum(corpus_lengths) / len(corpus_lengths) if corpus_lengths else 0,
         'lexical': {'vectorizer': vectorizer, 'matrix': tfidf_matrix},
         'semantic': semantic_index # Contains embeddings for all requested LLMs
         }

    indexing_time = time.time() - start_time
    # Calculate memory size of the index
    memory_size_mb = (sys.getsizeof(I) + sys.getsizeof(tfidf_matrix)) / (1024 * 1024)
    for model_name, embeddings in semantic_index.items():
        memory_size_mb += sys.getsizeof(embeddings) / (1024 * 1024)

    print(f"Indexing Complete! Level: {level} | Time: {indexing_time:.2f}s | Size: {memory_size_mb:.2f} MB")

    return I, indexing_time, memory_size_mb