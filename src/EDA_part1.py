import pandas as pd
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import spacy

# !! LMM-aided code block
# Determining the Optimal Summary Length, p, per Category
def analyze_sentence_distribution(D):
    # 1. Prepare Data
    data = []
    for doc in D:
        sentences = nltk.sent_tokenize(doc['text'])
        data.append({
            'category': doc['category'],
            'sent_count': len(sentences)
        })
    df = pd.DataFrame(data)

    # 2. Visualize the "Natural Length" in Separate Subplots
    categories = df['category'].unique()
    
    # Create a grid of plots: 3 rows, 2 columns 
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 12))
    axes = axes.flatten() 

    fig.suptitle("Distribution of Sentence Counts per Article by Category", fontsize=16, y=1.02)

    for i, cat in enumerate(categories):
        # Filter data for just this one category
        cat_data = df[df['category'] == cat]
        
        # Plot the individual histogram
        sns.histplot(data=cat_data, x='sent_count', kde=True, element="step", ax=axes[i], color=sns.color_palette("husl", 5)[i])
        
        # Calculate and draw the median specifically for this category
        cat_median = cat_data['sent_count'].median()
        axes[i].axvline(cat_median, color='red', linestyle='--', label=f'Median: {cat_median:.1f}')
        
        axes[i].set_title(f"Category: {cat.capitalize()}")
        axes[i].set_xlabel("Number of Sentences")
        axes[i].set_ylabel("Frequency")
        axes[i].legend()

    fig.delaxes(axes[5])

    plt.savefig("my_graph.png", dpi=300, bbox_inches="tight")
    plt.tight_layout()
    plt.show()

    # 3. Calculate Dynamic p per Category
    # We use a heuristic: p = 20% of the median length, with a minimum of 3
    # This ensures we respect the "natural length" of each genre.
    category_stats = df.groupby('category')['sent_count'].agg(['median', 'mean', 'std'])
    
    p_mapping = {}
    for cat, row in category_stats.iterrows():
        suggested_p = int(np.round(row['median'] * 0.20))
        p_mapping[cat] = max(3, suggested_p) 
        
    return p_mapping, category_stats

# LLM-aided code block
# Determine the dimension of noun-phrases
nlp = spacy.load("en_core_web_sm")

def analyze_noun_phrase_complexity(D):
    np_lengths = []
    
    # We removed the sample limit and now iterate over the entire dataset D
    for doc in D:
        spacy_doc = nlp(doc['text'])
        for chunk in spacy_doc.noun_chunks:
            # Counts the number of tokens (words) in each phrase
            length = len(chunk)
            np_lengths.append(length)
    
    df_np = pd.DataFrame(np_lengths, columns=['elements'])
    
    plt.figure(figsize=(10, 5))
    sns.countplot(data=df_np, x='elements', palette="viridis", hue='elements', legend=False)
    plt.title("Noun Phrase Complexity: Number of Elements per Phrase (Full Corpus)")
    plt.xlabel("Number of Words (Tokens) in Phrase")
    plt.ylabel("Frequency")
    plt.show()
    
    stats = df_np['elements'].value_counts(normalize=True).sort_index() * 100
    return stats