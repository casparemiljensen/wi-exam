import re
from time import time
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from datasets import load_dataset
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from gensim.test.utils import datapath
import logging

# Enable logging
logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)

# Load the CMU Book Summary Corpus dataset
dataset = load_dataset("textminr/cmu-book-summaries")['train']
summaries = [data['summary'] for data in dataset]

# Initialize stopwords
stop_words = set(stopwords.words("english"))


# Preprocessing function
def preprocess_texts(texts, stop_words):
    lemmatizer = WordNetLemmatizer()
    cleaned_texts = []
    for text in tqdm(texts, desc="Pre-processing text"):
        # Lowercase
        text = text.lower()
        # Remove punctuation and special characters
        text = re.sub(r"[^a-z\s]", "", text)
        # Tokenize
        words = word_tokenize(text)
        # Remove stopwords
        words = [word for word in words if word not in stop_words]
        # Lemmatize
        words = [lemmatizer.lemmatize(word) for word in words]
        # Join back into a string
        cleaned_texts.append(words)
    return cleaned_texts


# Preprocess the dataset
corpus = preprocess_texts(summaries, stop_words)
print(f"Total summaries loaded: {len(summaries)}")
print(f"Total preprocessed summaries in the corpus: {len(corpus)}")


# Function to train Word2Vec models
def train_word2vec_model(corpus, sg, vector_size, window, hs, negative, min_count):
    print(f"Training model with sg={sg}, vector_size={vector_size}, window={window}, negative={negative}")
    start_time = time()
    model = Word2Vec(
        sentences=corpus,
        vector_size=vector_size,
        window=window,
        sg=sg,  # ({0, 1}, optional) – Training algorithm: 1 for skip-gram; otherwise CBOW.
        hs=hs,  # hierarchial softmax
        negative=negative,   #  If > 0, negative sampling will be used, the int for negative specifies how many “noise words” should be drawn (usually between 5-20). If 0, negative sampling will not be used.
        workers=8,
        epochs=10,
        min_count=min_count,

    )
    elapsed_time = time() - start_time
    print(f"Model training completed in {elapsed_time:.2f} seconds.")
    return model


# Custom cosine similarity function
def cosine_similarity_terms(word1, word2, word_embeddings_df):
    vec1 = word_embeddings_df.loc[word1].values.reshape(1, -1)
    vec2 = word_embeddings_df.loc[word2].values.reshape(1, -1)
    similarity = cosine_similarity(vec1, vec2)
    return similarity[0][0]


# Function to find top similar words
def top_similar_words(word, word_embeddings_df, terms, top_n=5):
    target_vector = word_embeddings_df.loc[word].values.reshape(1, -1)
    similarities = cosine_similarity(word_embeddings_df, target_vector).flatten()
    top_indices = np.argsort(similarities)[-top_n-1:-1][::-1]
    similar_words = [(terms[i], similarities[i]) for i in top_indices]
    return similar_words


def evaluate_word_similarity(model, config):
    """
    Evaluates word similarity using the trained Word2Vec model.
    Prints cosine similarity between two specific words and the top similar words for a given word.
    """
    # print(f"\nEvaluating word similarity for configuration: {config}")

    # Convert the Word2Vec word vectors to a DataFrame for easier manipulation
    word_embs_df = pd.DataFrame(model.wv.vectors, index=model.wv.index_to_key)

    # Define words to test
    word1, word2 = "king", "queen"

    # Compute and print cosine similarity between two words
    similarity = cosine_similarity_terms(word1, word2, word_embs_df)
    # print(f"Cosine similarity between '{word1}' and '{word2}': {similarity:.4f}")

    # Find and print top 5 similar words for a given word
    similar_words = top_similar_words(word1, word_embs_df, model.wv.index_to_key, top_n=5)
    # print(f"Top 5 similar words to '{word}': {similar_words}")

    return {
        "similarity": similarity,
        "similar_words": similar_words,
        "details": (word1, similar_words)
    }


def evaluate_model(model, config):
    accuracy = model.wv.evaluate_word_analogies(datapath('questions-words.txt'))
    for section in accuracy[1]:
        correct_count = len(section['correct'])
        incorrect_count = len(section['incorrect'])
        total_count = correct_count + incorrect_count

        if total_count == 0:  # Handle sections with no examples
            section_accuracy = 0.0
            print(f"Problematic Section: {section['section']} - No evaluations performed.")
        else:
            section_accuracy = correct_count / total_count

    return accuracy


# Train and evaluate models with different configurations
def run_experiments():
    configurations = [

        {'sg': 1, 'vector_size': 100, 'window': 10, 'hs': 1, 'negative': 0, 'min_count': 5}, # baseline
        {'sg': 1, 'vector_size': 100, 'window': 10, 'hs': 0, 'negative': 2, 'min_count': 5},
        {'sg': 1, 'vector_size': 100, 'window': 10, 'hs': 0, 'negative': 5, 'min_count': 5},
        {'sg': 1, 'vector_size': 100, 'window': 10, 'hs': 0, 'negative': 10, 'min_count': 5},
        {'sg': 1, 'vector_size': 100, 'window': 10, 'hs': 0, 'negative': 15, 'min_count': 5},
        {'sg': 1, 'vector_size': 100, 'window': 10, 'hs': 0, 'negative': 20, 'min_count': 5},
    ]

    results = []
    total_start_time = time()

    for i, config in enumerate(configurations):
        print(f"\nRunning experiment {i+1}/{len(configurations)}...")
        start_time = time()

        # Train the model
        model = train_word2vec_model(corpus, **config)
        elapsed_time = time() - start_time

        # Evaluate the model
        accuracy = evaluate_model(model, config)
        result_sim = evaluate_word_similarity(model, config)

        # Store all results for this configuration
        results.append({
            'config': config,
            'accuracy': accuracy[0],  # Overall accuracy
            'section_accuracy': accuracy[1],  # Section-wise accuracy
            'time': elapsed_time,
            'similarity value': (result_sim["similarity"]),  # Prints similarity value
            'similarity words': (result_sim["similar_words"]),
            'details': (result_sim["details"])
        })

    total_elapsed_time = time() - total_start_time

    # Print detailed results for all configurations
    print("\nAll experiments completed.")
    print("=================================================")
    for result in results:
        config = result['config']
        accuracy = result['accuracy'] * 100  # Convert to percentage
        print(f"\nConfiguration: {config}")
        print(f"Overall Accuracy: {accuracy:.2f}%")
        print(f"Training Time: {result['time']:.2f} seconds")
        print("Section-wise Results:")
        for section in result['section_accuracy']:
            correct_count = len(section['correct'])
            incorrect_count = len(section['incorrect'])
            total_count = correct_count + incorrect_count

            if total_count == 0:  # Handle sections with no examples
                section_accuracy = 0.0
                print(f"  {section['section']}: No evaluations performed.")
            else:
                section_accuracy = correct_count / total_count
                print(f"  {section['section']}: {section_accuracy:.4f} accuracy "
                      f"({correct_count} correct, {incorrect_count} incorrect)")

    print("=================================================")
    print(f"Total time for all experiments: {total_elapsed_time:.2f} seconds.")


if __name__ == "__main__":
    run_experiments()
