import random
import numpy as np

vocabulary_file = 'word_embeddings.txt'

# Read words
print('Read words...')
with open(vocabulary_file, 'r', encoding="utf8") as f:
    words = [x.rstrip().split(' ')[0] for x in f.readlines()]

# Read word vectors
print('Read word vectors...')
with open(vocabulary_file, 'r', encoding="utf8") as f:
    vectors = {}
    for line in f:
        vals = line.rstrip().split(' ')
        vectors[vals[0]] = [float(x) for x in vals[1:]]

vocab_size = len(words)
vocab = {w: idx for idx, w in enumerate(words)}
ivocab = {idx: w for idx, w in enumerate(words)}

# Vocabulary and inverse vocabulary (dict objects)
print('Vocabulary size')
print(len(vocab))
print(vocab['man'])
print(len(ivocab))
print(ivocab[10])

# W contains vectors for
print('Vocabulary word vectors')
vector_dim = len(vectors[ivocab[0]])
W = np.zeros((vocab_size, vector_dim))
for word, v in vectors.items():
    if word == '<unk>':
        continue
    W[vocab[word], :] = v  # populates the W matrix with word vectors
    # assigns the corresponding word vector v to the row of the W matrix.
print(W.shape)


# Function to find the three most similar words
"""
A cosine similarity is a value that is bound by a constrained range of 0 and 1. 
The closer the value is to 1, the smaller the angle between 2 words. Them the words are more similar
"""


def cosine(vector1, vector2):
    dot_product = np.dot(vector1, vector2)

    squared_sum_1 = np.sum(vector1 ** 2)
    norm_vector1 = np.sqrt(squared_sum_1)

    squared_sum_2 = np.sum(vector2 ** 2)
    norm_vector2 = np.sqrt(squared_sum_2)

    if norm_vector1 == 0 or norm_vector2 == 0:
        # Handle the case of zero norm (avoid division by zero)
        return 0.0
    else:
        cosine_similarity = dot_product / (norm_vector1 * norm_vector2)
        return cosine_similarity


def find_analogy(word1, word2, word3, top_n=2):
    if word1 not in vocab or word2 not in vocab or word3 not in vocab:
        return []

    # Calculate the analogy vector using cosine similarity
    vector1 = W[vocab[word1]]  # get the feature vector of word 1
    vector2 = W[vocab[word2]]
    vector3 = W[vocab[word3]]
    analogy_vector = vector3 + (vector2 - vector1)

    # Calculate cosine similarities between the analogy vector and all word vectors
    # create a list of tuples - for easier distance sort
    similarities = []

    for idx, vector in enumerate(W):
        # exclude the words in analogy query from similarities list
        if ivocab[idx] == word1 or ivocab[idx] == word2 or ivocab[idx] == word3:
            continue

        cosine_similarity = cosine(analogy_vector, vector)
        similarities.append((idx, cosine_similarity))  # append the word and its corresponding similarity
    # Find the top_n words with the highest cosine similarities
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Return the top_n most similar words
    return similarities[:top_n]


def find_word_analogy(word_a, word_b, word_c, word_vectors, num_similar=2):
    if word_a not in vocab or word_b not in vocab or word_c not in vocab:
        return "One or more words not found in the vocabulary."

    # Get the word vectors for the input words
    vector_a = word_vectors[vocab[word_a]]
    vector_b = word_vectors[vocab[word_b]]
    vector_c = word_vectors[vocab[word_c]]

    # Calculate the analogy vector using the Euclidean analogy formula
    analogy_vector = vector_c + (vector_b - vector_a)

    # Compute Euclidean distances between the input word and all other words
    euclidean_distances = np.linalg.norm(word_vectors - analogy_vector, axis=1)

    # Get the indices of the most similar words (excluding the input word itself)
    similar_word_indices = np.argsort(euclidean_distances)[0:num_similar]

    # Retrieve the actual words and their distances based on the indices
    similar_words = [(ivocab[idx], euclidean_distances[idx]) for idx in similar_word_indices]

    return similar_words


# Main loop for analogy
while True:
    input_term = input("\nEnter three words (EXIT to break): ")
    if input_term == 'EXIT':
        break
    else:
        input_words = input_term.split("-")
        word1, word2, word3 = input_words
        analogy_results = find_word_analogy(word1, word2, word3, W)

        print("\n                               Similarity score\n")
        print("---------------------------------------------------------\n")
        for word, distance in analogy_results:
            print("%35s\t\t%f\n" % (word, distance))
