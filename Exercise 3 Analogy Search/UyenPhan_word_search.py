import random
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

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
# print('Vocabulary size')
# print(len(vocab))
# print(vocab['man'])
# print(len(ivocab))
# print(ivocab[10])

# W contains vectors for
# print('Vocabulary word vectors')
vector_dim = len(vectors[ivocab[0]])
W = np.zeros((vocab_size, vector_dim))
for word, v in vectors.items():
    if word == '<unk>':
        continue
    W[vocab[word], :] = v  # populating the W matrix with word vectors
    # assigns the corresponding word vector v to the row of the W matrix.
# print(W.shape)


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


def find_most_similar_words_euclidean(input_word, word_vectors, num_similar=3):
    if input_word not in vocab:
        return []  # Word not found in the vocabulary

    # Get the vector for the input word
    input_vector = word_vectors[vocab[input_word]]

    # Compute Euclidean distances between the input word and all other words
    euclidean_distances = np.linalg.norm(word_vectors - input_vector, axis=1)

    # Get the indices of the most similar words (excluding the input word itself)
    similar_word_indices = np.argsort(euclidean_distances)[0:num_similar]

    # Retrieve the actual words and their distances based on the indices
    similar_words = [(ivocab[idx], euclidean_distances[idx]) for idx in similar_word_indices]

    return similar_words


# Main loop for searching similar words
while True:
    input_term = input("\nEnter word (EXIT to break): ")
    if input_term == 'EXIT':
        break
    else:
        similar_words = find_most_similar_words_euclidean(input_term, W)
        print("Most similar words: ")
        print("\n                               Word       Distance\n")
        print("---------------------------------------------------------\n")
        for word, distance in similar_words:
            print("%35s\t\t%f\n" % (word, distance))
