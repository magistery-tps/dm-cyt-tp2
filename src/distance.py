from scipy.spatial.distance import euclidean
import logging


def distance(embeddings, word_a, word_b, distance_fn=euclidean):
    return distance_fn(embeddings[word_a], embeddings[word_b])


def closest(embeddings, embedding, distance=euclidean):
    return sorted(
        embeddings.keys(), 
        key=lambda word: distance(embeddings[word], embedding)
    )

def show_closest(work_embeddings, word, distance):
    logging.info('- distance: {}'.format(distance.__name__))
    logging.info('- Word: {}'.format(word))
    closest_words = closest(work_embeddings, work_embeddings[word], distance=distance)[1:10]
    logging.info('- Closest: {}'.format(closest_words))