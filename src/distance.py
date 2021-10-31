from scipy.spatial.distance import euclidean
import logging

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