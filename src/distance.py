from scipy.spatial.distance import euclidean
import logging

def distance_to_weight(cosine_dsitance): cosine_dsitance

def distance(embeddings, word_a, word_b, distance_fn=euclidean):
    if word_a in embeddings and word_b in embeddings:
        return distance_fn(embeddings[word_a], embeddings[word_b])
    else:
        logging.debug('Cant calculate cosine similarity bethween {} and {}.'.format(word_a, word_b))
        return None

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