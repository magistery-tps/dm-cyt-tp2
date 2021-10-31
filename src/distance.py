from scipy.spatial.distance import euclidean

def closest(embeddings, embedding, distance=euclidean):
    return sorted(
        embeddings.keys(), 
        key=lambda word: distance(embeddings[word], embedding)
    )

def show_closest(work_embeddings, word, distance):
    print('- distance: {}'.format(distance.__name__))
    print('- Word: {}'.format(word))
    closest_words = closest(work_embeddings, work_embeddings[word], distance=distance)[1:10]
    print('- Closest: {}\n'.format(closest_words))