import numpy as np
import logging
from tqdm import tqdm_notebook as tqdm
import sys
from gensim.models import KeyedVectors
from itertools import combinations
import logging as logger


class GoogleW2VSimilarity:
    def __init__(self, path):
        self.model = KeyedVectors.load_word2vec_format(path, binary=True)

    def cosine(self, word_a, word_b):
        try:
            return self.model.similarity(word_a, word_b)
        except Exception as e:
            logger.debug('Failed to calculare cosine similarity bethween {} and {}.'.format(word_a, word_b))
            return None
