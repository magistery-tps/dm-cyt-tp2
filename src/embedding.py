import numpy as np
import logging
from tqdm import tqdm_notebook as tqdm
import sys

def work_embeddings(words, file_path):
    index = {word: True for word in words}
    with tqdm(total=len(index.keys()), file=sys.stdout) as pbar:
        count = 0
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    values = line.split()
                    if values[0] in index:
                        count += 1
                        yield (values[0], np.asarray(values[1:], "float32"))
                        pbar.update(1)
                except Exception as exception:
                    logging.warn('Load {} vector error!. {}'.format(word, exception))
        pbar.update(len(index.keys()) - count)
    logging.info('Found {} words.'.format(count))