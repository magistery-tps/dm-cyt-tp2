import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_hist(
    get_data_fn  = None,
    figsize      = (6,4),
    ylabel       = 'Frecuencia',
    xlabel       = 'x',
    title        = '',
    bins         = np.linspace(0, 1, 15)
):
    plt.figure(figsize=figsize)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title if title else 'Histograma de {}'.format(xlabel))
    plt.hist(get_data_fn(), bins=bins);