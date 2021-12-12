import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

DEFAULT_FIGURE_SIZE = (10,5)

def plot_hist(
    get_data_fn     = None,
    figsize         = DEFAULT_FIGURE_SIZE,
    ylabel          = 'Frecuencia',
    xlabel          = 'x',
    title           = '',
    bins            = np.linspace(0, 1, 15),
    density         = False,
    title_font_size = 16
):
    plt.figure(figsize=figsize)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(
        title if title else 'Histograma de {}'.format(xlabel), 
        fontsize = title_font_size
    )
    
    if density:
        sns.distplot(get_data_fn(), hist=True, bins=bins)
    else:
        sns.histplot(get_data_fn(), bins=bins);
    