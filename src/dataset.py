from plot import plot_hist
from step import plot_frequency

def summary(dataset):
    print(dataset.shape)
    dataset = dataset \
        .pipe(plot_frequency, column = 'source') \
        .pipe(plot_frequency, column = 'response')

    plot_hist(
        lambda: dataset[dataset.sww_weight > 0].sww_weight, 
        xlabel = 'SWW: Probabilidad condicional (response / source)',
        density = True
    )
    plot_hist(
        lambda: dataset[dataset.w2v_google_weight > 0].w2v_google_weight, 
        xlabel = 'W2V: Pesos Google News',
        density = True
    )
    plot_hist(
        lambda: dataset[dataset.w2v_glove_weight > 0].w2v_glove_weight, 
        xlabel = 'W2V: Pesos Glove',
        density = True
    )
    return dataset.head()