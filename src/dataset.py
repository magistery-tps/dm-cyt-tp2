from plot import plot_hist


def summary(dataset):
    plot_hist(
        lambda: dataset[dataset.google_weight > 0].google_weight, 
        xlabel = 'Pesos Google News'
    )
    plot_hist(
        lambda: dataset[dataset.glove_weight > 0].glove_weight, 
        xlabel = 'Pesos Glove'
    )
    return dataset.head()