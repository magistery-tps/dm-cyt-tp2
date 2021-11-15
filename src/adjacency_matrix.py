import matplotlib.pyplot as plt
import pandas as pd

def plot_adjacency_matrix(matrix, title=''):
    plt.matshow(matrix)
    plt.colorbar()
    plt.suptitle(title)

def to_adjacency_matrix(df, column_a, column_b):
    df = pd.crosstab(df[column_a], df[column_b])
    indexes = df.columns.union(df.index)
    matrix = df.reindex(index = indexes, columns=indexes, fill_value=0)
    return matrix
