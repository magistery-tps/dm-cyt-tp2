import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx
from matplotlib import pyplot, patches
from graph import graph_cycles, \
                  graph_edge_weights, \
                  graph_subsampling, \
                  nodes_degree

def plot_adjacency_matrix(G, node_order=None, partitions=[], colors=[], title='Matriz de adyacencia'):
    """
    - G is a netorkx graph
    - node_order (optional) is a list of nodes, where each node in G
          appears exactly once
    - partitions is a list of node lists, where each node in G appears
          in exactly one node list
    - colors is a list of strings indicating what color each
          partition should be
    If partitions is specified, the same number of colors needs to be
    specified.
    """
    adjacency_matrix = nx.to_numpy_matrix(G, dtype=np.bool, nodelist=node_order)

    #Plot adjacency matrix in toned-down black and white
    fig = pyplot.figure(figsize=(5, 5)) # in inches
    pyplot.imshow(adjacency_matrix,
                  cmap="Greys",
                  interpolation="none")
    
    # The rest is just if you have sorted nodes by a partition and want to
    # highlight the module boundaries
    assert len(partitions) == len(colors)
    ax = pyplot.gca()
    for partition, color in zip(partitions, colors):
        current_idx = 0
        for module in partition:
            ax.add_patch(patches.Rectangle((current_idx, current_idx),
                                          len(module), # Width
                                          len(module), # Height
                                          facecolor="none",
                                          edgecolor=color,
                                          linewidth="1"))
            current_idx += len(module)
    
    pyplot.title(title)

def plot_graph(
    graph, 
    k           = 0.01, 
    figsize     = (25, 10), 
    edge_color  = 'gray',
    with_labels = True,
    font_color  = 'black',
    font_weight = 'normal',
    node_size   = 1000, 
    node_color  = 'tomato',
    title       = 'Grafo'
):
    plt.figure(figsize=figsize)
    centrality = nx.eigenvector_centrality(graph)
    nx.draw(
        graph,
        pos = nx.spring_layout(graph, k=k), 
        with_labels = with_labels,
        font_weight = font_weight,
        edge_color  = edge_color,
        font_color  = font_color,
        node_color  = node_color,
        node_size   = node_size 
    )
    plt.title(title)

def plot_edge_weight_hist(
    graph,
    figsize      = (6,4),
    ylabel       = 'Frecuencia',
    xlabel       = 'Peso',
    title        = 'Distribución del pesos de las aristas',
    title_prefix = '',
    bins         = np.linspace(0, 1, 15)
):
    title = title_prefix + ': ' + title if title_prefix else title
    plt.figure(figsize=figsize)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.hist(graph_edge_weights(graph), bins = bins);

def plot_clustering_coeficient_hist(
    graph,
    figsize      = (6,4),
    ylabel       = 'Frecuencia',
    xlabel       = 'Coeficiente de clustering',
    title        = 'Distribución del coeficiente de clustering',
    title_prefix = '',
    bins         = np.arange(0.2, 1, 0.05)
):
    title = title_prefix + ': ' + title if title_prefix else title
    plt.figure(figsize=figsize)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.hist(nx.clustering(graph).values(), bins=bins);


def graph_summary(
    graph, 
    title='Grafo de palabras',
    font_color  = 'tomato',
    font_weight = 'bold',
    k           = 50
):
    print(nx.info(graph))
    print('Es pesado? ', 'Si' if nx.is_weighted(graph) else 'No')
    print('Es Dirigido? ', 'Si' if nx.is_directed(graph) else 'No')
    print('Tiene ciclos? ', 'Si' if len(graph_cycles(graph)) > 0 else 'No')
    print('Tiene multiples aristas? ', 'Si' if graph.is_multigraph() else 'No')

    plot_adjacency_matrix(graph)

    sub_graph = graph_subsampling(graph, k)

    plot_graph(
        sub_graph,
        title       = title,
        font_color  = font_color,
        font_weight = font_weight,
        node_color  = [v for v in nx.degree_centrality(sub_graph).values()]
    )
    

def plot_nodes_degree_hist(
    graph,
    figsize      = (6,4),
    ylabel       = 'Frecuencia',
    xlabel       = 'Grade de nodos',
    title        = 'Distribución de grados',
    title_prefix = ''
):
    title = title_prefix + ': ' + title if title_prefix else title
    plt.figure(figsize=figsize)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.hist(nodes_degree(graph))


def plot_cumulative_nodes_degree_hist_comparative(
    graph_a,
    graph_b,
    label_a      = 'Graph A', 
    label_b      = 'Graph B', 
    ylabel       = 'Frecuencia',
    xlabel       = 'Grado',
    title        = 'Distribución de grados acumulada'
):
    plt.figure(figsize=(8,4))
    nodes_degree(graph_a).hist(
        density    = True,
        histtype   = 'step',
        label      = label_a, 
        cumulative = -1
    );
    nodes_degree(graph_b).hist(
        density    = True,
        ax         = plt.gca(),
        histtype   = 'step',
        label      = label_b, 
        cumulative = -1
    );
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.legend()

def plot_centrality_mesures_heatmap(
    graph_a, 
    graph_b, 
    label_a, 
    label_b,
    figsize = (8,4),
    title   = 'Correlación de medidas de centralidad'
):
    X = [
        nx.degree_centrality(graph_a).values(),
        nx.betweenness_centrality(graph_a).values(),
        nx.closeness_centrality(graph_a).values(),
        nx.eigenvector_centrality(graph_a).values()
    ]

    df_graph_b = pd.DataFrame(X, index=[
        '{} - Degree'.format(label_a),
        '{} - Betweeness'.format(label_a),
        '{} - Closeness'.format(label_a),
        '{} - Eigenvector'.format(label_a)
    ]).T

    X = [
        nx.degree_centrality(graph_b).values(),
        nx.betweenness_centrality(graph_b).values(),
        nx.closeness_centrality(graph_b).values(),
        nx.eigenvector_centrality(graph_b).values()
    ]
    
    df_graph_b = pd.DataFrame(X, index=[
        '{} - Degree'.format(label_b),
        '{} - Betweeness'.format(label_b),
        '{} - Closeness'.format(label_b),
        '{} - Eigenvector'.format(label_b)
    ]).T

    sns.heatmap(df_graph_b.join(df_graph_b).corr())
    plt.figure(figsize=figsize)
    plt.title(title)
