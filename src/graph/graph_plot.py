import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx
from matplotlib import pyplot, patches
from graph import graph_cycles, graph_edge_weights

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
    k           = 0.17, 
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
    ylabel = 'Frecuencia',
    xlabel = 'Peso',
    title  = 'Distribución del pesos de las aristas',
    bins   = np.linspace(0, 1, 15)
):
    plt.figure()
    plt.ylabel(ylabel)
    plt.xlabel(xlabel);
    plt.title(title);
    plt.hist(graph_edge_weights(graph), bins = bins);

def graph_summary(
    graph, 
    title='Grafo de palabras',
    font_color  = 'tomato',
    font_weight = 'bold'
):
    print(nx.info(graph))
    print('Es pesado? ', 'Si' if nx.is_weighted(graph) else 'No')
    print('Es Dirigido? ', 'Si' if nx.is_directed(graph) else 'No')
    print('Tiene ciclos? ', 'Si' if len(graph_cycles(graph)) > 0 else 'No')
    print('Tiene multiples aristas? ', 'Si' if graph.is_multigraph() else 'No')

    plot_adjacency_matrix(graph)

    plot_graph(
        graph, 
        title=title,
        font_color  = font_color,
        font_weight = font_weight,
        node_color  = [v for v in nx.degree_centrality(graph).values()]
    )