import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import pyplot, patches
from plot import plot_hist

import numpy as np
import pandas as pd
import networkx as nx

from graph import graph_cycles, \
                  graph_edge_weights, \
                  graph_subsampling, \
                  nodes_degree, \
                  centrality_measures, \
                  subgraph_without_isolated_nodes

def plot_modularity_coeficient(graph):
    plot_hist(
        lambda: generate_data(graph),
        xlabel = 'Diferentes particiones',
        ylabel = 'Coeficiente de Modularidad'
    )

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
    k               = 0.01,
    weight_desimals = 2,
    figsize         = (25, 10), 
    edge_color      = 'gray',
    with_labels     = True,
    font_color      = 'tomato',
    font_weight     = 'bold',
    node_size       = 2500, 
    node_color      = 'tomato',
    title           = 'Grafo'
):
    plt.figure(figsize=figsize)
    centrality = nx.eigenvector_centrality(graph)
    pos        = nx.spring_layout(graph, k=k)
    weights    = nx.get_edge_attributes(graph, 'weight')
    labels     = {pair: round(weight, weight_desimals)  for pair, weight in weights.items()}
    
    nx.draw(
        graph,
        pos, 
        with_labels = with_labels,
        font_weight = font_weight,
        edge_color  = edge_color,
        font_color  = font_color,
        node_color  = node_color,
        node_size   = node_size
    )

    nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels)

    plt.title(title)

def plot_edge_weight_hist(graph, title = ''):
    plot_hist(
        lambda: graph_edge_weights(graph), 
        xlabel = 'Peso',
        title  = (title + ' - ' if title else '') + 'Distribución del pesos de las aristas'
    ) 

def plot_clustering_coeficient_hist(graph, title = ''):
    plot_hist(
        lambda: nx.clustering(graph).values(), 
        xlabel = 'Coeficiente de clustering',
        title  = (title + ' - ' if title else '') + 'Distribución del coeficiente de clustering',
        bins   = np.arange(0.2, 1, 0.05)
    )

def plot_nodes_degree_hist(graph, title = ''):    
    plot_hist(
        lambda: nodes_degree(graph), 
        xlabel = 'Grade de nodos',
        title  = (title + ' - ' if title else '') + 'Distribucion de grado'
    )

def graph_summary(
    graph, 
    title='Grafo de palabras',
    font_color  = 'tomato',
    font_weight = 'bold',
    k_percent   = 0.1,
    k_layout    = 0.01,
    node_size   = 3800 
):
    print(nx.info(graph))
    print('Es pesado? ', 'Si' if nx.is_weighted(graph) else 'No')
    print('Es dirigido? ', 'Si' if nx.is_directed(graph) else 'No')
    print('Tiene ciclos? ', 'Si' if len(graph_cycles(graph)) > 0 else 'No')
    print('Tiene multiples aristas? ', 'Si' if graph.is_multigraph() else 'No')

    plot_adjacency_matrix(graph)

    sub_graph = graph_subsampling(graph, k_percent)
    sub_graph = subgraph_without_isolated_nodes(sub_graph)

    plot_graph(
        sub_graph,
        title       = title,
        font_color  = font_color,
        font_weight = font_weight,
        node_color  = [v for v in nx.degree_centrality(sub_graph).values()],
        k           = k_layout,
        node_size   = node_size
    )

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
    max_iter = 5000
):
    df_graph_a = pd.DataFrame(
        centrality_measures(graph_a, max_iter), 
        index=[
            '{} - Degree'.format(label_a),
            '{} - Betweeness'.format(label_a),
            '{} - Closeness'.format(label_a),
            '{} - Eigenvector'.format(label_a)
        ]
    ).T

    df_graph_b = pd.DataFrame(
        centrality_measures(graph_b, max_iter), 
        index=[
            '{} - Degree'.format(label_b),
            '{} - Betweeness'.format(label_b),
            '{} - Closeness'.format(label_b),
            '{} - Eigenvector'.format(label_b)
        ]
    ).T

    plt.figure(figsize=figsize)
    plt.title('Correlación de medidas de centralidad')
    sns.heatmap(df_graph_a.join(df_graph_b).corr())

