import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import pyplot, patches
from plot import plot_hist
import math

import numpy as np
import pandas as pd
import networkx as nx
from plot import DEFAULT_FIGURE_SIZE

def normalize(values):
    return (values - np.min(values)) / (np.max(values) - np.min(values))


from graph import graph_cycles, \
                  graph_edge_weights, \
                  graph_subsampling, \
                  nodes_degree, \
                  centrality_measures, \
                  subgraph_without_isolated_nodes, \
                  graph_largest_component, \
                  to_undirected_graph



def plot_graph_eigenvector_centrality(graph, title):
    plot_graph(
        graph,
        node_color = [v for v in nx.eigenvector_centrality(graph).values()],        
        figsize = (15, 5),
        title   = title + ': Centralidad de autovectores' 
    )
    
def plot_degree_centrality(graph, title):
    plot_graph(
        graph,
        node_color = [v for v in nx.degree_centrality(graph).values()],        
        figsize = (15, 5),
        title   = title + ': Centralidad de grardo' 
    )

def plot_betweenness_centrality(graph, title):
    plot_graph(
        graph,
        node_color = [v for v in nx.betweenness_centrality(graph).values()],        
        figsize = (15, 5),
        title   = title + ': Centralidad de intermediación' 
    )

def plot_closeness_centrality(graph, title):
    plot_graph(
        graph,
        node_color = [v for v in nx.closeness_centrality(graph).values()],        
        figsize = (15, 5),
        title   = title + ': Centralidad de cercania' 
    )

def plot_adjacency_matrix2(
    graph, 
    title           = 'Matriz de adyacencia',
    title_font_size = 16,
    figsize         = DEFAULT_FIGURE_SIZE
):
    plt.figure(figsize = figsize)
    plt.matshow(nx.adjacency_matrix(graph).todense());
    plt.colorbar();
    plt.suptitle(title, fontsize = title_font_size)

def plot_modularity_coeficient(graph):
    plot_hist(
        lambda: generate_data(graph),
        xlabel = 'Diferentes particiones',
        ylabel = 'Coeficiente de Modularidad'
    )

def plot_adjacency_matrix(
    G, 
    node_order = None, 
    partitions = [], 
    colors     = [], 
    title      = 'Matriz de adyacencia',
    figsize    = DEFAULT_FIGURE_SIZE
):
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
    fig = pyplot.figure(figsize=figsize) # in inches
    pyplot.imshow(
        adjacency_matrix,
        cmap          = "Greys",
        interpolation = "none"
    )

    # The rest is just if you have sorted nodes by a partition and want to
    # highlight the module boundaries
    assert len(partitions) == len(colors)

    ax = pyplot.gca()
    for partition, color in zip(partitions, colors):
        current_idx = 0
        for module in partition:
            ax.add_patch(patches.Rectangle(
                (current_idx, current_idx),
                len(module), # Width
                len(module), # Height
                facecolor = "none",
                edgecolor = color,
                linewidth = "1"
            ))
            current_idx += len(module)
    pyplot.title(title)


def plot_graph(
    graph,
    nodes_distance   = 5,
    weight_desimals  = 2,
    figsize          = (15, 6), 
    edge_color       = 'gray',
    with_labels      = True,
    font_color       = 'red',
    font_weight      = 'bold',
    node_size        = 3800, 
    node_color       = 'blue',
    node_edge_colors = 'gray',
    title            = 'Grafo',
    alpha            = 0.99,
    color_map        = plt.cm.Greens
):
    plt.figure(figsize=figsize)

    pos     = nx.spring_layout(graph, k= nodes_distance / math.sqrt(graph.order()))
    weights = nx.get_edge_attributes(graph, 'weight')
    labels  = {pair: round(weight, weight_desimals)  for pair, weight in weights.items()}
    
    nx.draw(
        graph,
        pos, 
        with_labels = with_labels,
        font_weight = font_weight,
        edge_color  = edge_color,
        font_color  = font_color,
        node_color  = node_color,
        edgecolors  = node_edge_colors,
        node_size   = node_size,
        cmap        = color_map,
        alpha       = alpha
    )

    nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels)

    if not isinstance(node_color, str):
        sm = plt.cm.ScalarMappable(
            norm = plt.Normalize(vmin=min(node_color), vmax=max(node_color)),
            cmap = color_map
        )
        plt.colorbar(sm)

    plt.title(title, fontsize=16)

def plot_weight_dist(graph, title = ''):
    plot_hist(
        lambda: normalize(graph_edge_weights(graph)), 
        xlabel  = 'Peso',
        title   = (title + ' - ' if title else '') + 'Distribución del pesos de las aristas',
        density = True
    )

def plot_clustering_coeficient_dist(
    graph, 
    title = '',
    bins  = np.arange(0, 1, 0.1)
):
    plot_hist(
        lambda: list(nx.clustering(graph).values()), 
        xlabel  = 'Coeficiente de clustering',
        title   = (title + ' - ' if title else '') + 'Distribución del coeficiente de clustering',
        bins    = bins,
        density = True
    )

def plot_degree_dist(graph, title = ''):    
    plot_hist(
        lambda: normalize(nodes_degree(graph)), 
        xlabel = 'Grade de nodos',
        title  = (title + ' - ' if title else '') + 'Distribucion de grado',
        density = True
    )

def graph_summary(
    dataset,
    graph, 
    title='Grafo de palabras',
    font_color     = 'red',
    font_weight    = 'bold',
    k_percent      = 0.1,
    nodes_distance = 5,
    node_size      = 3800 
):
    print('Información:')
    print(nx.info(graph))
    if not nx.is_directed(graph):
        print('- ¿Es un grafo conexo? ', 'Si' if nx.is_connected(graph) else 'No')
    print('- ¿Es un grafo pesado? ', 'Si' if nx.is_weighted(graph) else 'No')
    print('- ¿Es un grafo dirigido? ', 'Si' if nx.is_directed(graph) else 'No')
    print('- ¿Es un grafo con ciclos? ', 'Si' if len(graph_cycles(graph)) > 0 else 'No')
    print('- ¿Es un grafo con multiples aristas? ', 'Si' if graph.is_multigraph() else 'No')

    plot_adjacency_matrix2(graph, title= title + ': Matriz de adyacencia')
    
    
    sub_graph = graph_subsampling(graph, k_percent)
    sub_graph = to_undirected_graph(dataset, sub_graph)
    sub_graph = subgraph_without_isolated_nodes(sub_graph)
    sub_graph = graph_largest_component(sub_graph)

    plot_graph(
        sub_graph,
        title          = title + ': Grafo de palabras pesado',
        font_color     = font_color,
        font_weight    = font_weight,
        node_color     = [v for v in nx.degree_centrality(sub_graph).values()],
        nodes_distance = nodes_distance,
        node_size      = node_size
    )

def plot_cumulative_degree_dists(
    graph_a,
    graph_b,
    label_a         = 'Graph A', 
    label_b         = 'Graph B', 
    ylabel          = 'Frecuencia',
    xlabel          = 'Grado',
    title           = 'Distribución de grados acumulada',
    title_font_size = 16,
    figsize         = DEFAULT_FIGURE_SIZE
):
    nodes_degree(graph_a).hist(
        density    = True,
        histtype   = 'step',
        label      = label_a, 
        cumulative = -1,
        figsize    = figsize
    );
    nodes_degree(graph_b).hist(
        density    = True,
        ax         = plt.gca(),
        histtype   = 'step',
        label      = label_b, 
        cumulative = -1,
        figsize    = figsize
    );
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title, fontsize = title_font_size)
    plt.legend()

def plot_centrality_mesures_heatmap(
    graph_a, 
    graph_b, 
    label_a, 
    label_b,
    figsize = DEFAULT_FIGURE_SIZE,
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
    hm = sns.heatmap(df_graph_a.join(df_graph_b).corr())
    hm.set_xticklabels(hm.get_xticklabels(), rotation=45, horizontalalignment='right')
