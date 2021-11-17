import matplotlib.pyplot as plt
import networkx as nx


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
