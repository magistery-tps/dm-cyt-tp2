import matplotlib.pyplot as plt
import networkx as nx


def plot_graph(
    graph, 
    k           = 0.4, 
    figsize     = (20, 7), 
    edge_color  = 'gray',
    with_labels = True,
    font_weight = 'bold'
):
    plt.figure(figsize=figsize)
    centrality = nx.eigenvector_centrality(graph)
    nx.draw(
        graph,
        pos = nx.spring_layout(graph, k=k), 
        with_labels = with_labels,
        font_weight = font_weight,
        node_color  = [v for n,v in centrality.items()], 
        edge_color  = edge_color
    )