import matplotlib.pyplot as plt
import networkx as nx
from adjacency_matrix import plot_adjacency_matrix

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

    
def graph_edges(graph): return graph.edges(data=True)
    
def graph_summary(
    graph, 
    title='Grafo de palabras',
    font_color  = 'tomato',
    font_weight = 'bold'
):
    print(nx.info(graph))
    print('Is weigthed:',nx.is_weighted(graph))
    plot_adjacency_matrix(graph)
    plot_graph(
        graph, 
        title=title,
        font_color  = font_color,
        font_weight = font_weight,
        node_color  = [v for v in nx.degree_centrality(graph).values()]
    )