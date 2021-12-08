import networkx as nx
import random
import pandas as pd

def graph_cycles(graph): 
    return nx.cycle_basis(graph.to_undirected())

def graph_edges(graph): 
    return graph.edges(data=True)

def graph_edge_weights(graph): 
    return [att['weight'] for n1, n2, att in graph.edges(data=True)]

def graph_subsampling(graph, k = 100):
    return graph.subgraph(random.sample(graph.nodes, k))

def nodes_degree(graph):
    return pd.DataFrame(
        graph.degree,
        columns=['Node','Degree']
    ).set_index('Node')