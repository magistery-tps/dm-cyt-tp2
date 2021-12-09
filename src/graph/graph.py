import networkx as nx
import random
import pandas as pd
from networkx.algorithms.community import girvan_newman, modularity

def isolated_nodes(graph): return set(nx.isolates(graph))

def subgraph_without_isolated_nodes(graph):
    return graph.subgraph(set(graph.nodes) - isolated_nodes(graph))

def graph_modularity(graph):
    result = []
    for module in girvan_newman(graph):
        try:
            result.append(modularity(partition_set_to_dict(module), graph))
        except Exception as exception:
            pass
    return result

def graph_cycles(graph): 
    return nx.cycle_basis(graph.to_undirected())

def graph_edges(graph): 
    return graph.edges(data=True)

def graph_edge_weights(graph): 
    return [att['weight'] for n1, n2, att in graph.edges(data=True)]

def graph_subsampling(graph, percent = 0.1):
    k = int(len(graph.nodes) * percent)
    nodes = random.sample(graph.nodes, k)
    return graph.subgraph(nodes)

def nodes_degree(graph):
    return pd.DataFrame(
        graph.degree,
        columns=['Node','Degree']
    ).set_index('Node')

def centrality_measures(graph, max_iter=1000):
    return [
        nx.degree_centrality(graph).values(),
        nx.betweenness_centrality(graph).values(),
        nx.closeness_centrality(graph).values(),
        nx.eigenvector_centrality(graph, max_iter=max_iter).values()
    ]

def partition_set_to_dict(m):
    d = {}
    for i,c in enumerate(m):
        for n in c:
            d[n] = i
    return d