import networkx as nx

def graph_cycles(graph): 
    return nx.cycle_basis(graph.to_undirected())

def graph_edges(graph): 
    return graph.edges(data=True)

def graph_edge_weights(graph): 
    return [att['weight'] for n1, n2, att in graph.edges(data=True)]