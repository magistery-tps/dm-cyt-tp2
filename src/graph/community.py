import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from graph import plot_graph
import community.community_louvain as community_louvain

def plot_graph_partitions(graph, com_max, title, nodes_distance = 0.1):
    plot_graph(
        graph,
        node_color     = [v for n,v in partition_set_to_dict(com_max).items()],        
        figsize        = (15, 5),
        title          = title,
        with_labels    = False,
        with_weights   = False,
        node_size      = 250,
        nodes_distance = nodes_distance
    )

def plot_graph_louvain_partitions(graph, title, nodes_distance = 0.1):
    partition = community_louvain.best_partition(graph)
    plot_graph(
        graph,
        node_color     = [v for n,v in partition.items()],        
        figsize        = (15, 5),
        title          = title,
        with_labels    = False,
        with_weights   = False,
        node_size      = 250,
        nodes_distance = nodes_distance
    )

def assign_community_girvan_newman(graph):
    modulos     = nx.community.girvan_newman(graph)
    modularidad = []
    mod_max     = -999

    for communities in modulos:
        #print(tuple(sorted(c) for c in communities))

        n_modularidad = nx.community.modularity(
            graph,
            tuple(sorted(c) for c in communities)
        )

        #print(n_modularidad)
        
        modularidad.append(n_modularidad)
        
        if n_modularidad > mod_max:
            mod_max = n_modularidad
            com_max = tuple(sorted(c) for c in communities)

    #print(modularidad)
    
    plt.plot(modularidad);
    plt.xlabel('Diferentes particiones')
    plt.ylabel('Coeficiente de Modularidad')

    k = np.argmax(modularidad)

    print('Maxima Modularidad:', mod_max)
    # print(com_max)

    return k, mod_max, com_max, communities

def partition_set_to_dict(m):
    d = {}
    for i,c in enumerate(m):
        for n in c:
            d[n] = i
    return d