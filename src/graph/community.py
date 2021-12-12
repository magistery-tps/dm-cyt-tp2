import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def assign_community_girvan_newman(G):
    modulos     = nx.community.girvan_newman(G)
    modularidad = []
    mod_max     = -999

    for communities in modulos:
        #print(tuple(sorted(c) for c in communities))

        n_modularidad = nx.community.modularity(G,tuple(sorted(c) for c in communities))

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

    print(mod_max)
    # print(com_max)

    return k, mod_max, com_max, communities

def partition_set_to_dict(m):
    d = {}
    for i,c in enumerate(m):
        for n in c:
            d[n] = i
    return d