import networkx as nx

class GraphFactory:
    @staticmethod
    def create_weihted_graph(df, graph_type, origin='source', target='response'):
        return nx.from_pandas_edgelist(
            df, 
            origin,  
            target,
            'weight',
            create_using = graph_type
        )

    @staticmethod
    def create_undirected_weihted_graph(df):
        return GraphFactory.create_weihted_graph(df, graph_type = nx.Graph())

    @staticmethod
    def create_directed_weihted_graph(df):
        return GraphFactory.create_weihted_graph(df, graph_type = nx.DiGraph())