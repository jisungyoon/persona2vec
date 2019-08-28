import pandas as pd
import networkx as nx
from texttable import Texttable


def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable() 
    t.add_rows([["Parameter", "Value"]] +  [[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    print(t.draw())


def read_graph(input_file_path, weighted=False, directed=False):
    '''
    Reads the input network in networkx.
    :param input_file_path: File path of input graph
    :param weighted: weighted network(True) or unweighted network(False)
    :param directed: directed network(True) or undirected network(False)
    '''
    if weighted:
        G = nx.read_edgelist(input_file_path, nodetype=str, data=(('weight', float),), create_using=nx.DiGraph())
        
    else:
        G = nx.read_edgelist(input_file_path, nodetype=str, create_using=nx.DiGraph())
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1

    if not directed:
        G = G.to_undirected()

    return G
