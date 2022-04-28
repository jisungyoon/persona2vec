import csv
import errno
import logging
import os
from scipy import sparse
import networkx as nx
import numpy as np
from texttable import Texttable

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows(
        [["Parameter", "Value"]]
        + [[k.replace("_", " ").capitalize(), args[k]] for k in keys]
    )
    print(t.draw())


def mk_outdir(out_path):
    """
    Check and make a directory
    :param out_path: path for directory
    """
    if not os.path.exists(out_path):
        try:
            os.makedirs(out_path)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
    logging.info("output directory is created")


def read_graph(input_file_path, weighted=False, directed=False):
    """
    Reads the input network and return a networkx Graph object.
    :param input_file_path: File path of input graph
    :param weighted: weighted network(True) or unweighted network(False)
    :param directed: directed network(True) or undirected network(False)
    :return G: output network
    """
    if weighted:
        G = nx.read_edgelist(
            input_file_path,
            nodetype=str,
            create_using=nx.DiGraph(),
        )
    else:
        G = nx.read_edgelist(input_file_path, nodetype=str, create_using=nx.DiGraph())
        for edge in G.edges():
            G[edge[0]][edge[1]]["weight"] = 1

    if not directed:
        G = G.to_undirected()

    return G


def read_edge_file(file_path):
    """
    Read a edge list for link prediction (test or negative edges)
    :param file_path: Path of edge lists (.tsv format)
    :return: edge list
    """
    with open(file_path, "rt") as f:
        reader = csv.reader(f, delimiter="\t")
        data = list(map(tuple, reader))
    return data


#
# Homogenize the data format
#
def to_adjacency_matrix(net):
    if sparse.issparse(net):
        if type(net) == "scipy.sparse.csr.csr_matrix":
            return net
        return sparse.csr_matrix(net)
    elif "networkx" in "%s" % type(net):
        return nx.adjacency_matrix(net)
    elif "numpy.ndarray" == type(net):
        return sparse.csr_matrix(net)
