import csv
import logging
from os.path import join as osjoin

import networkx as nx
import numpy as np
from persona2vec.utils import mk_outdir
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


class NetworkTrainTestSplitter(object):
    """
    Train and Test set splitter for network data.
    This class is for link prediction tasks.
    """

    def __init__(self, G, directed=False, fraction=0.5):
        """
        :param G: Networkx graph object. Origin Graph
        :param fraction: Fraction of edges that will be removed (test_edge).
        """
        self.G = G
        self.directed = directed

        self.original_edge_set = set(G.edges)
        self.node_list = list(G.nodes)
        self.total_number_of_edges = len(self.original_edge_set)
        self.number_of_test_edges = int(self.total_number_of_edges * fraction)

        self.test_edges = []
        self.negative_edges = []

    def train_test_split(self):
        """
        Split train and test edges.
        Train network should have a one weakly connected component.
        """
        logging.info("Initiate train test set split")
        check_connectivity_method = (
            nx.is_weakly_connected if self.directed else nx.is_connected
        )
        while len(self.test_edges) != self.number_of_test_edges:
            edge_list = np.array(self.G.edges())
            candidate_idxs = np.random.choice(
                len(edge_list),
                self.number_of_test_edges - len(self.test_edges),
                replace=False,
            )
            for source, target in tqdm(edge_list[candidate_idxs]):
                self.G.remove_edge(source, target)
                if check_connectivity_method(self.G):
                    self.test_edges.append((source, target))
                else:
                    self.G.add_edge(source, target, weight=1)

    def train_test_split_fast(self):
        """
        Split train and test edges.
        Train network should have a one weakly connected component.
        """
        logging.info("Initiate train test set split")
        if self.directed:
            logging.error("Not implemented for directed graph")
            return

        while len(self.test_edges) != self.number_of_test_edges:
            edge_list = np.array(self.G.edges())
            candidate_idxs = np.random.choice(
                len(edge_list),
                self.number_of_test_edges - len(self.test_edges),
                replace=False,
            )
            for source, target in tqdm(edge_list[candidate_idxs]):
                # cases sure cannot remove the edge:
                # one node is dangling
                if self.G.degree(source) == 1 or self.G.degree(target) == 1:
                    continue

                self.G.remove_edge(source, target)
                # cases sure can remove the edge:
                # source is reachable for target through other nodes
                # instead of using the default check connectivity method
                # here we use a lazy BFS method to stop early if target is reachable
                reachable = False
                seen = {}  # level (number of hops) when seen in BFS
                level = 0  # the current level
                nextlevel = {source: 1}  # dict of nodes to check at next level
                while nextlevel:
                    thislevel = nextlevel  # advance to next level
                    nextlevel = {}  # and start a new list (fringe)
                    for v in thislevel:
                        if v not in seen:
                            seen[v] = level  # set the level of vertex v
                            nextlevel.update(self.G[v])  # add neighbors of v
                    if target in seen:
                        reachable = True
                        break
                    level = level + 1

                if reachable:
                    self.test_edges.append((source, target))
                else:
                    self.G.add_edge(source, target, weight=1)

    def generate_negative_edges(self):
        """
        Generate a negative samples for link prediction task
        """
        logging.info("Initiate generating negative edges")
        while len(self.negative_edges) != self.number_of_test_edges:
            source, target = np.random.choice(self.node_list, 2)
            if (source, target) not in self.original_edge_set and (
                target,
                source,
            ) not in self.original_edge_set:
                self.negative_edges.append((source, target))

    def save_splitted_result(self, path):
        """
        :param path: path for saving result files (train network, test_edges, negative edges)
        """
        mk_outdir(path)
        nx.write_edgelist(self.G, osjoin(path, "network.elist"))
        with open(osjoin(path, "test_edges.tsv"), "wt") as f:
            tsv_writer = csv.writer(f, delimiter="\t")
            tsv_writer.writerows(self.test_edges)
        with open(osjoin(path, "negative_edges.tsv"), "wt") as f:
            tsv_writer = csv.writer(f, delimiter="\t")
            tsv_writer.writerows(self.negative_edges)
        logging.info("Train-test splitter datas are stored")
