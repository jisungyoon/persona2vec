import networkx as nx
import numpy as np
from tqdm import tqdm
from os.path import join as osjoin

from persona2vec.utils import mk_outdir
import pickle

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


class networkTrainTestSplitter(object):
    def __init__(self, G):
        self.G = G

        self.original_edge_set = set(G.edges)
        self.node_list = list(G.nodes)
        self.total_number_of_edges = len(self.original_edge_set)
        self.number_of_test_edges = int(self.total_number_of_edges / 2)

        self.test_edges = []
        self.negative_edges = []

    def train_test_split(self):
        logging.info('Initiate train test set split')
        while len(self.test_edges) != self.number_of_test_edges:
            edge_list = np.array(self.G.edges())
            candidate_idxs = np.random.choice(
                len(edge_list), self.number_of_test_edges - count, replace=False)
            for source, target in tqdm(edge_list[candidate_idxs]):
                self.G.remove_edge(source, target)
                if nx.is_connected(self.G):
                    count += 1
                    self.test_edges.append((source, target))
                else:
                    self.G.add_edge(source, target, weight=1)

    def generate_negative_edges(self):
        logging.info('Initiate generating negative edges')
        while len(self.negative_edges) != self.number_of_test_edges:
            source, target = np.random.choice(self.node_list, 2)
            if (source, target) in self.original_edge_set:
                pass
            else:
                count += 1
                self.negative_edges.append((source, target))

    def save_splitted_result(self, path):
        mk_outdir(path)
        nx.write_edgelist(self.G, osjoin(path, 'network.elist'))
        with open(osjoin(path, 'test_edges.pkl'), 'wb') as f:
            pickle.dump(self.test_edges, f)
        with open(osjoin(path, 'negative_edges.pkl'), 'wb') as f:
            pickle.dump(self.negative_edges, f)
        logging.info('Train-test splitter datas are stored')
