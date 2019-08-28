import numpy as np
import networkx as nx
import random
from tqdm import tqdm
import pickle

from base_node2vec import Node2Vec
from ego_splitting import EgoNetSplitter

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


class Persona2Vec(Node2Vec):
    """
    Persona2Vec node embedding object
    """
    def __init__(self, G,
                 lambd = 0.1,
                 directed=False,
                 num_walks=10,
                 walk_length=80,
                 p=1,
                 q=1,
                 dimensions=128,
                 window_size=10,
                 base_iter=1,
                 workers=1):
        """
        :param G: NetworkX graph object. persona graah
        :param lambd: edge weight for persona edge, usually 0 ~ 1
        :param directed: Directed network(True) or undirected network(False)
        :param num_walks: Number of random walker per node
        :param walk_length: Length(number of nodes) of random walker
        :param p: the likelihood of immediately revisiting a node in the walk
        :param q: search to differentiate between “inward” and “outward” nodes in the walk
        :param dimensions: Dimension of embedding vectors
        :param window_size: Maximum distance between the current and predicted node in the network
        :param base_iter: Number of iterations (epochs) over the walks
        :param workers: Number of CPU cores that will be used in training
        """
        self.original_network = G
        self.lambd = lambd
        
        splitter = EgoNetSplitter(self.original_network, self.lambd)
        self.persona_network = splitter.persona_network
        self.social_network = splitter.social_network # to check how method works, it can be deleted afterwards.
        self.node_to_persona = splitter.personalities
        self.persona_to_node = splitter.personality_map
        del splitter
        
        super().__init__(self.persona_network,
                         directed=directed,
                         num_walks=num_walks,
                         walk_length=walk_length,
                         p=p,
                         q=q,
                         dimensions=dimensions,
                         window_size=window_size,
                         base_iter=base_iter,
                         workers=workers)
        
        
    def save_persona_network(self, file_name):
        nx.write_edgelist(self.persona_network, file_name)
            
            
    def save_persona_to_node_mapping(self, file_name):
        with open(file_name, 'wb') as f:
            pickle.dump(self.persona_to_node, f)
        
        
    def save_node_to_persona_mapping(self, file_name):
        with open(file_name, 'wb') as f:
            pickle.dump(self.node_to_persona, f)
                    
