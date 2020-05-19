import json
import logging

import networkx as nx
from persona2vec.ego_splitting import EgoNetSplitter
from persona2vec.node2vec import Node2Vec

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


class Persona2Vec(object):
    """
    Persona2Vec node embedding object
    """

    def __init__(
        self,
        G,
        lambd=0.1,
        clustering_method="connected_component",
        directed=False,
        num_walks_base=10,
        walk_length_base=40,
        window_size_base=5,
        num_walks_persona=5,
        walk_length_persona=80,
        window_size_persona=2,
        p=1.0,
        q=1.0,
        dimensions=128,
        epoch_base=1,
        epoch_persona=1,
        workers=4,
    ):
        """
        :param G: NetworkX graph object. persona graah
        :param lambd: edge weight for persona edge, usually 0 ~ 1
        :param clustering_method: name of the clustering method that uses in splitting personas, choose one of these ('connected_component''modulairty','label_prop')
        :param directed: Directed network(True) or undirected network(False)
        :param num_walks_base: Number of random walker per node for the base embedding - the first round
        :param walk_length_base: Length(number of nodes) of random walker for the base embedding - the first round
        :param window_size_base: Maximum distance between the current and predicted node in the network for the base embedding - the first round
        :param num_walks_persona: Number of random walker per node for the persona embedding - the second round
        :param walk_length_persona: Length(number of nodes) of random walker for the persona embedding - the second round 
        :param window_size_persona: Maximum distance between the current and predicted node in the network for the persona embedding - the second round
        :param p: the likelihood of immediately revisiting a node in the walk
        :param q: search to differentiate between “inward” and “outward” nodes in the walk
        :param dimensions: Dimension of embedding vectors
        :param epoch_base: Number of iterations (epochs) over the walks for the base embedding - the first round
        :param epoch_persona: Number of iterations (epochs) over the walks for the persona embedding - the second round
        :param workers: Number of CPU cores that will be used in training
        """
        self.original_network = G
        self.lambd = lambd
        self.clustering_method = clustering_method
        self.directed = directed

        self.num_walks_base = num_walks_base
        self.walk_length_base = walk_length_base
        self.window_size_base = window_size_base
        self.epoch_base = epoch_base

        self.num_walks_persona = num_walks_persona
        self.walk_length_persona = walk_length_persona
        self.window_size_persona = window_size_persona
        self.epoch_persona = epoch_persona

        self.p = p
        self.q = q
        self.dimensions = dimensions
        self.workers = workers

        self.get_base_embedding()
        self.generate_persona_network()
        self.get_persona_embedding()

    def get_base_embedding(self):
        """
        Get the base embeddings from the original network
        """
        self.base_model = Node2Vec(
            self.original_network,
            directed=self.directed,
            num_walks=self.num_walks_base,
            walk_length=self.walk_length_base,
            p=self.p,
            q=self.q,
            dimensions=self.dimensions,
            window_size=self.window_size_base,
            epoch=self.epoch_base,
            workers=self.workers,
        )
        self.base_model.simulate_walks()
        self.base_embedding = self.base_model.learn_embedding()

    def generate_persona_network(self):
        """
        Generate persona network with the given lambda
        """
        splitter = EgoNetSplitter(
            self.original_network,
            directed=self.directed,
            lambd=self.lambd,
            clustering_method=self.clustering_method,
        )
        self.persona_network = splitter.persona_network
        self.node_to_persona = splitter.node_to_persona
        self.persona_to_node = splitter.persona_to_node

    def get_persona_embedding(self):
        """
        Get the persona embeddings from the persona network starts from base embeding
        """
        self.persona_model = Node2Vec(
            self.persona_network,
            directed=True,
            num_walks=self.num_walks_persona,
            walk_length=self.walk_length_persona,
            p=self.p,
            q=self.q,
            dimensions=self.dimensions,
            window_size=self.window_size_persona,
            epoch=self.epoch_persona,
            workers=self.workers,
        )
        self.persona_model.simulate_walks()
        self.persona_model.initialize_persona_vectors(
            self.base_embedding, self.persona_to_node
        )
        self.embedding = self.persona_model.learn_embedding_one_epoch()

    def save_persona_network(self, file_path):
        """
        :param file_path: file_path for persona network
        :return:
        """
        nx.write_edgelist(self.persona_network, file_path)

    def save_persona_to_node_mapping(self, file_path):
        """
        :param file_path: file_path for persona to node mapper
        :return:
        """
        with open(file_path, "w") as f:
            json.dump(self.persona_to_node, f)

    def save_node_to_persona_mapping(self, file_path):
        """
        :param file_path: file_path for node to persona mapper
        :return:
        """
        with open(file_path, "w") as f:
            json.dump(self.node_to_persona, f)

    def save_base_embedding(self, file_path):
        """
        :param file_path: file_path for node to persona mapper
        :return:
        """
        self.base_model.save_embedding(file_path)

    def save_persona_embedding(self, file_path):
        """
        :param file_path: file_path for node to persona mapper
        :return:
        """
        self.persona_model.save_embedding(file_path)
