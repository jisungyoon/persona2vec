import json
import networkx as nx

from persona2vec.node2vec import Node2Vec
from persona2vec.ego_splitting import EgoNetSplitter

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


class Persona2Vec(object):
    """
    Persona2Vec node embedding object
    """

    def __init__(
        self,
        G,
        lambd=0.1,
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

        # Get the base embeddings from the original network
        self.base_model = Node2Vec(
            self.original_network,
            directed=directed,
            num_walks=num_walks_base,
            walk_length=walk_length_base,
            p=p,
            q=q,
            dimensions=dimensions,
            window_size=window_size_base,
            epoch=epoch_base,
            workers=workers,
        )
        self.base_model.simulate_walks()
        self.base_embedding = self.base_model.learn_embedding()

        # Generate persona network
        splitter = EgoNetSplitter(
            self.original_network, directed=directed, lambd=self.lambd
        )
        self.persona_network = splitter.persona_network
        self.node_to_persona = splitter.personalities
        self.persona_to_node = splitter.personality_map
        del splitter

        # Get persona embedding with base embedding
        self.persona_model = Node2Vec(
            self.persona_network,
            directed=True,
            num_walks=num_walks_persona,
            walk_length=walk_length_persona,
            p=p,
            q=q,
            dimensions=dimensions,
            window_size=window_size_persona,
            epoch=epoch_persona,
            workers=workers,
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
