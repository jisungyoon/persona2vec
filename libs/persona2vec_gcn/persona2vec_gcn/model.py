# %%
import json
import logging
import numpy as np
import networkx as nx
from persona2vec_gcn.ego_splitting import EgoNetSplitter
from persona2vec_gcn.vgae import DeepVGAE

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


class Persona2VecGCN(object):
    """
    Persona2Vec graph neural network object
    """

    def __init__(
        self,
        G,
        X,
        lambd=0.5,
        clustering_method="connected_component",
        directed=False,
        lr=0.01,
        hidden_dimensions=256,
        dimensions=128,
        epoch_base=10,
        epoch_persona=10,
        val_size=0.05,
        test_size=0.1
    ):
        """
        :param G: NetworkX graph object. persona graah
        :param X: node features. numpy array
        :param lambd: edge weight for persona edge, usually 0 ~ 1
        :param clustering_method: name of the clustering method that uses in splitting personas, choose one of these ('connected_component''modulairty','label_prop')
        :param directed: Directed network(True) or undirected network(False)
        :param hidden_dimensions: Hidden dimension of encoder
        :param dimensions: Dimension of node embedding
        :param lr: Learning rate
        :param epoch_base: Number of iterations (epochs) over the base model for the base embedding - the first round
        :param epoch_persona: Number of iterations (epochs) over the persona model for the persona embedding - the second round
        :param val_size: Fraction of the data to be used as validation data
        :param test_size: Fraction of the data to be used as test data
        """
        self.original_network = G
        self.X = X
        self.lambd = lambd
        self.clustering_method = clustering_method
        self.directed = directed

        self.lr = lr
        self.epoch_base = epoch_base
        self.epoch_persona = epoch_persona
        self.hidden_dimensions = hidden_dimensions
        self.dimensions = dimensions
        self.val_size = val_size
        self.test_size = test_size

        self.get_base_embedding()
        self.generate_persona_network()
        self.get_persona_embedding()

    def get_base_embedding(self):
        """
        Get the base embeddings from the original network
        """
        pass

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
        toNodeID={node:node_id for node_id, node in enumerate(self.G.nodes())}
        node_ids=np.array([toNodeID[self.persona_to_node[persona]] for persona in self.persona_network.nodes()])
        X_persona = self.X[node_ids, :]

        self.persona_model = DeepVGAE(
            G=self.persona_network,
            X=X_persona,
            num_features=None,
            directed=True,
            hidden_dimensions=self.hidden_dimensions,
            dimensions=self.dimensions,
            lr=self.lr,
            val_size=self.val_size,
            test_size=self.test_size,
            epochs=self.epoch_persona
        )
        self.embedding = self.persona_model.learn_embedding()

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

# %%
