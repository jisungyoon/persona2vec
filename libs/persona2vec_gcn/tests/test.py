import unittest

from persona2vec_gcn.model import Persona2VecGCN
from persona2vec_gcn.utils import read_graph
import numpy as np
import networkx as nx

class TestPersona2VecGCN(unittest.TestCase):
    def setUp(self):
        self.G = nx.karate_club_graph()
        self.A = nx.adjacency_matrix(self.G)

    def test_embedding(self):
        dimensions = 64
        model = Persona2VecGCN(self.G, X = self.A.toarray(), lambd=0.1, dimensions = dimensions)
        emb = np.vstack(model.embedding.values())
        assert emb.shape[1] == dimensions
        assert emb.shape[0] >= self.A.shape[0]

    def test_splitting_algorithm(self):
        dimensions = 64
        model = Persona2VecGCN(self.G, X = self.A.toarray(), lambd=0.1, dimensions = dimensions, clustering_method="modularity")
        model = Persona2VecGCN(self.G, X = self.A.toarray(), lambd=0.1, dimensions = dimensions, clustering_method="label_prop")

if __name__ == "__main__":
    unittest.main()
