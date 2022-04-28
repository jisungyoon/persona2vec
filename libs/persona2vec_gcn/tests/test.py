import unittest

import persona2vec_gcn as pvgcn
from persona2vec_gcn.model import Persona2VecGCN
from persona2vec_gcn.vgae import DeepVGAE
import numpy as np
import networkx as nx


class TestPersona2VecGCN(unittest.TestCase):
    def setUp(self):
        self.G = nx.karate_club_graph()
        self.A = nx.adjacency_matrix(self.G)

    def test_embedding(self):
        dimensions = 64
        X = pvgcn.laplacian_eigenmap(self.G, dimensions=16)
        model = Persona2VecGCN(self.G, X=X, lambd=0.1, dimensions=dimensions)
        emb = np.vstack(model.embedding.values())
        assert emb.shape[1] == dimensions
        assert emb.shape[0] >= self.A.shape[0]

    def test_splitting_algorithm(self):
        dimensions = 64
        X = pvgcn.laplacian_eigenmap(self.G, dimensions=16)
        model = Persona2VecGCN(
            self.G,
            X=X,
            lambd=0.1,
            dimensions=dimensions,
            clustering_method="modularity",
        )
        model = Persona2VecGCN(
            self.G,
            X=X,
            lambd=0.1,
            dimensions=dimensions,
            clustering_method="label_prop",
        )

    def test_deepvgae(self):
        dimensions = 64
        X = pvgcn.laplacian_eigenmap(self.G, dimensions=16)
        model = DeepVGAE(self.G, X=X, dimensions=dimensions)
        model.learn_embedding()
        emb = np.vstack(model.embedding.values())
        assert emb.shape[1] == dimensions

    def test_laplacian_eigenmap(self):
        dimensions = 5
        emb = pvgcn.laplacian_eigenmap(self.G, dimensions=dimensions)
        assert emb.shape[1] == dimensions

    def test_fastRP(self):
        dimensions = 5
        emb = pvgcn.fastRP(self.G, dimensions=dimensions)
        assert emb.shape[1] == dimensions


if __name__ == "__main__":
    unittest.main()
