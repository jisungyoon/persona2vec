import random
import itertools
import logging

from multiprocessing import Pool
from tqdm import tqdm
from collections import Counter
from itertools import chain
from gensim.models import Word2Vec

from persona2vec.utils import alias_setup, alias_draw


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


class Node2Vec(object):
    """
    Node2Vec node embedding object
    This code is from https://github.com/aditya-grover/node2vec.
    """

    def __init__(
        self,
        G,
        directed=False,
        num_walks=10,
        walk_length=80,
        p=1.0,
        q=1.0,
        dimensions=128,
        window_size=10,
        epoch=1,
        workers=1,
    ):
        """
        :param G: NetworkX graph object.
        :param directed: Directed network(True) or undirected network(False)
        :param num_walks: Number of random walker per node
        :param walk_length: Length(number of nodes) of random walker
        :param p: the likelihood of immediately revisiting a node in the walk
        :param q: search to differentiate between “inward” and “outward” nodes in the walk
        :param dimensions: Dimension of embedding vectors
        :param window_size: Maximum distance between the current and predicted node in the network
        :param epoch: Number of epochs over the walks
        :param workers: Number of CPU cores that will be used in training
        """
        self.G = G
        self.directed = directed

        # parameters for random walker
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.p = p
        self.q = q

        # parameters for learning embeddings
        self.dimensions = dimensions
        self.window_size = window_size
        self.epoch = epoch

        # computing configuration and path
        self.workers = workers
        self.by_pass_mode = True if p == q == 1 else False

        self.walks = []
        self.preprocess_transition_probs()

    def preprocess_transition_probs(self):
        """
        Preprocess transition probabilities for guiding the random walks.
        """
        G = self.G

        logging.info("Calculating transition probability:")
        alias_nodes = {}
        for node in tqdm(G.nodes()):
            unnormalized_probs = [G[node][nbr]["weight"] for nbr in G.neighbors(node)]
            norm_const = sum(unnormalized_probs)
            normalized_probs = [
                float(u_prob) / norm_const for u_prob in unnormalized_probs
            ]
            alias_nodes[node] = alias_setup(normalized_probs)

        self.alias_nodes = alias_nodes

        alias_edges = {}
        if not self.by_pass_mode:
            logging.info(
                "Calculating transition probability for a biased behavior of random walker:"
            )
            if self.directed:
                for edge in tqdm(G.edges()):
                    alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
            else:
                for edge in tqdm(G.edges()):
                    alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
                    alias_edges[(edge[1], edge[0])] = self.get_alias_edge(
                        edge[1], edge[0]
                    )

        self.alias_edges = alias_edges

        return

    def simulate_walks(self):
        """
        Repeatedly simulate random walks from each node
        """
        logging.info("Gerating Walk iteration:")
        with Pool(self.workers) as p:
            temp_walks = list(
                tqdm(
                    p.imap(self.simulate_walk_one_iter, range(self.num_walks)),
                    total=self.num_walks,
                )
            )
        self.walks = list(chain(*temp_walks))

    def simulate_walk_one_iter(self, _):
        nodes = list(self.G.nodes())
        random.shuffle(nodes)
        walks = [
            self.node2vec_walk(walk_length=self.walk_length, start_node=node)
            for node in nodes
        ]

        return walks

    def get_alias_edge(self, src, dst):
        """
        Get the alias edge setup lists for a given edge.
        :param src: Id of source node
        :param dst: Id of target node
        """
        G = self.G
        p = self.p
        q = self.q

        unnormalized_probs = []
        for dst_nbr in sorted(G.neighbors(dst)):
            if dst_nbr == src:
                unnormalized_probs.append(G[dst][dst_nbr]["weight"] / p)
            elif G.has_edge(dst_nbr, src):
                unnormalized_probs.append(G[dst][dst_nbr]["weight"])
            else:
                unnormalized_probs.append(G[dst][dst_nbr]["weight"] / q)
        norm_const = sum(unnormalized_probs)
        normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]

        return alias_setup(normalized_probs)

    def node2vec_walk(self, walk_length, start_node):
        """
        Simulate a random walk starting from start node.
        :param walk_length: Length of random walker
        :param start_node: Starting node of random walker
        :return walk: Generated trajectory of random walker
        """
        G = self.G
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges

        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = sorted(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                if len(walk) == 1 or self.by_pass_mode:
                    walk.append(
                        cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])]
                    )
                else:
                    prev = walk[-2]
                    next = cur_nbrs[
                        alias_draw(
                            alias_edges[(prev, cur)][0], alias_edges[(prev, cur)][1]
                        )
                    ]
                    walk.append(next)
            else:
                break

        return list(map(str, walk))

    def learn_embedding(self):
        """
        Learning an embedding of nodes in the base graph.
        :return self.embedding: Embedding of nodes in the latent space.
        """
        self.model = Word2Vec(
            self.walks,
            size=self.dimensions,
            window=self.window_size,
            min_count=0,
            sg=1,
            workers=self.workers,
            iter=self.epoch,
        )
        self.embedding = {node: self.model.wv[str(node)] for node in self.G.nodes()}

        return self.embedding

    def initialize_persona_vectors(self, base_emb, persona_to_node):
        """
        Initialize perosnas' inital vectors with their base embedding
        :param base_emb: Base embedding from the original network
        :param persona_to_node: Matching dict which connects node and personas.
        """
        self.model = Word2Vec(
            size=self.dimensions,
            window=self.window_size,
            min_count=0,
            sg=1,
            workers=self.workers,
            iter=self.epoch,
        )

        flattened_walks = list(itertools.chain(*self.walks))
        walk_counter = Counter(flattened_walks)
        self.model.build_vocab_from_freq(walk_counter, corpus_count=len(self.walks))
        for index, word in enumerate(self.model.wv.index2word):
            self.model.wv.vectors[index] = base_emb[persona_to_node[word]]

    def learn_embedding_one_epoch(self):
        """
        Train model for one epoch, and get embedding.
        """
        self.model.train(self.walks, total_examples=self.model.corpus_count, epochs=1)
        self.embedding = {node: self.model.wv[str(node)] for node in self.G.nodes()}

        return self.embedding

    def save_embedding(self, file_name):
        """
        :param file_name: name of file_name
        """
        self.model.wv.save_word2vec_format(file_name)
