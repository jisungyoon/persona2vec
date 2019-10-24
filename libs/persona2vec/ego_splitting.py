import networkx as nx
from tqdm import tqdm
from itertools import combinations, permutations
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


class EgoNetSplitter(object):
    """
    A revised implementation of "Ego-Splitting Framework: from Non-Overlapping
    to Overlapping Clusters" for Persona2Vec.
    Paper: https://www.eecs.yorku.ca/course_archive/2017-18/F/6412/reading/kdd17p145.pdf
    Video: https://www.youtube.com/watch?v=xMGZo-F_jss
    Slides: https://epasto.org/papers/kdd2017-Slides.pdf
    """

    def __init__(self, network, directed=False, lambd=0.1):
        """
        :param network: Networkx object.
        :param directed: Directed network(True) or undirected network(False)
        :param lambd: weight of persona edges
        """
        self.network = network
        self.directed = directed
        self.lambd = lambd
        self.create_egonets()
        self.map_personalities()
        self.create_persona_network()

    def create_egonet(self, node):
        """
        Creating an ego net, extracting personas and partitioning it.
        :param node: Node ID for egonet (ego node).
        """
        if self.directed:
            ego_net_minus_ego = self.network.subgraph(
                nx.all_neighbors(self.network, node))
            components = {i: nodes for i, nodes in enumerate(
                nx.weakly_connected_components(ego_net_minus_ego))}
        else:
            ego_net_minus_ego = self.network.subgraph(
                self.network.neighbors(node))
            components = {i: nodes for i, nodes in enumerate(
                nx.connected_components(ego_net_minus_ego))}
        new_mapping = {}
        personalities = []
        for i, (k, v) in enumerate(components.items()):
            name = node + '-' + str(i + 1)
            personalities.append(name)
            for other_node in v:
                new_mapping[other_node] = name
        self.components[node] = new_mapping
        self.personalities[node] = personalities

    def create_egonets(self):
        """
        Creating an egonet for each node.
        """
        self.components = {}
        self.personalities = {}
        logging.info("Creating egonets.")
        for node in tqdm(self.network.nodes()):
            self.create_egonet(node)

    def map_personalities(self):
        """
        Mapping the personas to new nodes.
        """
        self.personality_map = {persona: node for node in self.network.nodes(
        ) for persona in self.personalities[node]}

    def create_persona_network(self):
        """
        Create a persona network using the egonet components.
        """
        logging.info("Creating the persona network.")

        # Add social edges
        self.social_edges = [(self.components[edge[0]][edge[1]],
                              self.components[edge[1]][edge[0]])
                             for edge in tqdm(self.network.edges())]
        if self.directed:
            G = nx.from_edgelist(self.social_edges, create_using=nx.DiGraph())
        else:
            G = nx.from_edgelist(self.social_edges)
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1

        #  Add persona edges
        if self.directed:
            self.persona_edges = [(x, y, self.lambd)
                                  for node, personas
                                  in self.personalities.items()
                                  if len(personas) > 1
                                  for x, y in permutations(personas, 2)]
        else:
            self.persona_edges = [(x, y, self.lambd)
                                  for node, personas,
                                  in self.personalities.items()
                                  if len(personas) > 1
                                  for x, y in combinations(personas, 2)]
        G.add_weighted_edges_from(self.persona_edges)
        self.persona_network = G
