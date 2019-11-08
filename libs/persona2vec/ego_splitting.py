import networkx as nx
from tqdm import tqdm
from itertools import permutations
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


class EgoNetSplitter(object):
    """
    A implementation of network ego splitting procedure.
    For details, please check the "Persona2Vec" paper.
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
        self.map_node_to_persona()
        self.create_persona_network()

    def create_egonet(self, node):
        """
        Creating an ego net, extracting personas and partitioning it.
        :param node: Node ID for egonet (ego node).
        """
        if self.directed:
            ego_net_minus_ego = self.network.subgraph(
                nx.all_neighbors(self.network, node)
            )
            components = {
                i: nodes 
                for i, nodes in enumerate(
                nx.weakly_connected_components(ego_net_minus_ego)
                )
            }
        else:
            ego_net_minus_ego = self.network.subgraph(
                self.network.neighbors(node))
            components = {
                i: nodes 
                for i, nodes in enumerate(nx.connected_components(ego_net_minus_ego))
            }
        new_mapping = {}
        node_to_persona = []
        for i, (k, v) in enumerate(components.items()):
            name = "{}-{}".format(node, i + 1)
            node_to_persona.append(name)
            for other_node in v:
                new_mapping[other_node] = name
        self.components[node] = new_mapping
        self.node_to_persona[node] = node_to_persona

    def create_egonets(self):
        """
        Creating an egonet for each node.
        """
        self.components = {}
        self.node_to_persona = {}
        logging.info("Creating egonets.")
        for node in tqdm(self.network.nodes()):
            self.create_egonet(node)

    def map_node_to_persona(self):
        """
        Mapping the personas to new nodes.
        """
        self.persona_to_node = {
            persona: node 
            for node in self.network.nodes()
            for persona in self.node_to_persona[node]
        }

    def create_persona_network(self):
        """
        Create a persona network using the egonet components.
        """
        logging.info("Creating the persona network.")

        # Add social edges
        self.real_edges = [
            (self.components[edge[0]][edge[1]],
             self.components[edge[1]][edge[0]])
            for edge in tqdm(self.network.edges())
            if edge[0] != edge[1]
        ]
        if not self.directed:
            self.real_edges += [
                (self.components[edge[1]][edge[0]],
                 self.components[edge[0]][edge[1]])
                for edge in tqdm(self.network.edges())
                if edge[0] != edge[1]
            ]
            
        G = nx.from_edgelist(self.real_edges, create_using=nx.DiGraph())
        for edge in G.edges():
            G[edge[0]][edge[1]]["weight"] = 1

        #  Add persona edges
        degree_dict = dict(G.out_degree())
        self.persona_edges = [
            (x, y, self.lambd * (degree_dict[x]))
            for node, personas in self.node_to_persona.items()
            if len(personas) > 1
            for x, y in permutations(personas, 2)
        ]
        
        G.add_weighted_edges_from(self.persona_edges)
        self.persona_network = G
