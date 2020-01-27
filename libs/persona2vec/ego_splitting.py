import networkx as nx
from tqdm import tqdm
from itertools import permutations
import networkx.algorithms.community.modularity_max as modularity
import networkx.algorithms.community.label_propagation as label_prop
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


class EgoNetSplitter(object):
    """
    A implementation of network ego splitting procedure.
    For details, please check the "Persona2Vec" paper.
    """

    def __init__(
        self,
        network,
        directed=False,
        lambd=0.1,
        clustering_method="connected_component",
    ):
        """
        :param network: Networkx object.
        :param directed: Directed network(True) or undirected network(False)
        :param lambd: weight of persona edges
        :param clustering_method: name of the clustering method that uses in splitting personas.
        """
        self.network = network
        self.directed = directed
        self.lambd = lambd

        # clustering algorithms
        if clustering_method == "connected_component":
            if self.directed:
                self.ego_clustering_method = nx.weakly_connected_components
            else:
                self.ego_clustering_method = nx.connected_components
        elif clustering_method == "modularity":
            self.ego_clustering_method = modularity.greedy_modularity_communities
        elif clustering_method == "label_prop":
            self.ego_clustering_method = label_prop.label_propagation_communities
        else:
            logging.error("Not implemented for this clustering method")
            return

        # intialize unweighted edges with 1
        if not nx.is_weighted(self.network):
            for edge in self.network.edges():
                self.network[edge[0]][edge[1]]["weight"] = 1

        self.create_egonets()
        self.map_node_to_persona()
        self.create_persona_network()

    def create_egonet(self, node):
        """
        Creating an ego net, extracting personas and partitioning it.
        :param node: Node ID for egonet (ego node).
        """
        neighbor_set = set(nx.all_neighbors(self.network, node)) - set([node])
        ego_net_minus_ego = self.network.subgraph(neighbor_set)

        try:
            components = {
                i: nodes
                for i, nodes in enumerate(self.ego_clustering_method(ego_net_minus_ego))
            }
        except ZeroDivisionError as error:
            components = {i: [node] for i, node in enumerate(ego_net_minus_ego.nodes)}

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

        # Add original edges
        self.original_edges = [
            (
                self.components[edge[0]][edge[1]],
                self.components[edge[1]][edge[0]],
                edge[2]["weight"],
            )
            for edge in tqdm(self.network.edges(data=True))
            if edge[0] != edge[1]
        ]
        if not self.directed:
            self.original_edges += [
                (
                    self.components[edge[1]][edge[0]],
                    self.components[edge[0]][edge[1]],
                    edge[2]["weight"],
                )
                for edge in tqdm(self.network.edges(data=True))
                if edge[0] != edge[1]
            ]

        G = nx.DiGraph()
        G.add_weighted_edges_from(self.original_edges)

        #  Add persona edges
        degree_dict = dict(G.out_degree())
        degree_dict = {k: 1 if v == 0 else v for k, v in degree_dict.items()}
        self.persona_edges = [
            (x, y, self.lambd * (degree_dict[x]))
            for node, personas in self.node_to_persona.items()
            if len(personas) > 1
            for x, y in permutations(personas, 2)
        ]

        G.add_weighted_edges_from(self.persona_edges)
        self.persona_network = G
