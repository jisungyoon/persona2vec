import sys
import logging
import csv
import pickle
import networkx as nx

from persona2vec.utils import read_graph, read_edge_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


def convert_network_files_for_splitter(NETWORK_FILE, CONVERTED_NETWORK_FILE, DIRECTED):
    G = read_graph(NETWORK_FILE, directed=DIRECTED)
    G_ = nx.convert_node_labels_to_integers(G, label_attribute='old_label')
    nx.write_edgelist(G_, CONVERTED_NETWORK_FILE)
    return G_

def get_old_label_translator(G, TRANSLATOR_FILE):
    translator = {G.nodes[node]['old_label'] : str(node) for node in G.nodes}
    pickle.dump(translator, open(TRANSLATOR_FILE, 'wb'))
    return translator


def convert_edges(EDGE_FILE, TRANSLATOR, OUT_FILE):
    edge_list = read_edge_file(EDGE_FILE)
    converted_edge_list = [(TRANSLATOR[src], TRANSLATOR[tag]) for src, tag in edge_list]
    with open(OUT_FILE, 'wt') as f:
        tsv_writer = csv.writer(f, delimiter='\t')
        tsv_writer.writerows(converted_edge_list)


if __name__ == "__main__":
    NETWORK_FILE = sys.argv[1]
    TEST_EDGE_FILE = sys.argv[2]
    NEGATIVE_EDGE_FILE = sys.argv[3]
    CONVERTED_NETWORK_FILE = sys.argv[4]
    CONVERTED_TEST_EDGE_FILE = sys.argv[5]
    CONVERTED_NEGATIVE_EDGE_FILE = sys.argv[6]
    TRANSLATOR_FILE = sys.argv[7]
    DIRECTED = True if sys.argv[8] == "True" else False
    
    converted_G = convert_network_files_for_splitter(NETWORK_FILE, CONVERTED_NETWORK_FILE, DIRECTED)
    translator = get_old_label_translator(converted_G, TRANSLATOR_FILE)
    convert_edges(TEST_EDGE_FILE, translator, CONVERTED_TEST_EDGE_FILE)
    convert_edges(NEGATIVE_EDGE_FILE, translator, CONVERTED_NEGATIVE_EDGE_FILE)

