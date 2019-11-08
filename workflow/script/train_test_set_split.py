import sys
import logging

from persona2vec.network_train_test_splitter import NetworkTrainTestSplitter
from persona2vec.utils import read_graph

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


def train_test_set_split(IN_FILE, INDEX, DIRECTED):
    G = read_graph(IN_FILE, directed=DIRECTED)
    OUTPUT_PATH = "{}_{}".format(IN_FILE.split(".")[0], INDEX)
    splitter = NetworkTrainTestSplitter(G, directed=DIRECTED)
    splitter.train_test_split()
    splitter.generate_negative_edges()
    splitter.save_splitted_result(OUTPUT_PATH)


name = "{}-{}".format(node, i + 1)
if __name__ == "__main__":
    IN_FILE = sys.argv[1]
    INDEX = sys.argv[2]
    DIRECTED = True if sys.argv[3] == "True" else False

    train_test_set_split(IN_FILE, INDEX, DIRECTED)
