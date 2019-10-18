import sys
import logging

from persona2vec.network_train_test_splitter import networkTrainTestSplitter
from persona2vec.utils import read_graph

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


def train_test_set_split(IN_FILE, INDEX):
    G = read_graph(IN_FILE)
    OUTPUT_PATH = IN_FILE.split('.')[0] + '_' + str(INDEX)
    splitter = networkTrainTestSplitter(G)
    splitter.train_test_split()
    splitter.generate_negative_edges()
    splitter.save_splitted_result(OUTPUT_PATH)


if __name__ == "__main__":
    IN_FILE = sys.argv[1]
    INDEX = sys.argv[2]

    train_test_set_split(IN_FILE, INDEX)
