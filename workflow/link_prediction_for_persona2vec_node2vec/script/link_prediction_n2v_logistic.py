import sys
import logging
import csv

from persona2vec import node2vec
import numpy as np
from sklearn.linear_model import LogisticRegression
from persona2vec.utils import read_graph, read_edge_file
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


def do_link_prediction(
    NETWORK_FILE,
    TEST_EDGE_FILE,
    NEGATIVE_EDGE_FILE,
    DIRECTED,
    OUT_FILE,
    DIM,
    NUMBER_OF_CORES,
):
    G = read_graph(NETWORK_FILE, directed=DIRECTED)
    model = node2vec.Node2Vec(
        G, directed=DIRECTED, dimensions=DIM, workers=NUMBER_OF_CORES
    )
    model.simulate_walks()
    emb = model.learn_embedding()

    test_edges = read_edge_file(TEST_EDGE_FILE)
    negative_edges = read_edge_file(NEGATIVE_EDGE_FILE)

    test_hardmord_vectors = [np.multiply(emb[src], emb[tag]) for src, tag in test_edges]
    negative_hardmord_vectors = [np.multiply(emb[src], emb[tag]) for src, tag in negative_edges]
    xs = np.concatenate([test_hardmord_vectors, negative_hardmord_vectors])
    ys = np.concatenate([np.ones(len(test_hardmord_vectors)), np.zeros(len(negative_hardmord_vectors))])
    
    clf = LogisticRegression(random_state=0).fit(xs, ys)
    predicted_ys = clf.predict(xs)
    ROC_AUC_value = roc_auc_score(ys, predicted_ys)
    
    name = "\t".join([NETWORK_FILE, str(DIM)])
    f = open(OUT_FILE, "a")
    f.write("{}\t{}\n".format(*[name, str(ROC_AUC_value)]))
    f.close()


if __name__ == "__main__":
    NETWORK_FILE = sys.argv[1]
    TEST_EDGE_FILE = sys.argv[2]
    NEGATIVE_EDGE_FILE = sys.argv[3]
    DIRECTED = True if sys.argv[4] == "True" else False
    OUT_FILE = sys.argv[5]
    DIM = int(sys.argv[6])
    NUMBER_OF_CORES = int(sys.argv[7])
    REPETITION = int(sys.argv[8])

    for i in range(REPETITION):
        do_link_prediction(
            NETWORK_FILE,
            TEST_EDGE_FILE,
            NEGATIVE_EDGE_FILE,
            DIRECTED,
            OUT_FILE,
            DIM,
            NUMBER_OF_CORES,
        )