import sys
import logging
import csv

from persona2vec.model import Persona2Vec
from persona2vec.link_prediction import LinkPredictionTask
from persona2vec.utils import read_graph, read_edge_file

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


def do_link_prediction(NETWORK_FILE,
                       TEST_EDGE_FILE,
                       NEGATIVE_EDGE_FILE,
                       DIRECTED,
                       OUT_FILE,
                       LAMBDA,
                       DIM,
                       NUMBER_OF_CORES):
    G = read_graph(NETWORK_FILE, directed=DIRECTED)
    model = Persona2Vec(
        G, lambd=LAMBDA, directed=DIRECTED, dimensions=DIM, workers=NUMBER_OF_CORES)
    model.simulate_walks()
    emb = model.learn_embedding()

    test_edges = read_edge_file(TEST_EDGE_FILE)
    negative_edges = read_edge_file(NEGATIVE_EDGE_FILE)

    name = '\t'.join([NETWORK_FILE, str(LAMBDA), str(DIM)])
    test = LinkPredictionTask(
        test_edges, negative_edges, emb, name=name, is_persona_emb=True,
        node_to_persona=model.node_to_persona)
    test.do_link_prediction()
    test.write_result(OUT_FILE)

if __name__ == "__main__":
    NETWORK_FILE = sys.argv[1]
    TEST_EDGE_FILE = sys.argv[2]
    NEGATIVE_EDGE_FILE = sys.argv[3]
    DIRECTED = True if sys.argv[4] == 'True' else False
    OUT_FILE = sys.argv[5]
    LAMBDA = float(sys.argv[6])
    DIM = int(sys.argv[7])
    NUMBER_OF_CORES = int(sys.argv[8])
    REPETITION = int(sys.argv[9])
    
    for i in range(REPETITION):
        do_link_prediction(NETWORK_FILE, TEST_EDGE_FILE, NEGATIVE_EDGE_FILE, DIRECTED, OUT_FILE, LAMBDA, DIM, NUMBER_OF_CORES)
