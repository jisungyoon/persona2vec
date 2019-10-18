import sys
import logging
import pickle

from persona2vec.model import Persona2Vec
from persona2vec.link_prediction import linkPredictionTask
from persona2vec.utils import read_graph

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


def do_link_prediction(NETWORK_FILE,
                       TEST_EDGE_FILE,
                       NEGATIVE_EDGE_FILE,
                       OUT_FILE,
                       LAMBDA,
                       DIM,
                       NUMBER_OF_CORES):
    G = read_graph(NETWORK_FILE)
    model = Persona2Vec(
        G, lambd=LAMBDA, dimensions=DIM, workers=NUMBER_OF_CORES)
    model.simulate_walks()
    emb = model.learn_embedding()

    test_edges = pickle.load(open(TEST_EDGE_FILE, 'rb'))
    negative_edges = pickle.load(open(NEGATIVE_EDGE_FILE, 'rb'))

    name = '\t'.join([NETWORK_FILE, str(LAMBDA), str(DIM)])
    test = linkPredictionTask(
        G, test_edges, negative_edges, emb, name=name, persona=True,
        node_to_persona=model.node_to_persona)
    test.do_link_prediction()
    test.write_result(OUT_FILE)


if __name__ == "__main__":
    NETWORK_FILE = sys.argv[1]
    TEST_EDGE_FILE = sys.argv[2]
    NEGATIVE_EDGE_FILE = sys.argv[3]
    OUT_FILE = sys.argv[4]
    LAMBDA = float(sys.argv[5])
    DIM = int(sys.argv[6])
    NUMBER_OF_CORES = int(sys.argv[7])
    REPETITION = int(sys.argv[8])
    
    print(REPETITION)
    
    for i in range(REPETITION):
        do_link_prediction(NETWORK_FILE, TEST_EDGE_FILE, NEGATIVE_EDGE_FILE, OUT_FILE, LAMBDA, DIM, NUMBER_OF_CORES)
