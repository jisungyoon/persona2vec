import sys
import logging

from multiprocessing import Process
import pickle

from common_function import run_parallel
from persona2vec.model import Persona2Vec
from persona2vec.link_prediction import linkPredictionTask
from persona2vec.utils import read_graph

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


def do_link_prediction(proc_num, NETWORK_FILE, TEST_EDGE_FILE, NEGATIVE_EDGE_FILE, OUT_FILE, LAMBD, DIM):
    logging.info('Core-' + str(proc_num) + ' start')
    G = read_graph(NETWORK_FILE)
    model = Persona2Vec(
        G, lambd=LAMBD, dimensions=DIM, workers=4)
    model.simulate_walks()
    emb = model.learn_embedding()
    
    test_edges = pickle.load(open(TEST_EDGE_FILE, 'rb'))
    negative_edges = pickle.load(open(NEGATIVE_EDGE_FILE, 'rb'))
    
    name = '\t'.join([NETWORK_FILE, str(LAMBD), str(DIM)])
    test = linkPredictionTask(
        G, test_edges, negative_edges, emb, name=name, persona=True, node_to_persona=model.node_to_persona)
    test.do_link_prediction()
    test.write_result(OUT_FILE)
    
    
    



if __name__ == "__main__":
    NETWORK_FILE = sys.argv[1]
    TEST_EDGE_FILE = sys.argv[2]
    NEGATIVE_EDGE_FILE = sys.argv[3]
    OUT_FILE = sys.argv[4]
    LAMBD = float(sys.argv[5])
    DIM = int(sys.argv[6])
    REPETITION = int(sys.argv[7])
    
    run_parallel(do_link_prediction, [NETWORK_FILE, TEST_EDGE_FILE, NEGATIVE_EDGE_FILE, OUT_FILE, LAMBD, DIM], REPETITION)
