import sys
import os 
import logging

from multiprocessing import Process

from persona2vec.link_prediction import linkPredictionTask
from persona2vec.utils import read_graph

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

def run_parallel(function, args, number_of_cores):
    procs = [Process(target= function, args = [proc_num] + args) for proc_num in range(number_of_cores)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
        

def do_link_prediction(proc_num, IN_FILE, LAMBD, DIM):
    G = read_graph(IN_FILE)
    logging.info('Core-' + str(proc_num)  + ' start')
    test_object = linkPredictionTask(G, IN_FILE, lambd=LAMBD, dimensions=DIM, workers=10)
    test_object.train_test_split()
    test_object.generate_negative_edges()
    test_object.learn_persona2vec_emb()
    test_object.calculate_link_prediction_score()
    test_object.calculate_ROC_AUC_value()
    test_object.print_result()
    test_object.write_result(OUT_FILE)
    

    
if __name__ == "__main__":
    IN_FILE = sys.argv[1]
    OUT_FILE = sys.argv[2]
    LAMBD = float(sys.argv[3])
    DIM = int(sys.argv[4])
    REPETITION = int(sys.argv[5])
    
    run_parallel(do_link_prediction, [IN_FILE, LAMBD, DIM], REPETITION)

        
        
    
    