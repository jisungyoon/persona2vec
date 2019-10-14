import sys
import os 
import logging

from persona2vec.link_prediction import linkPredictionTask
from persona2vec.utils import read_graph

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    
if __name__ == "__main__":
    IN_FILE = sys.argv[1]
    OUT_FILE = sys.argv[2]
    LAMBD = float(sys.argv[3])
    DIM = int(sys.argv[4])
    REPETITION = int(sys.argv[5])
    
    print(IN_FILE, OUT_FILE, LAMBD, DIM, REPETITION)
    
    
#     for i in range(REPETITION):
#         G = read_graph(IN_FILE)
#         test_object = linkPredictionTask(G, IN_FILE, lambd=LAMBD, dimensions=DIM, workers=10)
#         test_object.train_test_split()
#         test_object.generate_negative_edges()
#         test_object.learn_persona2vec_emb()
#         test_object.calculate_link_prediction_score()
#         test_object.calculate_ROC_AUC_value()
#         test_object.print_result()
#         test_object.write_result(OUT_FILE)
