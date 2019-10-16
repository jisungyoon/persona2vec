import networkx as nx
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from persona2vec.utils import read_graph
from persona2vec.model import Persona2Vec

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


class linkPredictionTask(object):
    def __init__(self, G,
                 test_edges,
                 negative_edges,
                 name,
                 lambd=0.1,
                 num_walks=10,
                 walk_length=40,
                 dimensions=16,
                 window_size=5,
                 workers=4,
                 base_iter=5):
             
        self.G = G
        self.test_edges = test_edges
        self.negative_edges = negative_edges       
        self.name = name
        
        self.lambd = lambd
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.dimensions = dimensions
        self.window_size = window_size
        self.workers = workers
        self.base_iter = base_iter

    def learn_persona2vec_emb(self):
        logging.info('Initiate persona2vec')
        self.model = Persona2Vec(self.G,
                                 lambd=self.lambd,
                                 num_walks=self.num_walks,
                                 walk_length=self.walk_length,
                                 dimensions=self.dimensions,
                                 window_size=self.window_size,
                                 workers=self.workers,
                                 base_iter=self.base_iter)
        self.model.simulate_walks()
        self.emb = self.model.learn_embedding()

    def calculate_link_prediction_score(self):
        self.link_prediction_score_postive = np.array(
            self.calculate_score(self.test_edges))
        self.link_prediction_score_negative = np.array(
            self.calculate_score(self.negative_edges))

    def calculate_score(self, edge_list):
        score_list = []
        for src, tag in edge_list:
            src_personas = self.model.node_to_persona[src]
            tag_personas = self.model.node_to_persona[tag]
            max_sim = max([np.dot(self.emb[src_persona], self.emb[tag_persona])
                           for src_persona in src_personas for tag_persona in tag_personas])
            score_list.append(max_sim)
        return score_list

    def calculate_ROC_AUC_value(self):
        logging.info('Calcualte ROC_AUC values')
        y_true = np.concatenate([np.ones_like(self.link_prediction_score_postive), np.zeros_like(
            self.link_prediction_score_negative)])
        y_score = np.concatenate(
            [self.link_prediction_score_postive, self.link_prediction_score_negative], axis=0)
        self.ROC_AUC_value = roc_auc_score(y_true, y_score)

    def print_result(self):
        logging.info(self.name)
        logging.info(self.lambd)
        logging.info(self.dimensions)
        logging.info(self.ROC_AUC_value)

    def write_result(self, file_name):
        f = open(file_name, 'a')
        f.write('\t'.join([self.name, str(self.lambd), str(
            self.dimensions), str(self.ROC_AUC_value)]) + '\n')
        f.close()
