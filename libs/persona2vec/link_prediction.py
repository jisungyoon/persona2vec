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
                 emb,
                 name,
                 persona=False,
                 node_to_persona={}):
             
        self.G = G
        self.test_edges = test_edges
        self.negative_edges = negative_edges   
        self.emb = emb
        self.name = name
        
        # for persona related embedding
        self.persona = persona
        self.node_to_persona = node_to_persona
        self.aggregate_function = max
        
    def do_link_prediction(self):
        self.calculate_link_prediction_score()
        self.calculate_ROC_AUC_value()
        self.print_result()
    
    def calculate_link_prediction_score(self):
        calculate_method = self.calculate_score_persona if self.persona else self.calculate_score
        self.link_prediction_score_postive = np.array(
            calculate_method(self.test_edges))
        self.link_prediction_score_negative = np.array(
            calculate_method(self.negative_edges))

    def calculate_score_persona(self, edge_list):
        score_list = []
        for src, tag in edge_list:
            src_personas = self.node_to_persona[src]
            tag_personas = self.node_to_persona[tag]
            max_sim = self.aggregate_function([np.dot(self.emb[src_persona], self.emb[tag_persona])
                           for src_persona in src_personas for tag_persona in tag_personas])
            score_list.append(max_sim)
        return score_list
    
    def calculate_score(self, edge_list):
        score_list = [np.dot(self.emb[source], self.emb[target]) for source, target in edge_list]
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
        logging.info(self.ROC_AUC_value)

    def write_result(self, file_name):
        f = open(file_name, 'a')
        f.write('\t'.join([self.name, str(self.ROC_AUC_value)]) + '\n')
        f.close()
