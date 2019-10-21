import numpy as np
from sklearn.metrics import roc_auc_score

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


class LinkPredictionTask(object):
    """
    Link prediction object
    Given a test edges and negative edges and embedding, calculate ROC_AUC values.
    """
    def __init__(self,
                 test_edges,
                 negative_edges,
                 emb,
                 name,
                 is_persona_emb=False,
                 node_to_persona={},
                 aggregate_function=max):
        """
        :param test_edges: list of test edges
        :param negative_edges: list of negative edges
        :param emb: embedding results
        :param name: name of model for record
        :param is_persona_emb: Persona embedding or not
        :param node_to_persona: mapping dict node to personas (optional for persona embedding)
        :param aggregate_function: aggregate function (optional for persona embedding)
        """
        self.test_edges = test_edges
        self.negative_edges = negative_edges
        self.emb = emb
        self.name = name

        self.link_prediction_score_positive = []
        self.link_prediction_score_negative = []

        # for persona based embedding
        self.is_persona_emb = is_persona_emb
        self.node_to_persona = node_to_persona
        self.aggregate_function = aggregate_function

    def do_link_prediction(self):
        """
        Execute link prediction
        """
        self.calculate_link_prediction_score()
        self.calculate_ROC_AUC_value()
        self.print_result()

    def calculate_link_prediction_score(self):
        """
        Calculate similarity score for test and negative edges
        """
        calculate_method = self.calculate_score_persona if self.is_persona_emb else self.calculate_score
        self.link_prediction_score_positive = np.array(
            calculate_method(self.test_edges))
        self.link_prediction_score_negative = np.array(
            calculate_method(self.negative_edges))

    def calculate_score_persona(self, edge_list):
        """
        Calculate persona based similarity score for edge_list
        :param edge_list: list of target edges.
        :return: score_list: score list of given edge_lists
        """
        score_list = []
        for src, tag in edge_list:
            src_personas = self.node_to_persona[src]
            tag_personas = self.node_to_persona[tag]
            max_sim = self.aggregate_function(
                [np.dot(self.emb[src_persona], self.emb[tag_persona])
                 for src_persona in src_personas for tag_persona in tag_personas])
            score_list.append(max_sim)
        return score_list

    def calculate_score(self, edge_list):
        """
        Calculate similarity score for edge_list
        :param edge_list: list of target edges.
        :return: score_list: score list of given edge_lists
        """
        score_list = [np.dot(self.emb[source], self.emb[target])
                      for source, target in edge_list]
        return score_list

    def calculate_ROC_AUC_value(self):
        """
        Calculate ROC_AUC values
        """
        logging.info('Calculate ROC_AUC values')
        y_true = np.concatenate(
            [np.ones_like(self.link_prediction_score_positive),
             np.zeros_like(self.link_prediction_score_negative)])
        y_score = np.concatenate(
            [self.link_prediction_score_positive,
             self.link_prediction_score_negative], axis=0)
        self.ROC_AUC_value = roc_auc_score(y_true, y_score)

    def print_result(self):
        logging.info(self.name)
        logging.info(self.ROC_AUC_value)

    def write_result(self, file_path):
        """
        Write name and ROC_AUC values as tsv format
        :param file_path: file_path for recording ROC_AUC values
        """
        f = open(file_path, 'a')
        f.write("{}\t{}\n".format(*[self.name, str(self.ROC_AUC_value)]))
        f.close()