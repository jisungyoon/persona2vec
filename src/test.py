import argparse
from link_prediction import linkPredictionTask
from utils import tab_printer, read_graph



def main():
    """
    Parsing command line parameters.
    Reading data, embedding base graph, creating persona graph and learning a splitter.
    saving the persona mapping and the embedding.
    """
    name = 'ca-HepTh.elist'
    G = read_graph('graph/' + name)
    
    test_object = linkPredictionTask(G, name, lambd=0.1)
    test_object.train_test_split()
    test_object.generate_negative_edges()
    test_object.learn_persona2vec_emb()
    test_object.calculate_link_prediction_score()
    test_object.calculate_ROC_AUC_value()
    test_object.print_result()

if __name__ == "__main__":
    main()
