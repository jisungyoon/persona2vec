import csv
import logging
import pickle
import sys
from collections import defaultdict

import networkx as nx
from gensim.models import KeyedVectors
from persona2vec.link_prediction import LinkPredictionTask
from persona2vec.utils import read_edge_file, read_graph

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


if __name__ == "__main__":

    EMB_FILE = sys.argv[1]
    MAPPER_FILE = sys.argv[2]
    TEST_EDGE_FILE = sys.argv[-3]
    NEG_EDGE_FILE = sys.argv[-2]
    PERFORMANCE_FILE = sys.argv[-1]

    w2v_file = KeyedVectors.load_word2vec_format(EMB_FILE)
    emb = {key: w2v_file.wv[key] for key in w2v_file.vocab.keys()}

    test_edges = read_edge_file(TEST_EDGE_FILE)
    negative_edges = read_edge_file(NEG_EDGE_FILE)
    name = EMB_FILE.split("/")[7]  # location of network name

    node_to_persona = defaultdict(list)
    with open(MAPPER_FILE, "r") as f:
        lines = f.readlines()
    for line in lines:
        persona, node = line[:-1].split(" ")
        node_to_persona[node].append(persona)

    test = LinkPredictionTask(
        test_edges,
        negative_edges,
        emb,
        name=name,
        proximity_function="cos",
        is_persona_emb=True,
        node_to_persona=node_to_persona,
    )
    test.do_link_prediction()
    test.write_result(PERFORMANCE_FILE)
