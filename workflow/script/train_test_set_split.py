import sys
import logging

from multiprocessing import Process

from persona2vec.network_train_test_splitter import networkTrainTestSplitter
from persona2vec.utils import read_graph

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


def run_parallel(function, args, number_of_cores):
    procs = [Process(target=function, args=[proc_num] + args)
             for proc_num in range(number_of_cores)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()


def train_test_set_split(proc_num, IN_FILE):
    logging.info('Core-' + str(proc_num) + ' start')
    G = read_graph(IN_FILE)
    OUTPUT_PATH = IN_FILE.split('.')[0] + '_' + str(proc_num)
    splitter = networkTrainTestSplitter(G)
    splitter.train_test_split()
    splitter.generate_negative_edges()
    splitter.save_splitted_result(OUTPUT_PATH)


if __name__ == "__main__":
    IN_FILE = sys.argv[1]
    NUMBER_OF_TEST_SET = int(sys.argv[2])
    
    run_parallel(train_test_set_split, [IN_FILE], NUMBER_OF_TEST_SET)
    
    
    
