import sys
import os 
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


   
    
    
if __name__ == "__main__":
    IN_FILES = sys.argv[1:-2]
    NUMBER_OF_CORES = int(sys.argv[-2])
    OUT_DIR = sys.argv[-1]
    
    mk_outdir(OUT_DIR)
    main(IN_FILES, NUMBER_OF_CORES, OUT_DIR)