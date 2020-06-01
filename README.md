# persona2vec
A simple implementation of persona2vec

## Installation
You can use persona2vec as library. It is very simple.
```
python libs/setup.py install
```

### Requirements
The codebase is implemented in Python 3.7.3, and package versions used for development are just below. This library works well on various environment. If there is a problem, please let me know with issue. I will handle it.
```
networkx          2.3
tqdm              4.28.1
numpy             1.15.4
pandas            0.23.4
texttable         1.5.0
scipy             1.1.0
argparse          1.1.0
gensim            3.6.0
python-louvain      - 
```

## How to use
You can use persona2vec as library.
```
from persona2vec.model import Persona2Vec
from persona2vec.utils import read_graph

G = read_graph(NETWORK_FILE_NAME)
model = Persona2Vec(
    G, lambd=LAMBDA, dimensions=DIM, workers=NUMBER_OF_CORES)
emb = model.embedding
```
For detail, please check a example notebook, `examples/example_karate.ipynb`

### Datasets - inputs
There is a utility function **read_graph** in `persona2vec/utils.py` for reading input files.
You can easily make the edgelist file(*.elist) with networkx function [nx.write_edgelist](https://networkx.github.io/documentation/networkx1.10/reference/generated/networkx.readwrite.edgelist.write_edgelist.html) 


### Outputs
There are 3 outputs on persona2vec

1. **Persona network**, Persona network is a result network of ego-splitting. File format is edgelist(*.elist), which is same to format of inputs
  
2. **persona to node, node to persona mapping**, Mappings is a dict that connnect orginal node and splitted persona nodes or vice versa. Bascially, relation between node and persona is 1 to M relations. File format is pickle(*.json)
  
3. **Base embedding and Persona embedding**, Base embedding and persona embedding of Persona2vec. File format is pickle(.w2v), See [save_word2vec_format](https://radimrehurek.com/gensim/models/keyedvectors.html)

## Use as command line interface

The training of a persona2vec is handled by the `src/main.py` script which provides the following command line arguments.
The following commands learn an embedding and save it with the persona map. Training a model on the default dataset.
```
persona2vec --input [INPUT_FILES_DIR] 
            --persona-network [PERSONA_NETWORK_DIR] \
            --persona-to-node [PERSONA_TO_NODE_DIR] \
            --node-to-persona [NODE_TO_PERSONA_DIR] \
            --base-emb [BASE_EMB_DIR] \
            --persona-emb [PERSONA_EMB_DIR]
```
If you want to train a Persona2vec with 32 dimensions.
```
persona2vec --dimensions 32
```
And, you can also change configurations for random walker easily with
```
persona2vec --number-of-walks 20 --walk-length 80
```

#### Input and output options
   
```
  --input [INPUT]       Input network path as edgelist format
  --persona-network [PERSONA_NETWORK]
                        Persona network path.
  --persona-to-node [PERSONA_TO_NODE]
                        Persona to node mapping file.
  --node-to-persona [NODE_TO_PERSONA]
                        Node to persona mapping file.
  --base-emb [BASE_EMB]
                        Base Embeddings path
  --persona-emb [PERSONA_EMB]
                        Persona Embeddings path
```
#### Model options
```
  --lambd LAMBD         Edge weight for persona edge, usually 0~1.
  --clustering-method CLUSTERING_METHOD
                        name of the clustering method that uses in splitting
                        personas, choose one of these
                        ('connected_component''modulairty','label_prop')
  --dimensions DIMENSIONS
                        Number of dimensions. Default is 128.
  --walk-length-base WALK_LENGTH_BASE
                        Length of walk per source. Default is 40.
  --num-walks-base NUM_WALKS_BASE
                        Number of walks per source. Default is 10.
  --window-size-base WINDOW_SIZE_BASE
                        Context size for optimization. Default is 5.
  --epoch-base EPOCH_BASE
                        Number of epochs in the base embedding
  --walk-length-persona WALK_LENGTH_PERSONA
                        Length of walk per source. Default is 80.
  --num-walks-persona NUM_WALKS_PERSONA
                        Number of walks per source. Default is 10.
  --window-size-persona WINDOW_SIZE_PERSONA
                        Context size for optimization. Default is 10.
  --epoch-persona EPOCH_PERSONA
                        Number of epochs in persona embedding
  --p P                 Return hyperparameter for random-walker. Default is 1.
  --q Q                 Inout hyperparameter for random-walker. Default is 1.
  --workers WORKERS     Number of parallel workers. Default is 8.
  --weighted            Boolean specifying (un)weighted. Default is
                        unweighted.
  --unweighted
  --directed            Graph is (un)directed. Default is undirected.
  --undirected
```

