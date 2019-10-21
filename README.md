# persona2vec
A simple implementation of persona2vec

## Installation
You can use persona2vec as library. It is very simple.
```
python libs/setup.py install
```

### Requirements
The codebase is implemented in Python 3.5.2. package versions used for development are just below. This library works well on various envirionet. If there is a problem, please let me know with issue. I will handle it.
```
networkx          1.11
tqdm              4.28.1
numpy             1.15.4
pandas            0.23.4
texttable         1.5.0
scipy             1.1.0
argparse          1.1.0
gensim            3.6.0
```

## Usage
You can use persona2vec as library.
```
    G = read_graph(NETWORK_FILE_NAME)
    model = Persona2Vec(
        G, lambd=LAMBDA, dimensions=DIM, workers=NUMBER_OF_CORES)
    model.simulate_walks()
    emb = model.learn_embedding()
```
For detail, please check a exmaple notebook, examples/example_karate.ipynb

### Datasets - inputs
There is a utility function **read_graph** in ituls for reading input files.
You can easily make the edgelist file(*.elist) with networkx function [nx.write_edgelist](https://networkx.github.io/documentation/networkx1.10/reference/generated/networkx.readwrite.edgelist.write_edgelist.html) 


### Outputs
There are 3 outputs on persona2vec

1. **Persona network**, Persona network is a result network of ego-splitting. File format is edgelist(*.elist), which is same to format of inputs
  
2. **persona to node, node to persona mapping**, Mappings is a dict that connnect orginal node and splitted persona nodes or vice versa. Bascially, relation between node and persona is 1 to M relations. File format is pickle(.pkl)
  
3. **Persona embedding**, Result embedding of Spitter on persona graph. This embedding is final results of this resposiotry. File format is pickle(.pkl).

## Use as command line interface

The training of a persona2vec is handled by the `src/main.py` script which provides the following command line arguments.
The following commands learn an embedding and save it with the persona map. Training a model on the default dataset.
```
persona2vec --input [INPUT_FILES_DIR] 
            --persona-network [PERSONA_NETWORK_DIR] \
            --persona-to-node [PERSONA_TO_NODE_DIR] \
            --node-to-persona [NODE_TO_PERSONA_DIR] \
            --emb [RESULT_EMBE_DIR]
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
  --emb [EMB]           Persona Embeddings path
```
#### Model options
```
  --lambd LAMBD         Edge weight for persona edge, usually 0~1.
  --dimensions DIMENSIONS
                        Number of dimensions. Default is 128.
  --walk-length WALK_LENGTH
                        Length of walk per source. Default is 80.
  --num-walks NUM_WALKS
                        Number of walks per source. Default is 10.
  --window-size WINDOW_SIZE
                        Context size for optimization. Default is 10.
  --base_iter BASE_ITER
                        Number of epochs in embedding
  --p P                 Return hyperparameter for random-walker. Default is 1.
  --q Q                 Inout hyperparameter for random-walker. Default is 1.
  --workers WORKERS     Number of parallel workers. Default is 8.
  --weighted            Boolean specifying (un)weighted. Default is
                        unweighted.
  --unweighted
  --directed            Graph is (un)directed. Default is undirected.
  --undirected
```

