import argparse
from persona2vec.model import Persona2Vec
from persona2vec.utils import tab_printer, read_graph


def parse_args():
    """
    Parses the Splitter arguments.
    """
    parser = argparse.ArgumentParser(description="Run splitter with node2vec")

    # input and output files
    parser.add_argument('--input', nargs='?', default='examples/graph/karate.elist',
                        help='Input network path as edgelist format')

    parser.add_argument("--persona-network", nargs="?",
                        default="examples/graph/karate_persona.elist",
                        help="Persona network path.")

    parser.add_argument("--persona-to-node", nargs="?",
                        default="examples/mapping/karate_persona_to_node.pkl",
                        help="Persona to node mapping file.")

    parser.add_argument("--node-to-persona", nargs="?",
                        default="examples/mapping/karate_node_to_persona.pkl",
                        help="Node to persona mapping file.")

    parser.add_argument('--emb', nargs='?', default='examples/emb/karate.pkl',
                        help='Persona Embeddings path')

    # hyper-parameter of ego-splitting

    parser.add_argument("--lambd",
                        type=float,
                        default=0.1,
                        help="Edge weight for persona edge, usually 0~1.")

    # hyper-parameter for learning embedding

    parser.add_argument('--dimensions', type=int, default=128,
                        help='Number of dimensions. Default is 128.')

    parser.add_argument('--walk-length', type=int, default=80,
                        help='Length of walk per source. Default is 80.')

    parser.add_argument('--num-walks', type=int, default=10,
                        help='Number of walks per source. Default is 10.')

    parser.add_argument('--window-size', type=int, default=10,
                        help='Context size for optimization. Default is 10.')

    parser.add_argument('--base_iter', type=int, default=1,
                        help='Number of epochs in base embedding')

    parser.add_argument('--p', type=float, default=1,
                        help='Return hyperparameter for random-walker. Default is 1.')

    parser.add_argument('--q', type=float, default=1,
                        help='Inout hyperparameter for random-walker. Default is 1.')

    # computation configuration
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers. Default is 8.')

    # parameters for input graph type
    parser.add_argument('--weighted', dest='weighted', action='store_true',
                        help='Boolean specifying (un)weighted. Default is unweighted.')
    parser.add_argument('--unweighted', dest='unweighted',
                        action='store_false')
    parser.set_defaults(weighted=False)

    parser.add_argument('--directed', dest='directed', action='store_true',
                        help='Graph is (un)directed. Default is undirected.')
    parser.add_argument('--undirected', dest='undirected',
                        action='store_false')

    parser.set_defaults(directed=False)

    return parser.parse_args()


def main():
    """
    Parsing command line parameters.
    Reading data, embedding base graph, creating persona graph and learning a splitter.
    saving the persona mapping and the embedding.
    """
    args = parse_args()
    tab_printer(args)
    graph = read_graph(args.input, args.weighted, args.directed)
    model = Persona2Vec(graph,
                        lambd=args.lambd,
                        directed=args.directed,
                        num_walks=args.num_walks,
                        p=args.p,
                        q=args.q,
                        dimensions=args.dimensions,
                        window_size=args.window_size,
                        base_iter=args.base_iter,
                        workers=args.workers)

    model.simulate_walks()
    model.learn_embedding()

    model.save_persona_network(args.persona_network)
    model.save_persona_to_node_mapping(args.persona_to_node)
    model.save_node_to_persona_mapping(args.node_to_persona)
    model.save_embedding(args.emb)


if __name__ == "__main__":
    main()
