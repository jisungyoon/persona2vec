import argparse

from persona2vec.model import Persona2Vec
from persona2vec.utils import read_graph, tab_printer


def parse_args():
    """
    Parses the persona2vec arguments.
    """
    parser = argparse.ArgumentParser(description="Run persona2vec with node2vec")

    # input and output files
    parser.add_argument(
        "--input",
        nargs="?",
        default="examples/graph/karate.elist",
        help="Input network path as edgelist format",
    )

    parser.add_argument(
        "--persona-network",
        nargs="?",
        default="examples/graph/karate_persona.elist",
        help="Persona network path.",
    )

    parser.add_argument(
        "--persona-to-node",
        nargs="?",
        default="examples/mapping/karate_persona_to_node.pkl",
        help="Persona to node mapping file.",
    )

    parser.add_argument(
        "--node-to-persona",
        nargs="?",
        default="examples/mapping/karate_node_to_persona.pkl",
        help="Node to persona mapping file.",
    )

    parser.add_argument(
        "--base-emb",
        nargs="?",
        default="examples/emb/karate_base.pkl",
        help="Base Embeddings path",
    )

    parser.add_argument(
        "--persona-emb",
        nargs="?",
        default="examples/emb/karate_persona.pkl",
        help="Persona Embeddings path",
    )

    # hyper-parameter of ego-splitting

    parser.add_argument(
        "--lambd",
        type=float,
        default=0.1,
        help="Edge weight for persona edge, usually 0~1.",
    )
    
    parser.add_argument(
        "--clustering-method",
        type=str,
        default="connected_component"",
        help="name of the clustering method that uses in splitting personas, choose one of these ('connected_component''modulairty','label_prop')",
    )

    # hyper-parameter for learning embedding

    parser.add_argument(
        "--dimensions",
        type=int,
        default=128,
        help="Number of dimensions. Default is 128.",
    )

    parser.add_argument(
        "--walk-length-base",
        type=int,
        default=40,
        help="Length of walk per source. Default is 40.",
    )

    parser.add_argument(
        "--num-walks-base",
        type=int,
        default=10,
        help="Number of walks per source. Default is 10.",
    )

    parser.add_argument(
        "--window-size-base",
        type=int,
        default=5,
        help="Context size for optimization. Default is 5.",
    )

    parser.add_argument(
        "--epoch-base",
        type=int,
        default=1,
        help="Number of epochs in the base embedding",
    )

    parser.add_argument(
        "--walk-length-persona",
        type=int,
        default=80,
        help="Length of walk per source. Default is 80.",
    )

    parser.add_argument(
        "--num-walks-persona",
        type=int,
        default=5,
        help="Number of walks per source. Default is 10.",
    )

    parser.add_argument(
        "--window-size-persona",
        type=int,
        default=2,
        help="Context size for optimization. Default is 10.",
    )

    parser.add_argument(
        "--epoch-persona",
        type=int,
        default=1,
        help="Number of epochs in persona embedding",
    )

    parser.add_argument(
        "--p",
        type=float,
        default=1,
        help="Return hyperparameter for random-walker. Default is 1.",
    )

    parser.add_argument(
        "--q",
        type=float,
        default=1,
        help="Inout hyperparameter for random-walker. Default is 1.",
    )

    # computation configuration
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers. Default is 8.",
    )

    # parameters for input graph type
    parser.add_argument(
        "--weighted",
        dest="weighted",
        action="store_true",
        help="Boolean specifying (un)weighted. Default is unweighted.",
    )
    parser.add_argument("--unweighted", dest="unweighted", action="store_false")
    parser.set_defaults(weighted=False)

    parser.add_argument(
        "--directed",
        dest="directed",
        action="store_true",
        help="Graph is (un)directed. Default is undirected.",
    )
    parser.add_argument("--undirected", dest="undirected", action="store_false")

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
    G = read_graph(args.input, args.weighted, args.directed)
    model = Persona2Vec(
        G,
        lambd=args.lambd,
        clustering_method="connected_component",
        directed=args.directed,
        num_walks_base=args.num_walks_base,
        walk_length_base=args.walk_length_base,
        window_size_base=args.window_size_base,
        num_walks_persona=args.num_walks_persona,
        walk_length_persona=args.walk_length_persona,
        window_size_persona=args.window_size_persona,
        p=args.p,
        q=args.q,
        dimensions=args.dimensions,
        epoch_base=args.epoch_base,
        epoch_persona=args.epoch_persona,
        workers=args.workers,
    )

    model.save_persona_network(args.persona_network)
    model.save_persona_to_node_mapping(args.persona_to_node)
    model.save_node_to_persona_mapping(args.node_to_persona)
    model.save_base_embedding(args.base_emb)
    model.save_persona_embedding(args.persona_emb)


if __name__ == "__main__":
    main()
