import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from torch_geometric.nn.models import InnerProductDecoder, VGAE
from torch_geometric.nn.conv import GCNConv
from torch_geometric.utils import from_networkx, negative_sampling, remove_self_loops, add_self_loops
from torch_geometric.utils import train_test_split_edges

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix

def read_graph(input_file_path, weighted=False, directed=False):
    """
    Reads the input network and return a networkx Graph object.
    :param input_file_path: File path of input graph
    :param weighted: weighted network(True) or unweighted network(False)
    :param directed: directed network(True) or undirected network(False)
    :return G: output network
    """
    if weighted:
        G = nx.read_edgelist(input_file_path, nodetype=str, create_using=nx.DiGraph(),)
    else:
        G = nx.read_edgelist(input_file_path, nodetype=str, create_using=nx.DiGraph())
        for edge in G.edges():
            G[edge[0]][edge[1]]["weight"] = 1

    if not directed:
        G = G.to_undirected()

    return G


class GCNEncoder(nn.Module):
    """
    Encoder which uses Graph Convolution modules.
    """
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCNEncoder, self).__init__()
        # print(type(in_channels), type(hidden_channels), type(out_channels))
        # print(in_channels, hidden_channels, out_channels)
        self.gcn_shared = GCNConv(in_channels, hidden_channels)
        self.gcn_mu = GCNConv(hidden_channels, out_channels)
        self.gcn_logvar = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.gcn_shared(x, edge_index))
        mu = self.gcn_mu(x, edge_index)
        logvar = self.gcn_logvar(x, edge_index)
        return mu, logvar


class VariationalAutoEncoder(VGAE):
    """
    Variational Graph Auto-Encoder: https://arxiv.org/abs/1611.07308
    """
    def __init__(self, enc_in_channels, enc_hidden_channels, enc_out_channels):
        super(VariationalAutoEncoder, self).__init__(encoder=GCNEncoder(enc_in_channels,
                                                          enc_hidden_channels,
                                                          enc_out_channels),
                                                     decoder=InnerProductDecoder())

    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        adj_pred = self.decoder.forward_all(z)
        return adj_pred

    def loss(self, x, pos_edge_index, all_edge_index):
        z = self.encode(x, pos_edge_index)

        pos_loss = -torch.log(
            self.decoder(z, pos_edge_index, sigmoid=True) + 1e-15).mean()

        # Do not include self-loops in negative samples
        all_edge_index_tmp, _ = remove_self_loops(all_edge_index)
        all_edge_index_tmp, _ = add_self_loops(all_edge_index_tmp)

        neg_edge_index = negative_sampling(all_edge_index_tmp, z.size(0), pos_edge_index.size(1))
        neg_loss = -torch.log(1 - self.decoder(z, neg_edge_index, sigmoid=True) + 1e-15).mean()

        kl_loss = 1 / x.size(0) * self.kl_loss()

        return pos_loss + neg_loss + kl_loss

    def single_test(self, x, train_pos_edge_index, test_pos_edge_index, test_neg_edge_index):
        with torch.no_grad():
            z = self.encode(x, train_pos_edge_index)
        roc_auc_score, average_precision_score = self.test(z, test_pos_edge_index, test_neg_edge_index)
        return roc_auc_score, average_precision_score


class DeepVGAE:
    """
    Deep Variational Auto-Encoder node embedding object
    """
    def __init__(
        self,
        G,
        num_features=None,
        directed=False,
        hidden_dimensions=256,
        dimensions=128,
        lr=0.01,
        val_size=0.05,
        test_size=0.1,
        epochs=10
    ):
        """
        :param G: NetworkX graph object.
        :param num_features: Number of features for node else number of nodes
        :param directed: Directed network(True) or undirected network(False)
        :param hidden_dimensions: Hidden dimension of encoder
        :param dimensions: Dimension of node embedding
        :param lr: Learning rate
        :param val_size: Fraction of the data to be used as validation data
        :param test_size: Fraction of the data to be used as test data
        :param epochs: Number of epochs
        """
        self.G = G
        self.directed = directed

        # parameters for VGAE model
        self.num_features = num_features if num_features else self.G.number_of_nodes()
        self.hidden_dimensions = hidden_dimensions
        self.dimensions = dimensions

        # hyper-parameters for VGAE model
        self.lr = lr
        self.epochs = epochs
        self.val_size = val_size
        self.test_size = test_size

        # declaring model object
        self.model = VariationalAutoEncoder(
            self.num_features,
            self.hidden_dimensions,
            self.dimensions
        )
        self.optimizer = Adam(self.model.parameters(), lr=self.lr)

        self.gen_node_mappings()
        self.network_to_data()

    def gen_node_mappings(self):
        self.node_mappings = {node: i for i, node in enumerate(self.G)}

    def network_to_data(self):
        """
        Converting networkx.Graph() object to torch_geometric.data.Data
        """
        self.data = from_networkx(self.G)
        self.data.x = torch.from_numpy(csr_matrix.toarray(nx.adjacency_matrix(self.G))).float()

    def learn_embedding(self):
        """
        Training the VGAE model on given network and generating embeddings.
        """
        torch.manual_seed(12345)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        all_edge_index = self.data.edge_index
        self.data = train_test_split_edges(self.data, self.val_size, self.test_size)

        for epoch in range(self.epochs):
            self.model.train()
            self.optimizer.zero_grad()
            loss = self.model.loss(self.data.x, self.data.train_pos_edge_index, all_edge_index)
            loss.backward()
            self.optimizer.step()

            self.model.eval()
            roc_auc, ap = self.model.single_test(self.data.x,
                                                 self.data.train_pos_edge_index,
                                                 self.data.test_pos_edge_index,
                                                 self.data.test_neg_edge_index)
            print("Epoch {} - Loss: {} ROC_AUC: {} Precision: {}".format(epoch, loss.cpu().item(), roc_auc, ap))

        emb = self.model.encode(self.data.x, all_edge_index).cpu().detach().numpy().tolist()
        self.embedding = { node: emb[self.node_mappings[node]] for node in self.G.nodes() }
        return self.embedding

    def save_embedding(self, file_path):
        """
        :param file_path: file path to store the embeddings of the model
        :return:
        """
        if self.embedding is None:
            self.learn_embedding()
        with open(file_path, "w") as f:
            json.dump(self.embedding, f)


# Ignore below code. It is just used for testing the model
if __name__=='__main__':

    G = read_graph("data/ppi.elist")
    model = DeepVGAE(G)
    print(model.learn_embedding())
    model.save_embedding("data/embeddings/test_emb.pkl")

#     # importing dataset related functions
#     from torch_geometric.datasets import Planetoid, PPI
#     import torch_geometric.transforms as T

#     # Parameters are set as per benchmarking GNN paper: https://arxiv.org/pdf/2102.12557.pdf
#     args = {
#         "enc_in_channels": 50,
#         "enc_hidden_channels": 256,
#         "enc_out_channels": 128,
#         "lr": 0.01,
#         "epoch": 10,
#         "val_size": 0.05,
#         "test_size": 0.1
#     }

#     torch.manual_seed(12345)
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     model = VariationalAutoEncoder(
#         args["enc_in_channels"],
#         args["enc_hidden_channels"],
#         args["enc_out_channels"]
#     ).to(device)
#     optimizer = Adam(model.parameters(), lr=args["lr"])

#     dataset = PPI(os.path.join('data', 'PPI'), transform=T.NormalizeFeatures())
#     data = dataset[0].to(device)
#     all_edge_index = data.edge_index
#     data = train_test_split_edges(data, args["val_size"], args["test_size"])
#     print(data.x.type(), all_edge_index.type())

#     for epoch in range(args["epoch"]):
#         model.train()
#         optimizer.zero_grad()
#         loss = model.loss(data.x, data.train_pos_edge_index, all_edge_index)
#         loss.backward()
#         optimizer.step()

#         model.eval()
#         roc_auc, ap = model.single_test(data.x,
#                                         data.train_pos_edge_index,
#                                         data.test_pos_edge_index,
#                                         data.test_neg_edge_index)
#         print("Epoch {} - Loss: {} ROC_AUC: {} Precision: {}".format(epoch, loss.cpu().item(), roc_auc, ap))
