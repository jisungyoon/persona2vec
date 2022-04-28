from scipy import sparse
import numpy as np
from . import utils
from sklearn.decomposition import TruncatedSVD

def fastRP(G, dimensions, steps=10):
    """https://arxiv.org/pdf/1908.11512.pdf."""
    A = utils.to_adjacency_matrix(G)
    R = sparse.random(
        A.shape[0],
        dimensions,
        density=1 / 3,
        random_state=42,
        data_rvs=lambda x: np.random.choice(
            [-np.sqrt(3), np.sqrt(3)], size=x, replace=True
        ),
    ).toarray()

    S = np.zeros(R.shape)
    denom = np.maximum(np.array(A.sum(axis=1)).reshape(-1), 1e-32)
    normalized_conv_matrix = sparse.diags(1 / denom) @ A.copy()
    for _ in range(steps):
        S += R
        S = normalized_conv_matrix @ S
    S /= np.maximum(1, steps)
    return S

def laplacian_eigenmap(G, dimensions):
    A = utils.to_adjacency_matrix(G)

    # Compute the (shifted) normalized laplacian matrix
    deg = np.array(A.sum(axis=1)).reshape(-1)
    Dsqrt = sparse.diags(1 / np.maximum(np.sqrt(deg), 1e-12), format="csr")
    L = Dsqrt @ A @ Dsqrt

    # Eigen decomposition
    svd = TruncatedSVD(n_components=dimensions + 1, random_state=42)
    svd.fit(L)
    u = svd.components_.T
    s = np.abs(svd.singular_values_)

    # remove trivial eigenvector
    order = np.argsort(-s)[1:]
    return u[:, order]