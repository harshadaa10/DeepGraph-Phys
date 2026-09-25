import numpy as np
import networkx as nx


# --------------------------------------------------
# Facial graph nodes
# --------------------------------------------------

FACIAL_NODES = [
    "forehead",
    "left_cheek",
    "right_cheek",
    "left_eye",
    "right_eye",
    "nose",
    "mouth",
]


# --------------------------------------------------
# Anatomical graph edges
# --------------------------------------------------

FACIAL_EDGES = [
    ("forehead", "left_eye"),
    ("forehead", "right_eye"),

    ("left_eye", "nose"),
    ("right_eye", "nose"),

    ("left_eye", "left_cheek"),
    ("right_eye", "right_cheek"),

    ("left_cheek", "nose"),
    ("right_cheek", "nose"),

    ("left_cheek", "mouth"),
    ("right_cheek", "mouth"),

    ("nose", "mouth"),
]


def build_facial_graph():
    """
    Build the anatomical facial-region graph.
    """

    graph = nx.Graph()

    graph.add_nodes_from(
        FACIAL_NODES
    )

    graph.add_edges_from(
        FACIAL_EDGES
    )

    return graph


def get_adjacency_matrix(graph):
    """
    Return graph adjacency matrix using
    a fixed facial-node ordering.
    """

    adjacency = nx.to_numpy_array(
        graph,
        nodelist=FACIAL_NODES,
        dtype=np.float64,
    )

    return adjacency


def get_degree_matrix(adjacency):
    """
    Compute degree matrix D.
    """

    degrees = np.sum(
        adjacency,
        axis=1
    )

    degree_matrix = np.diag(
        degrees
    )

    return degree_matrix


def get_laplacian_matrix(adjacency):
    """
    Compute unnormalized graph Laplacian:

        L = D - A
    """

    degree_matrix = get_degree_matrix(
        adjacency
    )

    laplacian = (
        degree_matrix
        - adjacency
    )

    return laplacian 