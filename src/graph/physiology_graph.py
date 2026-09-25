import numpy as np
import networkx as nx


PHYSIOLOGY_NODES = [
    "forehead",
    "left_cheek",
    "right_cheek",
]


PHYSIOLOGY_EDGES = [
    ("forehead", "left_cheek"),
    ("forehead", "right_cheek"),
    ("left_cheek", "right_cheek"),
]


def build_physiology_graph():
    """
    Build the 3-node graph used for
    physiological/rPPG consistency analysis.
    """

    graph = nx.Graph()

    graph.add_nodes_from(
        PHYSIOLOGY_NODES
    )

    graph.add_edges_from(
        PHYSIOLOGY_EDGES
    )

    return graph


def get_physiology_adjacency(
    graph,
):
    """
    Return adjacency matrix using a fixed
    node order.
    """

    adjacency = nx.to_numpy_array(
        graph,
        nodelist=PHYSIOLOGY_NODES,
        dtype=np.float64,
    )

    return adjacency


def get_physiology_laplacian(
    adjacency,
):
    """
    Calculate L = D - A.
    """

    degrees = np.sum(
        adjacency,
        axis=1,
    )

    degree = np.diag(
        degrees
    )

    laplacian = (
        degree
        - adjacency
    )

    return laplacian