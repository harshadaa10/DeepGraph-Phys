from pathlib import Path
import sys

import matplotlib.pyplot as plt
import networkx as nx


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

sys.path.append(
    str(PROJECT_ROOT)
)


from src.graph.facial_graph import (
    FACIAL_NODES,
    build_facial_graph,
    get_adjacency_matrix,
    get_degree_matrix,
    get_laplacian_matrix,
)


OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "figures"
    / "facial_graph.png"
)


def main():

    print(
        "\nDeepGraph-Phys — Facial Graph Test"
    )

    print("-" * 50)

    # ----------------------------------------------
    # Build graph
    # ----------------------------------------------

    graph = build_facial_graph()

    print(
        "Number of nodes:",
        graph.number_of_nodes()
    )

    print(
        "Number of edges:",
        graph.number_of_edges()
    )

    print("\nNodes:")

    for node in graph.nodes:
        print(
            " -",
            node
        )

    print("\nEdges:")

    for edge in graph.edges:
        print(
            " -",
            edge
        )

    # ----------------------------------------------
    # Matrices
    # ----------------------------------------------

    adjacency = get_adjacency_matrix(
        graph
    )

    degree = get_degree_matrix(
        adjacency
    )

    laplacian = get_laplacian_matrix(
        adjacency
    )

    print(
        "\nNode order:"
    )

    print(
        FACIAL_NODES
    )

    print(
        "\nAdjacency matrix A:"
    )

    print(
        adjacency.astype(int)
    )

    print(
        "\nDegree matrix D:"
    )

    print(
        degree.astype(int)
    )

    print(
        "\nGraph Laplacian L = D - A:"
    )

    print(
        laplacian.astype(int)
    )

    # ----------------------------------------------
    # Basic checks
    # ----------------------------------------------

    print(
        "\nGraph connected:",
        nx.is_connected(graph)
    )

    print(
        "Laplacian symmetric:",
        (
            laplacian.T
            == laplacian
        ).all()
    )

    print(
        "Laplacian row sums:",
        laplacian.sum(axis=1)
    )

    # ----------------------------------------------
    # Visualization
    # ----------------------------------------------

    positions = {
        "forehead": (0, 3),

        "left_eye": (-1, 2),
        "right_eye": (1, 2),

        "nose": (0, 1),

        "left_cheek": (-1.5, 1),
        "right_cheek": (1.5, 1),

        "mouth": (0, 0),
    }

    plt.figure(
        figsize=(8, 8)
    )

    nx.draw(
        graph,
        pos=positions,
        with_labels=True,
        node_size=3000,
        font_size=9,
        font_weight="bold",
    )

    plt.title(
        "DeepGraph-Phys Facial Region Graph"
    )

    plt.tight_layout()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.savefig(
        OUTPUT_PATH,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    print(
        "\nGraph visualization saved to:"
    )

    print(
        OUTPUT_PATH
    )

    print(
        "\nFacial graph test completed successfully."
    )


if __name__ == "__main__":
    main()