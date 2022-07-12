import numpy as np
import pandas as pd
import napari
from napari.layers import Graph
from napari_graph import UndirectedGraph


def build_graph(n_nodes: int, sparsity: float) -> UndirectedGraph:
    adj_matrix = np.random.uniform(size=(n_nodes, n_nodes)) < sparsity
    np.fill_diagonal(adj_matrix, 0)

    edges = np.stack(adj_matrix.nonzero()).T
    nodes_df = pd.DataFrame(
        400 * np.random.uniform(size=(n_nodes, 4)),
        columns=["t", "z", "y", "x"],
    )

    graph = UndirectedGraph(len(nodes_df), ndim=4, n_edges=len(edges))
    graph.init_nodes_from_dataframe(nodes_df, ["t", "z", "y", "x"])
    graph.add_edges(edges)

    return graph


if __name__ == "__main__":

    viewer = napari.Viewer()
    n_nodes = 10000
    graph = build_graph(n_nodes, 2 / n_nodes)
    layer = Graph(graph, out_of_slice_display=True)
    viewer.add_layer(layer)

    napari.run()
