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
        100 * np.random.randn(n_nodes, 3),
        columns=["z", "y", "x"]
    )

    graph = UndirectedGraph(len(nodes_df), ndim=3, n_edges=len(edges))
    graph.init_nodes_from_dataframe(nodes_df, ["z", "y", "x"])
    graph.add_edges(edges)

    return graph


if __name__ == "__main__":

    viewer = napari.Viewer()
    graph = build_graph(10000, 0.01)
    layer = Graph(graph, out_of_slice_display=True)
    viewer.add_layer(layer)

    napari.run()
