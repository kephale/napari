from vispy import gloo
from vispy.visuals import LineVisual

from ...layers import Graph
from ..layers.points import VispyPointsLayer


class VispyGraphLayer(VispyPointsLayer):
    def __init__(self, layer: Graph):
        super().__init__(layer)
        self.node.add_subvisual(LineVisual())

    def _on_data_change(self):
        self._set_graph_edges_data()
        super()._on_data_change()

    def _set_graph_edges_data(self):
        """Temporary function, should be refactored to use existing VectorVisual"""
        # FIXME
        if len(self.node._subvisuals) <= 4:
            return

        edges = self.layer.edges_coordinates
        flat_edges = edges.reshape((-1, edges.shape[-1]))  # (N x 2, D)
        flat_edges = flat_edges[:, ::-1]

        subvisual = self.node._subvisuals[4]
        subvisual._line_visual._pos_vbo = gloo.VertexBuffer()
        subvisual.set_data(
            flat_edges,
            color='white',
            connect='segments',
            width=1,
        )
