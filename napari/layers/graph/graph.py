import numpy as np
from napari_graph import UndirectedGraph
from napari_graph._base_graph import BaseGraph

from ..points.points import _BasePoints


def data_to_graph(data) -> BaseGraph:

    if data is None:
        return UndirectedGraph(n_nodes=100, ndim=3, n_edges=200)  # FIXME

    elif isinstance(data, BaseGraph):
        return data

    raise NotImplementedError


class Graph(_BasePoints):
    def __init__(
        self,
        data=None,
        *,
        ndim=None,
        features=None,
        properties=None,
        text=None,
        symbol='o',
        size=10,
        edge_width=0.1,
        edge_width_is_relative=True,
        edge_color='black',
        edge_color_cycle=None,
        edge_colormap='viridis',
        edge_contrast_limits=None,
        face_color='white',
        face_color_cycle=None,
        face_colormap='viridis',
        face_contrast_limits=None,
        out_of_slice_display=False,
        n_dimensional=None,
        name=None,
        metadata=None,
        scale=None,
        translate=None,
        rotate=None,
        shear=None,
        affine=None,
        opacity=1,
        blending='translucent',
        visible=True,
        cache=True,
        property_choices=None,
        experimental_clipping_planes=None,
        shading='none',
        experimental_canvas_size_limits=(0, 10000),
        shown=True,
    ):
        # Save the point coordinates
        self._data = data_to_graph(data)

        super().__init__(
            data,
            ndim=ndim,
            features=features,
            properties=properties,
            text=text,
            symbol=symbol,
            size=size,
            edge_width=edge_width,
            edge_width_is_relative=edge_width_is_relative,
            edge_color=edge_color,
            edge_color_cycle=edge_color_cycle,
            edge_colormap=edge_colormap,
            edge_contrast_limits=edge_contrast_limits,
            face_color=face_color,
            face_color_cycle=face_color_cycle,
            face_colormap=face_colormap,
            face_contrast_limits=face_contrast_limits,
            out_of_slice_display=out_of_slice_display,
            n_dimensional=n_dimensional,
            name=name,
            metadata=metadata,
            scale=scale,
            translate=translate,
            rotate=rotate,
            shear=shear,
            affine=affine,
            opacity=opacity,
            blending=blending,
            visible=visible,
            cache=cache,
            property_choices=property_choices,
            experimental_clipping_planes=experimental_clipping_planes,
            shading=shading,
            experimental_canvas_size_limits=experimental_canvas_size_limits,
            shown=shown,
        )

    @property
    def _points_data(self) -> np.ndarray:
        return self._data.coordinates()

    @property
    def data(self) -> np.ndarray:
        """(N, D) array: coordinates for N points in D dimensions."""
        pass

    @data.setter
    def data(self, data) -> None:
        pass
