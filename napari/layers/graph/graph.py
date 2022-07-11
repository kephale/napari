from typing import List, Tuple, Union

import numpy as np
from napari_graph import UndirectedGraph
from napari_graph._base_graph import BaseGraph

from ...utils.translations import trans
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
        self._edges_indices_view = []

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

    def _slice_data(
        self, dims_indices
    ) -> Tuple[List[int], Union[float, np.ndarray]]:
        """Determines the slice of points given the indices."""
        slice_indices, scale = super()._slice_data(dims_indices)
        valid = (
            self.data._buffer2world[slice_indices] != BaseGraph._NODE_EMPTY_PTR
        )
        slice_indices = slice_indices[valid]
        if isinstance(scale, np.ndarray):
            scale = scale[valid]
        return slice_indices, scale

    def _set_view_slice(self) -> None:
        """Sets the view given the indices to slice with."""
        super()._set_view_slice()
        self._set_edges_indices_view()

    def _set_edges_indices_view(self):
        """Sets edges indices view from `_indices_view`"""
        if len(self.data) == 0 or len(self._indices_view) == 0:
            self._edges_indices_view = []
        else:
            mask = np.zeros(self.data.n_allocated_nodes, dtype=bool)
            mask[self._indices_view] = True
            _, edges = self.data.edges_buffers(is_buffer_domain=True)
            both_in_view = np.logical_and(mask[edges[:, 0]], mask[edges[:, 1]])
            (self._edges_indices_view,) = np.nonzero(both_in_view)

    @property
    def edges_coordinates(self) -> np.ndarray:
        _, edges = self.data.edges_buffers(is_buffer_domain=True)
        coords = self.data._coords[edges]
        coords = coords[..., self._dims_displayed]
        return coords

    @property
    def _view_edges_coordinates(self) -> np.ndarray:
        return self.edges_coordinates[self._edges_indices_view]

    @property
    def data(self) -> BaseGraph:
        return self._data

    @data.setter
    def data(self, data) -> None:
        prev_size = self.data.n_allocated_nodes
        self._data = data_to_graph(data)
        self._data_changed(prev_size)

    def _get_state(self):
        """Get dictionary of layer state.

        Returns
        -------
        state : dict
            Dictionary of layer state.
        """
        state = super()._get_state()
        # state.update({})   # FIXME
        return state

    def add(self, coords, indices=None) -> None:
        """Adds nodes at coordinates.

        Parameters
        ----------
        coords : sequence of indices to add point at
        indices : optional indices of the newly inserted nodes.
        """
        coords = np.atleast_2d(coords)
        if indices is None:
            new_starting_idx = self.data._buffer2world.max() + 1
            indices = np.arange(
                new_starting_idx, new_starting_idx + len(coords)
            )

        if len(coords) != len(indices):
            raise ValueError(
                trans._(
                    'coordinates and indices must have the same length. Found {coords_size} and {idx_size}',
                    coords_size=len(coords),
                    idx_size=len(indices),
                )
            )

        prev_size = self.data.n_allocated_nodes

        for idx, coord in zip(indices, coords):
            self.data.add_node(idx, coord)

        self._data_changed(prev_size)

    def remove(self, indices: Union[np.ndarray, List[int]]) -> None:
        """Removes nodes given their indices."""
        prev_size = self.data.n_allocated_nodes
        if isinstance(indices, np.ndarray):
            indices = indices.tolist()
        indices.sort(reverse=True)
        for idx in indices:
            self.data.remove_node(idx)
        self._data_changed(prev_size)

    def _remove_from_data(self, indices: Union[np.ndarray, List[int]]) -> None:
        """Auxiliary function to remove items given their indices."""
        indices = self.data._buffer2world[indices]
        self.remove(indices)

    def _data_changed(self, prev_size: int) -> None:
        self._update_props_and_style(self.data.n_allocated_nodes, prev_size)
        self._update_dims()
        self.events.data(value=self.data)
        self._set_editable()

    def _move_points(
        self, ixgrid: Tuple[np.ndarray, np.ndarray], shift: np.ndarray
    ) -> None:
        """Move points along a set a coordinates given a shift.

        Parameters
        ----------
        ixgrid : Tuple[np.ndarray, np.ndarray]
            Crossproduct indexing grid of node indices and dimensions, see `np.ix_`
        shift : np.ndarray
            Selected coordinates shift
        """
        self.data._coords[ixgrid] += shift

    def _paste_data(self):
        raise NotImplementedError

    def _copy_data(self):
        raise NotImplementedError
