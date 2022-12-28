from typing import List, Tuple, Union

import numpy as np
from napari_graph import BaseGraph, UndirectedGraph

from ...utils.translations import trans
from ..points.points import _BasePoints


def data_to_graph(data) -> BaseGraph:

    if data is None:
        return UndirectedGraph(n_nodes=100, ndim=3, n_edges=200)  # FIXME

    elif isinstance(data, BaseGraph):
        return data

    raise NotImplementedError


class Graph(_BasePoints):
    """Graph layer.
    
    # TODO review documentation, copied from _BasePoints

    Parameters
    ----------
    data : napari-graph Graph structure.
        Graph data.
    ndim : int
        Number of dimensions for shapes. When data is not None, ndim must be D.
        An empty points layer can be instantiated with arbitrary ndim.
    features : dict[str, array-like] or DataFrame
        Features table where each row corresponds to a point and each column
        is a feature.
    properties : dict {str: array (N,)}, DataFrame
        Properties for each point. Each property should be an array of length N,
        where N is the number of points.
    property_choices : dict {str: array (N,)}
        possible values for each property.
    text : str, dict
        Text to be displayed with the points. If text is set to a key in properties,
        the value of that property will be displayed. Multiple properties can be
        composed using f-string-like syntax (e.g., '{property_1}, {float_property:.2f}).
        A dictionary can be provided with keyword arguments to set the text values
        and display properties. See TextManager.__init__() for the valid keyword arguments.
        For example usage, see /napari/examples/add_points_with_text.py.
    symbol : str
        Symbol to be used for the point markers. Must be one of the
        following: arrow, clobber, cross, diamond, disc, hbar, ring,
        square, star, tailed_arrow, triangle_down, triangle_up, vbar, x.
    size : float, array
        Size of the point marker in data pixels. If given as a scalar, all points are made
        the same size. If given as an array, size must be the same or broadcastable
        to the same shape as the data.
    edge_width : float, array
        Width of the symbol edge in pixels.
    edge_width_is_relative : bool
        If enabled, edge_width is interpreted as a fraction of the point size.
    edge_color : str, array-like, dict
        Color of the point marker border. Numeric color values should be RGB(A).
    edge_color_cycle : np.ndarray, list
        Cycle of colors (provided as string name, RGB, or RGBA) to map to edge_color if a
        categorical attribute is used color the vectors.
    edge_colormap : str, napari.utils.Colormap
        Colormap to set edge_color if a continuous attribute is used to set face_color.
    edge_contrast_limits : None, (float, float)
        clims for mapping the property to a color map. These are the min and max value
        of the specified property that are mapped to 0 and 1, respectively.
        The default value is None. If set the none, the clims will be set to
        (property.min(), property.max())
    face_color : str, array-like, dict
        Color of the point marker body. Numeric color values should be RGB(A).
    face_color_cycle : np.ndarray, list
        Cycle of colors (provided as string name, RGB, or RGBA) to map to face_color if a
        categorical attribute is used color the vectors.
    face_colormap : str, napari.utils.Colormap
        Colormap to set face_color if a continuous attribute is used to set face_color.
    face_contrast_limits : None, (float, float)
        clims for mapping the property to a color map. These are the min and max value
        of the specified property that are mapped to 0 and 1, respectively.
        The default value is None. If set the none, the clims will be set to
        (property.min(), property.max())
    out_of_slice_display : bool
        If True, renders points not just in central plane but also slightly out of slice
        according to specified point marker size.
    n_dimensional : bool
        This property will soon be deprecated in favor of 'out_of_slice_display'.
        Use that instead.
    name : str
        Name of the layer.
    metadata : dict
        Layer metadata.
    scale : tuple of float
        Scale factors for the layer.
    translate : tuple of float
        Translation values for the layer.
    rotate : float, 3-tuple of float, or n-D array.
        If a float convert into a 2D rotation matrix using that value as an
        angle. If 3-tuple convert into a 3D rotation matrix, using a yaw,
        pitch, roll convention. Otherwise assume an nD rotation. Angles are
        assumed to be in degrees. They can be converted from radians with
        np.degrees if needed.
    shear : 1-D array or n-D array
        Either a vector of upper triangular values, or an nD shear matrix with
        ones along the main diagonal.
    affine : n-D array or napari.utils.transforms.Affine
        (N+1, N+1) affine transformation matrix in homogeneous coordinates.
        The first (N, N) entries correspond to a linear transform and
        the final column is a length N translation vector and a 1 or a napari
        `Affine` transform object. Applied as an extra transform on top of the
        provided scale, rotate, and shear values.
    opacity : float
        Opacity of the layer visual, between 0.0 and 1.0.
    blending : str
        One of a list of preset blending modes that determines how RGB and
        alpha values of the layer visual get mixed. Allowed values are
        {'opaque', 'translucent', and 'additive'}.
    visible : bool
        Whether the layer visual is currently being displayed.
    cache : bool
        Whether slices of out-of-core datasets should be cached upon retrieval.
        Currently, this only applies to dask arrays.
    shading : str, Shading
        Render lighting and shading on points. Options are:

        * 'none'
          No shading is added to the points.
        * 'spherical'
          Shading and depth buffer are changed to give a 3D spherical look to the points
    antialiasing: float
        Amount of antialiasing in canvas pixels.
    canvas_size_limits : tuple of float
        Lower and upper limits for the size of points in canvas pixels.
    shown : 1-D array of bool
        Whether to show each point.
    """

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
        edge_width=0.05,
        edge_width_is_relative=True,
        edge_color='dimgray',
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
        canvas_size_limits=(0, 10000),
        shown=True,
    ):
        # Save the point coordinates
        self._data = data_to_graph(data)
        self._graph_edges_view = []

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
            canvas_size_limits=canvas_size_limits,
            shown=shown,
        )

    @property
    def _points_data(self) -> np.ndarray:
        return self._data._coords

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
            self._graph_edges_view = []
        else:
            mask = np.zeros(self.data.n_allocated_nodes, dtype=bool)
            mask[self._indices_view] = True
            _, edges = self.data.edges_buffers(is_buffer_domain=True)
            self._graph_edges_view = edges[
                np.logical_and(mask[edges[:, 0]], mask[edges[:, 1]])
            ]

    @property
    def edges_coordinates(self) -> np.ndarray:
        _, edges = self.data.edges_buffers(is_buffer_domain=True)
        coords = self.data._coords[edges][..., self._slice_input.displayed]
        return coords

    @property
    def _view_edges_coordinates(self) -> np.ndarray:
        return self.data._coords[self._graph_edges_view][
            ..., self._slice_input.displayed
        ]

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

    def remove_selected(self):
        """Removes selected points if any."""
        if len(self.selected_data):
            indices = self.data._buffer2world[list(self.selected_data)]
            self.remove(indices)
            self.selected_data = set()

    def remove(self, indices: Union[np.ndarray, List[int]]) -> None:
        """Removes nodes given their indices."""
        prev_size = self.data.n_allocated_nodes
        if isinstance(indices, np.ndarray):
            indices = indices.tolist()
        indices.sort(reverse=True)
        for idx in indices:
            self.data.remove_node(idx)
        self._data_changed(prev_size)

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
