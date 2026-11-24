"""Progressive multiscale surfaces streamed by Lodstone into napari.

Lodstone's ``MeshStream`` owns level selection from the camera, frustum
culling, chunk reads, decoded caching and stale-phase rejection. napari owns
the ``Surface`` layer, the camera and the Qt thread on which geometry is
replaced. Each publication is one complete level: Lodstone never mixes
resolutions within a displayed phase, so there are no seams between levels.

Mesh sources report coordinates in XYZ order (Lodstone's convention). napari
orders spatial axes ZYX, so vertices are reversed on publication and the
camera is mapped back into the source frame for planning.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
from lodstone import Runtime, View
from lodstone.mesh import MeshData, MeshKey, MeshSource, MeshStream

from napari.experimental._lodstone_loading import (
    _acquire_viewer_runtime,
    _camera_view,
    _QtDispatcher,
    _release_viewer_runtime,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from napari.layers import Surface

LOGGER = logging.getLogger(__name__)

DEFAULT_MESH_MEMORY_LIMIT = 128 << 20

# Source XYZ (homogeneous) to napari ZYX (homogeneous).
_XYZ_TO_ZYX = np.array(
    [
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def _source_bounds(source: MeshSource) -> tuple[np.ndarray, np.ndarray]:
    """Return the XYZ bounds that enclose every chunk of every level."""
    lows, highs = [], []
    for level in source.levels:
        for chunk in level.chunks:
            low, high = np.asarray(chunk.bounds, dtype=float)
            lows.append(low)
            highs.append(high)
    if not lows:
        raise ValueError('a progressive surface needs at least one mesh chunk')
    return np.min(lows, axis=0), np.max(highs, axis=0)


def _source_to_data(axis_order: str) -> np.ndarray:
    if axis_order == 'xyz':
        return _XYZ_TO_ZYX
    if axis_order == 'zyx':
        return np.eye(4)
    raise ValueError("axis_order must be 'xyz' or 'zyx'")


def _combine(
    chunks: Sequence[tuple[MeshKey, MeshData]], source_to_data: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate self-contained fragments into one napari vertex/face set."""
    vertices, faces = [], []
    offset = 0
    for _key, data in chunks:
        if not len(data.faces):
            continue
        vertices.append(np.asarray(data.vertices, dtype=np.float32))
        faces.append(np.asarray(data.faces, dtype=np.int64) + offset)
        offset += len(data.vertices)
    if not faces:
        return np.empty((0, 3), np.float32), np.empty((0, 3), np.int64)
    combined, combined_faces = _weld(
        np.concatenate(vertices), np.concatenate(faces)
    )
    # A permutation (or identity) keeps float32 exact: no matrix product.
    order = [int(np.flatnonzero(row[:3])[0]) for row in source_to_data[:3]]
    return combined[:, order], combined_faces


def _weld(
    vertices: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Merge vertices that chunks duplicate along their shared boundaries.

    Each chunk carries its own copy of the vertices on its boundary. Left
    apart, smooth shading averages normals per chunk and every chunk edge
    shows as a crease. Copies are bitwise equal, so exact matching suffices.
    """
    vertices = np.ascontiguousarray(vertices)
    rows = vertices.view(np.dtype((np.void, vertices.dtype.itemsize * 3)))
    _, first, inverse = np.unique(
        rows.ravel(), return_index=True, return_inverse=True
    )
    if len(first) == len(vertices):
        return vertices, faces
    return vertices[first], inverse.reshape(-1)[faces]


def _placeholder(
    low: np.ndarray, high: np.ndarray, source_to_data: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """A zero-area triangle spanning the bounds, so the layer has its extent.

    napari cannot hold an empty surface, and the camera should frame the
    whole object before the first level arrives.
    """
    corners = np.array([low, high, low], dtype=float)
    points = (np.c_[corners, np.ones(3)] @ source_to_data.T)[:, :3]
    return points.astype(np.float32), np.array([[0, 1, 2]])


class NapariSurfaceTarget:
    """Lodstone ``MeshTarget`` that replaces a napari ``Surface``'s geometry.

    ``replace`` runs on the dispatch thread (the Qt thread in a viewer). An
    empty publication keeps the bounds placeholder rather than an empty
    surface, which napari cannot represent.
    """

    def __init__(
        self,
        layer: Surface,
        source_to_data: np.ndarray,
        placeholder: tuple[np.ndarray, np.ndarray],
    ) -> None:
        self.layer = layer
        self.source_to_data = source_to_data
        self.placeholder = placeholder
        self.level: int | None = None
        self.vertices = 0
        self.faces = 0
        self.publications = 0

    def replace(
        self, level: int, chunks: Sequence[tuple[MeshKey, MeshData]]
    ) -> None:
        vertices, faces = _combine(chunks, self.source_to_data)
        if not len(faces):
            vertices, faces = self.placeholder
        # Record the publication first: ``layer.data`` emits synchronously and
        # listeners read the target to describe what is shown.
        self.level = level
        self.vertices, self.faces = len(vertices), len(faces)
        self.publications += 1
        self.layer.data = (vertices, faces)


class LodstoneSurfaceLoader:
    """Connect a napari camera to a Lodstone ``MeshStream`` for one layer."""

    def __init__(
        self,
        layer: Surface,
        source: MeshSource,
        viewer,
        *,
        axis_order: str = 'xyz',
        memory_limit: int = DEFAULT_MESH_MEMORY_LIMIT,
        pixel_error: float = 2.0,
        dispatch: Callable[[Callable[[], None]], None] | None = None,
        runtime: Runtime | None = None,
    ) -> None:
        self.layer = layer
        self.source = source
        self._viewer = viewer
        self.source_to_data = _source_to_data(axis_order)
        low, high = _source_bounds(source)
        self._center = (low + high) / 2
        self._diagonal = float(np.linalg.norm(high - low))
        self.target = NapariSurfaceTarget(
            layer,
            self.source_to_data,
            _placeholder(low, high, self.source_to_data),
        )
        self._dispatcher = None
        if dispatch is None:
            self._dispatcher = _QtDispatcher()
            dispatch = self._dispatcher.dispatch
        self._owns_runtime = runtime is None
        self._runtime = (
            _acquire_viewer_runtime(viewer) if runtime is None else runtime
        )
        self.stream = MeshStream(
            source,
            self.target,
            dispatch=dispatch,
            runtime=self._runtime,
            memory_limit=memory_limit,
            pixel_error=pixel_error,
        )
        self._closed = False
        self._last_matrix: np.ndarray | None = None
        self._connections = [
            (viewer.scene.camera.events.center, self.refresh),
            (viewer.scene.camera.events.zoom, self.refresh),
            (viewer.scene.camera.events.angles, self.refresh),
            (viewer.dims.events.ndisplay, self.refresh),
            (viewer.canvas.events.size, self.refresh),
            (viewer.layers.events.removed, self._on_layer_removed),
        ]
        for emitter, callback in self._connections:
            emitter.connect(callback)

    def view(self) -> View | None:
        """The current camera in the source frame, or ``None`` outside 3D."""
        if self._viewer.dims.ndisplay != 3:
            return None
        displayed = tuple(self.layer._slice_input.displayed)
        if len(displayed) != 3:
            return None
        world_center = self.layer.data_to_world(
            (np.r_[self._center, 1.0] @ self.source_to_data.T)[:3]
        )
        world_view = _camera_view(
            self._viewer,
            self.layer,
            (1,) * self.layer.ndim,
            depth_span=2.0 * max(self._diagonal, 1.0) * _max_scale(self.layer),
            depth_center=np.asarray(world_center, dtype=float)[-3:],
        )
        data_to_world = np.asarray(
            self.layer._data_to_world.affine_matrix, dtype=float
        )
        matrix = world_view.world_to_clip @ data_to_world @ self.source_to_data
        return View((0, 1, 2), (None,) * 3, world_view.viewport, matrix)

    def refresh(self, event=None) -> None:
        """Ask Lodstone for the level and chunks the current camera needs."""
        if self._closed:
            return
        view = self.view()
        if view is None:
            # 2D slices of a surface keep the last 3D phase, or start coarse.
            if (
                self.target.level is None
                and self.stream.requested_level is None
            ):
                self._request_coarsest()
            return
        if (
            self._last_matrix is not None
            and np.array_equal(self._last_matrix, view.world_to_clip)
            and self.stream.error is None
        ):
            return
        self._last_matrix = np.array(view.world_to_clip)
        self.stream.update(view)

    def _request_coarsest(self) -> None:
        low, high = _source_bounds(self.source)
        center, half = (
            (low + high) / 2,
            max(float(np.max(high - low)) / 2, 1e-9),
        )
        # A view that fits the whole object, whatever the host camera is.
        matrix = np.eye(4)
        matrix[:3, :3] /= half
        matrix[:3, 3] = -center / half
        coarsest = len(self.source.levels) - 1
        self.stream.update(
            View((0, 1, 2), (None,) * 3, (1, 1), matrix), level=coarsest
        )

    def snapshot(self) -> dict[str, Any]:
        """Stream and target state for diagnostics and tests."""
        snapshot = self.stream.snapshot()
        return {
            'stream': snapshot,
            'level': self.target.level,
            'vertices': self.target.vertices,
            'faces': self.target.faces,
            'publications': self.target.publications,
        }

    def _on_layer_removed(self, event) -> None:
        if event.value is self.layer:
            self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for emitter, callback in self._connections:
            emitter.disconnect(callback)
        self.stream.close()
        if self._owns_runtime:
            _release_viewer_runtime(self._viewer, self._runtime)


def _max_scale(layer) -> float:
    linear = np.asarray(layer._data_to_world.affine_matrix, dtype=float)[
        :-1, :-1
    ]
    return float(max(np.max(np.linalg.norm(linear, axis=0)), 1e-12))


def add_lodstone_surface(
    source: MeshSource,
    viewer=None,
    *,
    axis_order: str = 'xyz',
    memory_limit: int = DEFAULT_MESH_MEMORY_LIMIT,
    pixel_error: float = 2.0,
    name: str | None = None,
    dispatch: Callable[[Callable[[], None]], None] | None = None,
    runtime: Runtime | None = None,
    **layer_kwargs: Any,
) -> Surface:
    """Add a multiscale surface whose level follows the 3D camera.

    Parameters
    ----------
    source : lodstone.mesh.MeshSource
        Mesh levels ordered finest to coarsest, for example a
        ``lodstone.sources.ZarrVectorsMeshSource``.
    viewer : napari.Viewer or ViewerModel, optional
        Created if not given.
    axis_order : {'xyz', 'zyx'}
        Coordinate order of the source vertices. Lodstone sources use XYZ.
    memory_limit : int
        Bytes of decoded geometry Lodstone may cache and publish at once.
    pixel_error : float
        Largest projected level error, in screen pixels, before refining.
    dispatch : callable, optional
        Runs geometry replacement on the host thread. Defaults to the Qt
        thread; pass ``lambda callback: callback()`` without a Qt loop.
    **layer_kwargs
        Passed to ``napari.layers.Surface`` (``scale``, ``colormap``...).

    Returns
    -------
    napari.layers.Surface
        The layer; its loader is ``layer.metadata['lodstone_loader']``.
    """
    if viewer is None:
        from napari import Viewer

        viewer = Viewer()
    from napari.layers import Surface

    source_to_data = _source_to_data(axis_order)
    low, high = _source_bounds(source)
    layer = Surface(
        _placeholder(low, high, source_to_data), name=name, **layer_kwargs
    )
    viewer.add_layer(layer)
    loader = LodstoneSurfaceLoader(
        layer,
        source,
        viewer,
        axis_order=axis_order,
        memory_limit=memory_limit,
        pixel_error=pixel_error,
        dispatch=dispatch,
        runtime=runtime,
    )
    layer.metadata['lodstone_loader'] = loader
    loader.refresh()
    return layer
