"""Progressive multiscale surfaces streamed by Lodstone."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip('lodstone')

from lodstone.mesh import MeshChunk, MeshData, MeshLevel

from napari.components import ViewerModel
from napari.experimental._lodstone_meshes import (
    _XYZ_TO_ZYX,
    _combine,
    add_lodstone_surface,
)


def _grid(n: int, size: float, offset: tuple[float, float, float]):
    """A flat n x n quad sheet in the XY plane, as one self-contained chunk."""
    xs = np.linspace(0, size, n + 1)
    x, y = np.meshgrid(xs, xs, indexing='ij')
    vertices = np.c_[x.ravel(), y.ravel(), np.zeros(x.size)] + offset
    faces = []
    for i in range(n):
        for j in range(n):
            a = i * (n + 1) + j
            b, c, d = a + 1, a + n + 1, a + n + 2
            faces += [[a, b, d], [a, d, c]]
    return MeshData(vertices.astype(np.float32), np.asarray(faces))


class SheetSource:
    """Two chunks side by side in X; finer levels have more triangles."""

    def __init__(self, resolutions=(32, 8, 2), size=100.0, depth=10.0):
        self.size = size
        self._resolutions = resolutions
        bounds = [
            ((0.0, 0.0, 0.0), (size, size, depth)),
            ((size, 0.0, 0.0), (2 * size, size, depth)),
        ]
        self.levels = tuple(
            MeshLevel(
                tuple(
                    MeshChunk((i,), bound) for i, bound in enumerate(bounds)
                ),
                error=size / n,
            )
            for n in resolutions
        )
        self.reads: list[tuple[int, tuple[int, ...]]] = []

    async def read(self, level, key):
        self.reads.append((level, key))
        n = self._resolutions[level]
        return _grid(n, self.size, (key[0] * self.size, 0.0, 0.0))


def _inline(callback):
    callback()


@pytest.fixture
def viewer():
    viewer = ViewerModel()
    yield viewer
    viewer.layers.clear()


def _loader(layer):
    return layer.metadata['lodstone_loader']


def test_combine_reverses_xyz_and_offsets_faces():
    first = MeshData(
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], np.float32),
        np.array([[0, 1, 2]]),
    )
    second = MeshData(
        np.array([[10, 11, 12], [13, 14, 15], [16, 17, 18]], np.float32),
        np.array([[0, 1, 2]]),
    )
    empty = MeshData(np.empty((0, 3), np.float32), np.empty((0, 3), int))
    vertices, faces = _combine(
        [(('m', (0,)), first), (('m', (1,)), empty), (('m', (2,)), second)],
        _XYZ_TO_ZYX,
    )
    np.testing.assert_array_equal(vertices[0], [3, 2, 1])
    np.testing.assert_array_equal(vertices[3], [12, 11, 10])
    np.testing.assert_array_equal(faces, [[0, 1, 2], [3, 4, 5]])


def test_combine_welds_vertices_shared_across_chunks():
    # Two triangles of one quad, split into chunks that each copy the
    # shared edge. Unwelded, smooth shading creases along that edge.
    left = MeshData(
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32),
        np.array([[0, 1, 2]]),
    )
    right = MeshData(
        np.array([[1, 0, 0], [1, 1, 0], [0, 1, 0]], np.float32),
        np.array([[0, 1, 2]]),
    )
    vertices, faces = _combine(
        [(('m', (0,)), left), (('m', (1,)), right)], _XYZ_TO_ZYX
    )
    assert len(vertices) == 4
    corners = vertices[faces][..., ::-1]  # back to XYZ
    np.testing.assert_array_equal(
        corners,
        [
            [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
            [[1, 0, 0], [1, 1, 0], [0, 1, 0]],
        ],
    )


def test_title_listeners_see_the_published_level(viewer):
    viewer.dims.ndisplay = 3
    source = SheetSource()
    layer = add_lodstone_surface(source, viewer, dispatch=_inline)
    loader = _loader(layer)
    seen = []
    layer.events.data.connect(
        lambda _: seen.append((loader.target.faces, len(layer.data[1])))
    )
    loader.stream.wait()
    viewer.camera.zoom *= 64
    loader.stream.wait()
    assert seen
    assert all(target == shown for target, shown in seen)


def test_2d_display_starts_with_the_coarsest_level(viewer):
    source = SheetSource()
    layer = add_lodstone_surface(source, viewer, dispatch=_inline)
    loader = _loader(layer)
    loader.stream.wait()
    assert viewer.dims.ndisplay == 2
    assert loader.target.level == len(source.levels) - 1
    # Two 2 x 2 sheets: 8 triangles each.
    assert len(layer.data[1]) == 16
    # Vertices are ZYX in napari: the sheet lies in the Z == 0 plane.
    np.testing.assert_array_equal(layer.data[0][:, 0], 0)
    assert layer.data[0][:, 2].max() == pytest.approx(2 * source.size)
    loader.close()


def test_zoom_in_refines_and_zoom_out_coarsens(viewer):
    source = SheetSource()
    layer = add_lodstone_surface(source, viewer, dispatch=_inline)
    loader = _loader(layer)
    loader.stream.wait()
    viewer.dims.ndisplay = 3
    viewer.scene.camera.center = (5.0, 50.0, 100.0)

    viewer.scene.camera.zoom = 0.02
    loader.stream.wait()
    coarse = loader.target.level
    # Level errors are 100 / n world units; zoom is screen pixels per unit.
    assert coarse == 2

    viewer.scene.camera.zoom = 1.0
    loader.stream.wait()
    fine = loader.target.level
    assert fine == 0
    assert len(layer.data[1]) > 16

    viewer.scene.camera.zoom = 0.02
    loader.stream.wait()
    assert loader.target.level == coarse
    loader.close()


def test_frustum_culling_drops_offscreen_chunks(viewer):
    source = SheetSource(resolutions=(4, 2))
    layer = add_lodstone_surface(source, viewer, dispatch=_inline)
    loader = _loader(layer)
    loader.stream.wait()
    viewer.dims.ndisplay = 3
    viewer.scene.camera.angles = (0.0, 0.0, 90.0)
    # Look closely at the middle of chunk 0 (X in [0, 100]).
    viewer.scene.camera.center = (5.0, 50.0, 50.0)
    viewer.scene.camera.zoom = 20.0
    loader.stream.wait()
    assert loader.target.level == 0
    displayed = {key[1] for key in loader.stream.displayed_keys}
    assert displayed == {(0,)}
    # Only chunk 0 vertices (napari X is the last axis).
    assert layer.data[0][:, 2].max() <= source.size
    loader.close()


def test_layer_scale_maps_the_camera_back_to_the_source(viewer):
    source = SheetSource(resolutions=(4, 2))
    layer = add_lodstone_surface(
        source, viewer, dispatch=_inline, scale=(1.0, 1.0, 10.0)
    )
    loader = _loader(layer)
    loader.stream.wait()
    viewer.dims.ndisplay = 3
    viewer.scene.camera.angles = (0.0, 0.0, 90.0)
    # World X is ten times source X: world 500 is source X == 50, chunk 0.
    viewer.scene.camera.center = (5.0, 50.0, 500.0)
    viewer.scene.camera.zoom = 2.0
    loader.stream.wait()
    assert {key[1] for key in loader.stream.displayed_keys} == {(0,)}
    loader.close()


def test_removing_the_layer_closes_the_stream(viewer):
    source = SheetSource()
    layer = add_lodstone_surface(source, viewer, dispatch=_inline)
    loader = _loader(layer)
    loader.stream.wait()
    viewer.layers.remove(layer)
    assert loader.stream.snapshot().state == 'closed'
    reads = len(source.reads)
    viewer.scene.camera.zoom = 50.0
    assert len(source.reads) == reads


def test_zarr_vectors_levels_stream_into_napari(viewer, tmp_path):
    pytest.importorskip('zarr_vectors')
    from lodstone.biological_mesh import write_surface_levels
    from lodstone.sources.zarr_vectors import ZarrVectorsMeshSource

    def sheet(n):
        data = _grid(n, 16.0, (0.0, 0.0, 0.0))
        return [(np.asarray(data.vertices), np.asarray(data.faces), 1)]

    root = write_surface_levels(tmp_path, [sheet(16), sheet(4)])
    source = ZarrVectorsMeshSource(root, level_errors=(1.0, 4.0))
    try:
        layer = add_lodstone_surface(source, viewer, dispatch=_inline)
        loader = _loader(layer)
        loader.stream.wait()
        assert loader.target.level == 1
        viewer.dims.ndisplay = 3
        viewer.scene.camera.center = (0.0, 8.0, 8.0)
        viewer.scene.camera.zoom = 60.0
        loader.stream.wait()
        assert loader.target.level == 0
        assert len(layer.data[1]) == 2 * 16 * 16
        loader.close()
    finally:
        source.close()
