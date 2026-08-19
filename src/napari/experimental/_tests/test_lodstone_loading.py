from __future__ import annotations

from types import SimpleNamespace

import dask.array as da
import numpy as np
from lodstone import Plan, Region, Tile, TileKey, Update

from napari.experimental._lodstone_loading import (
    LEVEL_LABEL_OFFSET,
    MISSING_LEVEL_LABEL,
    PlanComparison,
    PlanTrace,
    _camera_view,
    _level_transforms,
    _LevelDiagnosticArray,
    _NapariTarget,
    add_lodstone_level_diagnostics,
    add_lodstone_loading_image,
    add_lodstone_loading_labels,
)


class _VirtualData:
    def __init__(self) -> None:
        self.values = np.zeros((2, 4, 4), dtype=np.uint16)
        self.loaded_chunks = set()
        self.chunk_source = {}

    def set_offset(self, key, value) -> None:
        self.values[key] = value


def test_level_diagnostic_array_reads_source_and_returns_level_label() -> None:
    class RecordingArray:
        shape = (8, 8)
        dtype = np.dtype(np.uint16)
        chunks = (4, 4)

        def __init__(self) -> None:
            self.reads = []

        def __getitem__(self, key):
            self.reads.append(key)
            return np.arange(64, dtype=self.dtype).reshape(self.shape)[key]

    source = RecordingArray()
    diagnostic = _LevelDiagnosticArray(source, level=2)

    result = diagnostic[1:5, 2:7]

    assert source.reads == [(slice(1, 5), slice(2, 7))]
    assert diagnostic.shape == source.shape
    assert diagnostic.chunks == source.chunks
    assert diagnostic.fill_value == MISSING_LEVEL_LABEL
    np.testing.assert_array_equal(
        result,
        np.full((4, 5), 2 + LEVEL_LABEL_OFFSET, dtype=np.uint8),
    )


def test_target_applies_full_nd_chunk_before_render_callback() -> None:
    vdata = _VirtualData()
    delivered = []
    loader = SimpleNamespace(
        _data=[vdata],
        _on_chunks=lambda generation, level, batch: delivered.append(
            (generation, level, batch),
        ),
    )
    target = _NapariTarget(loader)
    target.generation = 7
    region = Region((1, 0, 0), (2, 4, 4))
    update = Update(
        TileKey(0, (1, 0, 0), ()),
        region,
        np.ones(region.shape, dtype=np.uint16),
        np.eye(4),
    )

    target.apply([update])

    assert np.all(vdata.values[1] == 1)
    assert ((1, 2), (0, 4), (0, 4)) in vdata.loaded_chunks
    assert delivered == [(7, vdata, [region.slices()])]


def test_plan_trace_compares_geometry_not_planner_specific_keys() -> None:
    region = Region((0, 0), (8, 8))
    left = Tile(TileKey(0, (0, 0), ()), region, 0.0)
    right = Tile(TileKey(0, (99, 99), ()), region, 42.0)
    first = Plan((left,), frozenset({left.key}), 0, (left,))
    second = Plan((right,), frozenset({right.key}), 0, (right,))

    trace = PlanTrace.from_plan(first)
    comparison = PlanComparison(
        SimpleNamespace(),
        trace,
        PlanTrace.from_plan(second),
    )

    assert comparison.matches
    assert trace.tiles == ((0, (0, 0), (8, 8), 0),)
    assert trace.wanted == trace.tiles


def test_level_transforms_include_downsampling_and_layer_transform() -> None:
    layer_matrix = np.eye(3)
    layer_matrix[0, 0] = 3
    layer_matrix[1, 1] = 5
    layer_matrix[:-1, -1] = (7, 11)
    layer = SimpleNamespace(
        _data_to_world=SimpleNamespace(affine_matrix=layer_matrix),
    )
    data = SimpleNamespace(ndim=2, _scale_factors=((1, 1), (2, 4)))

    finest, coarse = _level_transforms(layer, data)

    np.testing.assert_allclose(finest, layer_matrix)
    np.testing.assert_allclose(
        coarse,
        np.array([[6, 0, 7], [0, 20, 11], [0, 0, 1]]),
    )


def test_camera_view_captures_2d_viewport_center_and_zoom() -> None:
    viewer = SimpleNamespace(
        canvas=SimpleNamespace(size=(200, 400)),
        camera=SimpleNamespace(center=(0, 10, 20), zoom=4),
        dims=SimpleNamespace(point=(0, 0)),
    )
    layer = SimpleNamespace(
        _slice_input=SimpleNamespace(displayed=(0, 1)),
        world_to_data=lambda point: point,
    )

    view = _camera_view(viewer, layer, (100, 100), depth_span=1)

    assert view.viewport == (200, 400)
    np.testing.assert_allclose(
        view.world_to_clip @ np.array([10, 20, 0, 1]),
        (0, 0, 0, 1),
    )
    # Four pixels per world unit on both canvas axes.
    assert view.world_to_clip[0, 0] * view.viewport[0] / 2 == 4
    assert view.world_to_clip[1, 1] * view.viewport[1] / 2 == 4


def test_camera_view_captures_3d_orientation_and_hidden_index() -> None:
    viewer = SimpleNamespace(
        canvas=SimpleNamespace(size=(200, 400)),
        camera=SimpleNamespace(
            center=(8, 16, 24),
            zoom=4,
            view_direction=(-1, 0, 0),
            up_direction=(0, -1, 0),
        ),
        dims=SimpleNamespace(point=(99, 8, 16, 24)),
    )
    layer = SimpleNamespace(
        _slice_input=SimpleNamespace(displayed=(1, 2, 3)),
        world_to_data=lambda point: point,
    )

    view = _camera_view(viewer, layer, (10, 20, 30, 40), depth_span=80)

    assert view.index == (9, None, None, None)
    np.testing.assert_allclose(
        view.world_to_clip @ np.array([8, 16, 24, 1]),
        (0, 0, 0, 1),
    )
    # Moving with the view direction moves from the camera into the scene.
    assert (view.world_to_clip @ np.array([7, 16, 24, 1]))[2] > 0


def test_real_fetch_pass_records_napari_and_lodstone_plans(
    qtbot,
    make_napari_viewer,
) -> None:
    base = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64)
    arrays = [
        da.from_array(base, chunks=(16, 16)),
        da.from_array(base[::2, ::2], chunks=(16, 16)),
        da.from_array(base[::4, ::4], chunks=(16, 16)),
    ]
    viewer = make_napari_viewer()
    layer = add_lodstone_loading_image(arrays, viewer=viewer)
    loader = layer.metadata['progressive_loader']
    try:
        qtbot.waitUntil(lambda: bool(loader._plan_comparisons), timeout=10000)
        comparison = loader.plan_comparisons[-1]

        assert comparison.view.viewport == viewer.canvas.size
        assert comparison.view.displayed_axes == tuple(
            layer._slice_input.displayed
        )
        assert comparison.napari.tiles
        assert comparison.lodstone.tiles
        assert comparison.geometry_matches
        assert loader._submitted_plan is not None
        assert PlanTrace.from_plan(loader._submitted_plan) == comparison.napari
        qtbot.waitUntil(
            lambda: bool(loader.execution_diagnostics), timeout=10000
        )
        diagnostics = loader.execution_diagnostics[-1]
        assert diagnostics.desired_tiles == len(comparison.napari.tiles)
        assert diagnostics.wanted_tiles == len(comparison.napari.wanted)
        assert diagnostics.unique_native_chunks > 0
        assert diagnostics.source_reads > 0

        comparison_count = len(loader.plan_comparisons)
        viewer.scene.camera.center = (16, 48)
        viewer.scene.camera.zoom = 20
        # Headless tests do not have a canvas draw to refresh corner_pixels.
        layer.corner_pixels = np.array([[0, 33], [36, 63]])
        loader._data[0].loaded_chunks.clear()
        loader._active = None
        loader._check()
        qtbot.waitUntil(
            lambda: len(loader.plan_comparisons) > comparison_count,
            timeout=10000,
        )
        assert loader.plan_comparisons[-1].geometry_matches

    finally:
        loader.close()


def test_real_3d_fetch_pass_records_napari_and_lodstone_plans(
    qtbot,
    make_napari_viewer,
) -> None:
    base = np.zeros((32, 32, 32), dtype=np.uint16)
    arrays = [
        da.from_array(base, chunks=(8, 8, 8)),
        da.from_array(base[::2, ::2, ::2], chunks=(8, 8, 8)),
        da.from_array(base[::4, ::4, ::4], chunks=(8, 8, 8)),
    ]
    viewer = make_napari_viewer()
    viewer.dims.ndisplay = 3
    layer = add_lodstone_loading_image(arrays, viewer=viewer)
    loader = layer.metadata['progressive_loader']
    try:
        qtbot.waitUntil(lambda: bool(loader.plan_comparisons), timeout=10000)
        comparison = loader.plan_comparisons[-1]

        assert comparison.view.displayed_axes == tuple(
            layer._slice_input.displayed
        )
        assert comparison.napari.tiles
        assert comparison.lodstone.tiles
        assert comparison.geometry_matches
        assert loader._submitted_plan is not None
        assert PlanTrace.from_plan(loader._submitted_plan) == comparison.napari

        comparison_count = len(loader.plan_comparisons)
        viewer.scene.camera.angles = (25, 35, 10)
        viewer.scene.camera.zoom = 20
        loader._data[0].loaded_chunks.clear()
        loader._active = None
        loader._check()
        qtbot.waitUntil(
            lambda: len(loader.plan_comparisons) > comparison_count,
            timeout=10000,
        )
        assert loader.plan_comparisons[-1].geometry_matches

        comparison_count = len(loader.plan_comparisons)
        viewer.scene.camera.zoom = 1
        loader._data[1].loaded_chunks.clear()
        loader._active = None
        loader._check()
        qtbot.waitUntil(
            lambda: len(loader.plan_comparisons) > comparison_count,
            timeout=10000,
        )
        comparison = loader.plan_comparisons[-1]
        assert comparison.napari.target_level == 1
        assert comparison.lodstone.target_level == 1
        assert comparison.geometry_matches
    finally:
        loader.close()


def test_plan_geometry_matches_after_hidden_axis_step(
    qtbot,
    make_napari_viewer,
) -> None:
    base = np.zeros((3, 64, 64), dtype=np.uint16)
    arrays = [
        da.from_array(base, chunks=(1, 16, 16)),
        da.from_array(base[:, ::2, ::2], chunks=(1, 16, 16)),
        da.from_array(base[:, ::4, ::4], chunks=(1, 16, 16)),
    ]
    viewer = make_napari_viewer()
    layer = add_lodstone_loading_image(arrays, viewer=viewer)
    loader = layer.metadata['progressive_loader']
    try:
        qtbot.waitUntil(lambda: bool(loader.plan_comparisons), timeout=10000)
        comparison_count = len(loader.plan_comparisons)

        viewer.dims.set_point(0, 2)
        qtbot.waitUntil(
            lambda: len(loader.plan_comparisons) > comparison_count,
            timeout=10000,
        )
        comparison = loader.plan_comparisons[-1]

        assert comparison.view.index == (2, None, None)
        assert comparison.geometry_matches
    finally:
        loader.close()


def test_lodstone_labels_use_shared_progressive_loader(
    qtbot,
    make_napari_viewer,
) -> None:
    base = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64)
    arrays = [
        da.from_array(base, chunks=(16, 16)),
        da.from_array(base[::2, ::2], chunks=(16, 16)),
        da.from_array(base[::4, ::4], chunks=(16, 16)),
    ]
    viewer = make_napari_viewer()
    layer = add_lodstone_loading_labels(arrays, viewer=viewer, fill_value=9)
    loader = layer.metadata['progressive_loader']
    try:
        qtbot.waitUntil(lambda: bool(loader.plan_comparisons), timeout=10000)

        assert layer._type_string == 'labels'
        assert all(level.fill_value == 9 for level in loader._data)
        assert loader.plan_comparisons[-1].geometry_matches
    finally:
        loader.close()


def test_level_diagnostic_layer_marks_loaded_blocks_by_source_level(
    qtbot,
    make_napari_viewer,
) -> None:
    base = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64)
    arrays = [
        da.from_array(base, chunks=(16, 16)),
        da.from_array(base[::2, ::2], chunks=(16, 16)),
        da.from_array(base[::4, ::4], chunks=(16, 16)),
    ]
    viewer = make_napari_viewer()
    layer = add_lodstone_level_diagnostics(
        arrays, viewer=viewer, name='level coverage'
    )
    loader = layer.metadata['progressive_loader']
    try:
        qtbot.waitUntil(
            lambda: bool(loader.execution_diagnostics), timeout=10000
        )

        assert layer._type_string == 'labels'
        assert layer.name == 'level coverage'
        assert layer.metadata['level_diagnostic_labels'] == {
            1: 'missing',
            2: 'L0',
            3: 'L1',
            4: 'L2',
        }
        assert layer.metadata['level_diagnostic_legend'] == {
            'missing': 'magenta',
            'L0': 'green',
            'L1': 'yellow',
            'L2': 'orange',
        }
        assert loader._debug_overlay is not None
        assert all(
            vdata.fill_value == MISSING_LEVEL_LABEL for vdata in loader._data
        )
        for level, vdata in enumerate(loader._data):
            for chunk_id in vdata.loaded_chunks:
                key = tuple(
                    slice(
                        start - vdata.translate[axis],
                        stop - vdata.translate[axis],
                    )
                    for axis, (start, stop) in enumerate(chunk_id)
                )
                assert np.all(
                    vdata.hyperslice[key] == level + LEVEL_LABEL_OFFSET
                )
    finally:
        loader.close()


def test_lodstone_preserves_rectilinear_chunk_geometry(
    qtbot,
    make_napari_viewer,
) -> None:
    base = np.arange(35 * 37, dtype=np.uint16).reshape(35, 37)
    arrays = [
        da.from_array(base, chunks=((8, 11, 16), (9, 13, 15))),
        da.from_array(base[::2, ::2], chunks=((7, 11), (6, 13))),
    ]
    viewer = make_napari_viewer()
    layer = add_lodstone_loading_image(arrays, viewer=viewer)
    loader = layer.metadata['progressive_loader']
    try:
        qtbot.waitUntil(lambda: bool(loader.plan_comparisons), timeout=10000)

        assert loader._lodstone_source.pyramid.levels[0].chunk_grid == (
            (8, 11, 16),
            (9, 13, 15),
        )
        assert loader.plan_comparisons[-1].geometry_matches
    finally:
        loader.close()


def test_napari_plan_remains_authoritative_when_generic_plan_differs(
    qtbot,
    make_napari_viewer,
) -> None:
    base = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64)
    arrays = [
        da.from_array(base, chunks=(16, 16)),
        da.from_array(base[::2, ::2], chunks=(16, 16)),
    ]
    viewer = make_napari_viewer()
    layer = add_lodstone_loading_image(arrays, viewer=viewer)
    loader = layer.metadata['progressive_loader']
    try:
        qtbot.waitUntil(lambda: bool(loader.plan_comparisons), timeout=10000)
        comparison_count = len(loader.plan_comparisons)
        loader._shared_planners[2] = SimpleNamespace(
            plan=lambda *_args, **_kwargs: Plan((), frozenset(), 0, ()),
        )
        loader._data[0].loaded_chunks.clear()
        loader._active = None

        loader._check()
        qtbot.waitUntil(
            lambda: len(loader.plan_comparisons) > comparison_count,
            timeout=10000,
        )

        comparison = loader.plan_comparisons[-1]
        assert not comparison.geometry_matches
        assert comparison.napari.tiles
        assert not comparison.lodstone.tiles
        assert loader._submitted_plan is not None
        assert PlanTrace.from_plan(loader._submitted_plan) == comparison.napari
    finally:
        loader.close()
