from __future__ import annotations

from types import SimpleNamespace

import dask.array as da
import numpy as np
import pytest

pytest.importorskip('lodstone', reason='requires the progressive extra')
from lodstone import Plan, Region, TileKey, Update

from napari.experimental._lodstone_loading import (
    LEVEL_LABEL_OFFSET,
    MISSING_LEVEL_LABEL,
    PlanTrace,
    _acquire_viewer_runtime,
    _camera_view,
    _level_transforms,
    _NapariTarget,
    _QtDispatcher,
    _release_viewer_runtime,
    add_lodstone_level_diagnostics,
    add_lodstone_loading_image,
    add_lodstone_loading_labels,
)


def test_viewer_layers_share_lodstone_runtime() -> None:
    viewer = SimpleNamespace()

    first = _acquire_viewer_runtime(viewer)
    second = _acquire_viewer_runtime(viewer)

    assert first is second
    _release_viewer_runtime(viewer, first)
    assert not first.closed
    _release_viewer_runtime(viewer, second)
    assert first.closed
    assert not hasattr(viewer, '_lodstone_runtime')


def test_qt_dispatcher_tolerates_teardown_before_delivery(qtbot) -> None:
    from qtpy.QtCore import QCoreApplication, QEvent

    dispatcher = _QtDispatcher()
    dispatcher.deleteLater()
    QCoreApplication.sendPostedEvents(dispatcher, QEvent.Type.DeferredDelete)

    # A late Stream.close() status callback becomes a harmless no-op.
    dispatcher.dispatch(lambda: None)


class _VirtualData:
    def __init__(self) -> None:
        self.values = np.zeros((2, 4, 4), dtype=np.uint16)
        self.loaded_chunks = set()
        self.chunk_source = {}

    def set_offset(self, key, value) -> None:
        self.values[key] = value

    def set_chunk(self, key, value, source_level) -> bool:
        self.set_offset(key, value)
        chunk_id = tuple((sl.start, sl.stop) for sl in key)
        self.loaded_chunks.add(chunk_id)
        self.chunk_source[chunk_id] = source_level
        return True


def _assert_submitted_uses_selected_geometry(loader):
    comparison = loader.bounded_plan_comparisons[-1]
    expected = (
        comparison.lodstone
        if comparison.geometry_matches
        else comparison.napari
    )
    actual = PlanTrace.from_plan(loader._submitted_plan)
    assert actual.target_level == expected.target_level
    assert frozenset(actual.tiles) == frozenset(expected.tiles)
    assert frozenset(expected.wanted) <= frozenset(actual.wanted)
    expected_wanted = frozenset(expected.wanted)
    assert {
        tile[0] for tile in actual.wanted if tile not in expected_wanted
    } <= {loader._resident_level}


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

    staged = target.stage([update])

    assert np.all(vdata.values[1] == 1)
    assert ((1, 2), (0, 4), (0, 4)) in vdata.loaded_chunks
    assert delivered == []

    target.apply(staged)

    assert delivered == [(7, vdata, [region.slices()])]


def test_target_reports_renderer_submissions() -> None:
    vdata = _VirtualData()
    delivered = []
    loader = SimpleNamespace(
        _data=[vdata],
        _resident_level=None,
        _on_resident_chunks=None,
        _on_chunks=lambda generation, level, batch: delivered.append(batch),
    )
    target = _NapariTarget(loader)
    block = np.ones((1, 4, 4), dtype=np.uint16)

    target.apply(((0, (block.shape,), ([0, 0, 0], [1, 4, 4], block)),))

    metrics = target.performance_metrics()
    assert metrics.submitted_bytes == block.nbytes
    assert metrics.presentations == 1


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

    view = _camera_view(
        viewer,
        layer,
        (10, 20, 30, 40),
        depth_span=80,
        depth_center=(10, 15, 20),
    )

    assert view.index == (9, None, None, None)
    np.testing.assert_allclose(
        view.world_to_clip @ np.array([8, 16, 24, 1]),
        (0, 0, 0.05, 1),
    )
    # Depth clipping is centered on the data, not the orbit point. This
    # keeps a panned foreground slab from falling outside the synthetic
    # Lodstone frustum while napari still renders it.
    assert (view.world_to_clip @ np.array([20, 15, 20, 1]))[2] == -0.25
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
        bounded = loader.bounded_plan_comparisons[-1]
        assert bounded.geometry_matches
        assert loader._submitted_plan is not None
        _assert_submitted_uses_selected_geometry(loader)
        qtbot.waitUntil(
            lambda: bool(loader.execution_diagnostics), timeout=10000
        )
        diagnostics = loader.execution_diagnostics[-1]
        assert diagnostics.desired_tiles == len(comparison.napari.tiles)
        assert diagnostics.wanted_tiles == len(comparison.napari.wanted)
        assert diagnostics.unique_native_chunks > 0
        assert diagnostics.source_reads > 0
        assert loader._lodstone_stream.batch_size == 1
        assert loader._lodstone_stream.cpu_cache_limit == max(
            loader._interval_max_bytes,
            loader._resident_max_bytes,
        )

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
        assert (
            loader._lodstone_target.layout(
                comparison.view, loader._lodstone_source.pyramid
            ).focus_depth_weight
            == 0.5
        )
        assert (
            loader._lodstone_target.layout(
                comparison.view, loader._lodstone_source.pyramid
            ).focus_depth_target
            == 0.5
        )
        bounded = loader.bounded_plan_comparisons[-1]
        assert bounded.geometry_matches
        assert (
            comparison.napari.target_level == comparison.lodstone.target_level
        )
        assert loader._submitted_plan is not None
        _assert_submitted_uses_selected_geometry(loader)
        # Coarse-prefilled clipmap pages refine through every available
        # intermediate level instead of waiting for one atomic fine page.
        assert {tile.level for tile in loader._submitted_plan.wanted} <= set(
            range(
                comparison.napari.target_level,
                loader._resident_level + 1,
            )
        )

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
        comparison = loader.plan_comparisons[-1]
        assert (
            comparison.napari.target_level == comparison.lodstone.target_level
        )
        _assert_submitted_uses_selected_geometry(loader)

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
        _assert_submitted_uses_selected_geometry(loader)
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


def test_lodstone_volume_keeps_full_coarse_clipmap(
    qtbot,
    make_napari_viewer,
) -> None:
    base = np.zeros((64, 64, 64), dtype=np.uint16)
    base[16:48, 16:48, 16:48] = 100
    arrays = [
        da.from_array(base, chunks=(16, 16, 16)),
        da.from_array(base[::2, ::2, ::2], chunks=(16, 16, 16)),
        da.from_array(base[::4, ::4, ::4], chunks=(16, 16, 16)),
    ]
    viewer = make_napari_viewer()
    layer = add_lodstone_loading_image(
        arrays,
        viewer=viewer,
        interval_max_bytes=32**3 * base.dtype.itemsize,
        tile_max_bytes_3d=32**3 * base.dtype.itemsize,
    )
    loader = layer.metadata['progressive_loader']
    try:
        viewer.dims.ndisplay = 3
        qtbot.waitUntil(
            lambda: layer._slice_input.ndisplay == 3,
            timeout=10000,
        )
        loader._ensure_persistent_overview()
        visual = viewer.window._qt_viewer.layer_to_visual[layer]
        node = visual._layer_node.get_node(3)

        assert node.clipmap_enabled
        assert loader._has_clipmap_overview()
        assert node._vol_shape == base.shape
        assert tuple(node._overview_texture.shape[:3]) == arrays[-1].shape
        detail_extent = layer.corner_pixels[1] - layer.corner_pixels[0] + 1
        assert np.any(detail_extent < base.shape)

        layer.locked_data_level = 1
        qtbot.waitUntil(lambda: layer.data_level == 1, timeout=10000)

        def level_one_bounds_presented():
            expected = ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
            if node._clipmap_detail_bounds == expected:
                return True
            if loader._dbuf is not None:
                loader._dbuf.present()
            return node._clipmap_detail_bounds == expected

        qtbot.waitUntil(
            level_one_bounds_presented,
            timeout=10000,
        )
    finally:
        loader.close()


def test_lodstone_volume_skips_clipmap_over_resident_budget(
    make_napari_viewer,
) -> None:
    base = np.zeros((64, 64, 64), dtype=np.uint16)
    arrays = [
        da.from_array(base, chunks=(16, 16, 16)),
        da.from_array(base[::2, ::2, ::2], chunks=(16, 16, 16)),
    ]
    viewer = make_napari_viewer()
    layer = add_lodstone_loading_image(
        arrays,
        viewer=viewer,
    )
    loader = layer.metadata['progressive_loader']
    try:
        loader._resident_max_bytes = 1024
        viewer.dims.ndisplay = 3
        loader._ensure_persistent_overview()
        node = loader._get_volume_node()

        assert node is not None
        assert not node.clipmap_enabled
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
        assert layer.dtype == base.dtype
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


def test_bounded_region_plan_is_used_when_generic_plan_differs(
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
        _assert_submitted_uses_selected_geometry(loader)
    finally:
        loader.close()
