import dask.array as da
import numpy as np
import pytest

from napari.experimental._progressive_loading import (
    ProgressiveLoader,
    add_progressive_loading_image,
    chunk_priority_2D,
    chunk_priority_3D,
    chunk_slices,
)
from napari.experimental._virtual_data import VirtualData


@pytest.fixture
def multiscale_arrays():
    """A small in-memory multiscale pyramid backed by dask."""
    base = np.random.default_rng(0).integers(
        1, 255, size=(256, 256), dtype=np.uint8
    )
    levels = [base, base[::2, ::2].copy(), base[::4, ::4].copy()]
    return [da.from_array(level, chunks=(32, 32)) for level in levels]


# ---------- chunk geometry ----------


def test_chunk_slices_full(multiscale_arrays):
    vdata = VirtualData(multiscale_arrays[0])
    slices = chunk_slices(vdata)
    assert len(slices) == 2
    assert len(slices[0]) == 256 // 32
    assert slices[0][0] == slice(0, 32)
    assert slices[0][-1] == slice(224, 256)


def test_chunk_slices_interval(multiscale_arrays):
    vdata = VirtualData(multiscale_arrays[0])
    interval = ((40, 0), (100, 32))
    slices = chunk_slices(vdata, interval=interval)
    assert slices[0] == [slice(32, 64), slice(64, 96), slice(96, 128)]
    assert slices[1] == [slice(0, 32)]


def test_chunk_slices_accepts_raw_arrays(multiscale_arrays):
    slices = chunk_slices(multiscale_arrays[1])
    assert len(slices[0]) == 128 // 32


def test_chunk_priority_2d_center_first(multiscale_arrays):
    vdata = VirtualData(multiscale_arrays[0])
    keys = chunk_slices(vdata)
    queue = chunk_priority_2D(keys, (0, 0), (256, 256))
    assert len(queue) == 64
    first_center = np.array([(sl.start + sl.stop) / 2 for sl in queue[0]])
    last_center = np.array([(sl.start + sl.stop) / 2 for sl in queue[-1]])
    view_center = np.array([128, 128])
    assert np.linalg.norm(first_center - view_center) <= np.linalg.norm(
        last_center - view_center
    )


def test_chunk_priority_2d_empty():
    assert chunk_priority_2D([[], []], (0, 0), (0, 0)) == []


def test_chunk_priority_3d_orders_by_depth():
    arr = da.zeros((64, 64, 64), chunks=(16, 16, 16), dtype=np.uint8)
    vdata = VirtualData(arr)
    keys = chunk_slices(vdata)
    queue = chunk_priority_3D(
        keys,
        (0, 0, 0),
        (64, 64, 64),
        camera_center=(32, 32, 32),
        view_direction=(1, 0, 0),
        zoom=1.0,
    )
    assert len(queue) == 64
    # chunks closer to the camera (small z) and near the center line first
    first = queue[0]
    assert first[0].start < 32


# ---------- viewer integration ----------


def _wait_for_idle_loader(qtbot, loader, timeout=30000):
    """Wait until the loader has no in-flight fetch workers."""

    def idle():
        return loader._worker is None and loader._resident_worker is None

    qtbot.waitUntil(idle, timeout=timeout)


def test_add_progressive_loading_image(
    qtbot, make_napari_viewer, multiscale_arrays
):
    viewer = make_napari_viewer()
    layer = add_progressive_loading_image(multiscale_arrays, viewer=viewer)

    # a single multiscale layer; the layer list is not polluted
    assert len(viewer.layers) == 1
    assert layer.multiscale
    assert len(layer.data) == len(multiscale_arrays)

    loader = layer.metadata['progressive_loader']
    assert isinstance(loader, ProgressiveLoader)
    _wait_for_idle_loader(qtbot, loader)

    # the coarsest level is fully resident with real data
    coarsest = loader._data[len(loader._data) - 1]
    np.testing.assert_array_equal(
        coarsest.hyperslice, np.asarray(multiscale_arrays[-1])
    )


def test_progressive_loading_data_matches_source(
    qtbot, make_napari_viewer, multiscale_arrays
):
    viewer = make_napari_viewer()
    layer = add_progressive_loading_image(multiscale_arrays, viewer=viewer)
    loader = layer.metadata['progressive_loader']
    _wait_for_idle_loader(qtbot, loader)

    level = layer.data_level
    vdata = loader._data[level]
    interval = vdata.interval
    assert interval is not None
    key = tuple(slice(mn, mx) for mn, mx in zip(*interval, strict=True))
    np.testing.assert_array_equal(
        np.asarray(vdata[key]),
        np.asarray(multiscale_arrays[level][key]),
    )


def test_locked_data_level_is_loaded(
    qtbot, make_napari_viewer, multiscale_arrays
):
    viewer = make_napari_viewer()
    layer = add_progressive_loading_image(multiscale_arrays, viewer=viewer)
    loader = layer.metadata['progressive_loader']
    _wait_for_idle_loader(qtbot, loader)

    layer.locked_data_level = 0
    _wait_for_idle_loader(qtbot, loader)

    vdata = loader._data[0]
    assert vdata.interval is not None
    assert len(vdata.loaded_chunks) > 0
    np.testing.assert_array_equal(
        np.asarray(vdata[0:256, 0:256]), np.asarray(multiscale_arrays[0])
    )


def test_removing_layer_closes_loader(
    qtbot, make_napari_viewer, multiscale_arrays
):
    viewer = make_napari_viewer()
    layer = add_progressive_loading_image(multiscale_arrays, viewer=viewer)
    loader = layer.metadata['progressive_loader']

    viewer.layers.remove(layer)
    assert loader._closed
    qtbot.waitUntil(lambda: loader._worker is None, timeout=10000)


def test_fetch_pass_is_cancelled_on_view_change(
    qtbot, make_napari_viewer, multiscale_arrays
):
    viewer = make_napari_viewer()
    layer = add_progressive_loading_image(multiscale_arrays, viewer=viewer)
    loader = layer.metadata['progressive_loader']
    _wait_for_idle_loader(qtbot, loader)

    generation = loader._generation
    layer.locked_data_level = 0
    layer.locked_data_level = 1
    # each view change bumps the generation (after the debounced check
    # runs) so stale chunks from cancelled passes are dropped
    qtbot.waitUntil(lambda: loader._generation > generation, timeout=10000)
    _wait_for_idle_loader(qtbot, loader)


def test_contrast_limits_estimated(
    qtbot, make_napari_viewer, multiscale_arrays
):
    viewer = make_napari_viewer()
    layer = add_progressive_loading_image(multiscale_arrays, viewer=viewer)
    low, high = layer.contrast_limits
    assert low < high
    loader = layer.metadata['progressive_loader']
    _wait_for_idle_loader(qtbot, loader)


def test_interval_clamped_to_memory_budget(
    qtbot, make_napari_viewer, multiscale_arrays
):
    from napari.experimental._virtual_data import MultiScaleVirtualData

    viewer = make_napari_viewer()
    data = MultiScaleVirtualData(multiscale_arrays)
    layer = viewer.add_image(
        data._data, multiscale=True, contrast_limits=(0, 255)
    )
    loader = ProgressiveLoader(viewer, layer, data, interval_max_bytes=4096)
    layer.metadata['progressive_loader'] = loader
    min_coord, max_coord = loader._level_interval(0)
    extent = np.asarray(max_coord) - np.asarray(min_coord)
    assert np.prod(extent) <= 4096
    loader.close()


# ---------- 3D automatic level selection ----------


@pytest.fixture
def multiscale_3d_arrays():
    base = np.random.default_rng(0).integers(
        1, 255, size=(64, 64, 64), dtype=np.uint8
    )
    levels = [base, base[::2, ::2, ::2].copy(), base[::4, ::4, ::4].copy()]
    return [da.from_array(level, chunks=(16, 16, 16)) for level in levels]


def test_auto_level_3d_follows_zoom(
    qtbot, make_napari_viewer, multiscale_3d_arrays
):
    viewer = make_napari_viewer()
    viewer.dims.ndisplay = 3
    layer = add_progressive_loading_image(multiscale_3d_arrays, viewer=viewer)
    loader = layer.metadata['progressive_loader']
    _wait_for_idle_loader(qtbot, loader)

    # zoomed far out: coarsest level
    viewer.camera.zoom = 0.01
    qtbot.waitUntil(lambda: layer.data_level == 2, timeout=10000)
    # the resolution selector still reads "Auto" to the user: the level
    # was driven without emitting locked_data_level events
    assert not loader._user_locked
    _wait_for_idle_loader(qtbot, loader)

    # zoomed in: finest level
    viewer.camera.zoom = 50.0
    qtbot.waitUntil(lambda: layer.data_level == 0, timeout=10000)
    _wait_for_idle_loader(qtbot, loader)
    assert len(loader._data[0].loaded_chunks) > 0


def test_auto_level_3d_respects_user_pin(
    qtbot, make_napari_viewer, multiscale_3d_arrays
):
    viewer = make_napari_viewer()
    viewer.dims.ndisplay = 3
    layer = add_progressive_loading_image(multiscale_3d_arrays, viewer=viewer)
    loader = layer.metadata['progressive_loader']
    _wait_for_idle_loader(qtbot, loader)

    layer.locked_data_level = 2
    assert loader._user_locked
    viewer.camera.zoom = 50.0
    loader._check()
    # auto mode must not override the user's pin
    assert layer.data_level == 2
    _wait_for_idle_loader(qtbot, loader)

    # back to Auto: zoom-driven selection resumes
    layer.locked_data_level = None
    assert not loader._user_locked
    qtbot.waitUntil(lambda: layer.data_level == 0, timeout=10000)
    _wait_for_idle_loader(qtbot, loader)


def test_auto_level_3d_released_in_2d(
    qtbot, make_napari_viewer, multiscale_3d_arrays
):
    viewer = make_napari_viewer()
    viewer.dims.ndisplay = 3
    layer = add_progressive_loading_image(multiscale_3d_arrays, viewer=viewer)
    loader = layer.metadata['progressive_loader']
    viewer.camera.zoom = 50.0
    qtbot.waitUntil(lambda: loader._auto_locked is not None, timeout=10000)
    _wait_for_idle_loader(qtbot, loader)

    viewer.dims.ndisplay = 2
    qtbot.waitUntil(lambda: loader._auto_locked is None, timeout=10000)
    # napari's own 2D level selection is back in control
    assert layer._locked_data_level is None
    _wait_for_idle_loader(qtbot, loader)


def test_auto_level_3d_can_be_disabled(
    qtbot, make_napari_viewer, multiscale_3d_arrays
):
    viewer = make_napari_viewer()
    viewer.dims.ndisplay = 3
    layer = add_progressive_loading_image(
        multiscale_3d_arrays, viewer=viewer, auto_level_3d=False
    )
    loader = layer.metadata['progressive_loader']
    viewer.camera.zoom = 50.0
    loader._check()
    # napari's 3D behavior: coarsest level
    assert layer.data_level == len(multiscale_3d_arrays) - 1
    _wait_for_idle_loader(qtbot, loader)


def test_zoom_target_level_respects_memory_budget(
    qtbot, make_napari_viewer, multiscale_3d_arrays
):
    viewer = make_napari_viewer()
    viewer.dims.ndisplay = 3
    from napari.experimental._virtual_data import MultiScaleVirtualData

    data = MultiScaleVirtualData(multiscale_3d_arrays)
    layer = viewer.add_image(
        data._data, multiscale=True, contrast_limits=(0, 255)
    )
    loader = ProgressiveLoader(viewer, layer, data, interval_max_bytes=20**3)
    layer.metadata['progressive_loader'] = loader
    viewer.camera.zoom = 50.0
    # level 0 (64^3) and level 1 (32^3) exceed the budget; 2 (16^3) fits
    assert loader._zoom_target_level_3d() == 2
    loader.close()


def test_chunk_priority_3d_degenerate_camera():
    """NaN/zero camera state must not produce NaN priorities or warnings.

    Regression test: before the 3D camera is fully initialized (e.g. the
    window has not been shown yet), view_direction can be zero and
    center/zoom non-finite, which corrupted the chunk sort order.
    """
    import warnings

    arr = da.zeros((64, 64, 64), chunks=(16, 16, 16), dtype=np.uint8)
    vdata = VirtualData(arr)
    keys = chunk_slices(vdata)
    degenerate_cameras = [
        {'camera_center': (0, 0, 0), 'view_direction': (0, 0, 0)},
        {
            'camera_center': (np.nan, np.nan, np.nan),
            'view_direction': (1, 0, 0),
        },
        {
            'camera_center': (np.inf, 0, 0),
            'view_direction': (1, 0, 0),
            'zoom': np.nan,
        },
        {
            'camera_center': (32, 32, 32),
            'view_direction': (1, 0, 0),
            'zoom': 0.0,
        },
        # huge-but-finite center: overflows the priority arithmetic
        {
            'camera_center': (1e308, -1e308, 1e308),
            'view_direction': (1, 0, 0),
        },
        {
            'camera_center': (32, 32, 32),
            'view_direction': (1, 0, 0),
            'zoom': 1e308,
        },
    ]
    for camera in degenerate_cameras:
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            queue = chunk_priority_3D(keys, (0, 0, 0), (64, 64, 64), **camera)
        assert len(queue) == 64
        # the most central chunks must still come first
        first_center = np.array([(sl.start + sl.stop) / 2 for sl in queue[0]])
        assert np.all(np.abs(first_center - 32) <= 16)


def test_zoom_target_level_3d_uninitialized_camera(
    qtbot, make_napari_viewer, multiscale_3d_arrays
):
    """A NaN/zero zoom (camera not yet initialized) selects the coarsest
    level instead of falling through to the finest."""
    viewer = make_napari_viewer()
    viewer.dims.ndisplay = 3
    layer = add_progressive_loading_image(multiscale_3d_arrays, viewer=viewer)
    loader = layer.metadata['progressive_loader']
    coarsest = len(multiscale_3d_arrays) - 1

    # shim the camera: assigning NaN/zero zoom to a real camera breaks
    # napari's own transforms, but a real camera can hold such values
    # transiently before the window is first shown
    from types import SimpleNamespace

    real_viewer = loader._viewer
    for bad_zoom in (float('nan'), float('inf'), 0.0):
        loader._viewer = SimpleNamespace(
            camera=SimpleNamespace(zoom=bad_zoom), dims=real_viewer.dims
        )
        assert loader._zoom_target_level_3d() == coarsest
    loader._viewer = real_viewer

    # camera in a valid state: zoom-driven selection resumes
    viewer.camera.zoom = 50.0
    qtbot.waitUntil(lambda: layer.data_level == 0, timeout=10000)
    _wait_for_idle_loader(qtbot, loader)


def test_auto_level_3d_survives_selector_echo(
    qtbot, make_napari_viewer, multiscale_3d_arrays
):
    """The resolution-selector widget may echo the auto-driven level back
    through the public setter; this must not suspend auto mode."""
    viewer = make_napari_viewer()
    viewer.dims.ndisplay = 3
    layer = add_progressive_loading_image(multiscale_3d_arrays, viewer=viewer)
    loader = layer.metadata['progressive_loader']
    viewer.camera.zoom = 0.01
    qtbot.waitUntil(lambda: loader._auto_locked is not None, timeout=10000)
    _wait_for_idle_loader(qtbot, loader)

    # simulate the widget writing the current (auto) value back
    layer.locked_data_level = loader._auto_locked
    assert not loader._user_locked

    # auto mode still follows the zoom afterwards
    viewer.camera.zoom = 50.0
    qtbot.waitUntil(lambda: layer.data_level == 0, timeout=10000)
    _wait_for_idle_loader(qtbot, loader)


# ---------- never-empty canvas (backdrop across level switches) ----------


def test_backdrop_prefers_nearest_loaded_level(
    qtbot, make_napari_viewer, multiscale_arrays
):
    """A level switch should source its backdrop from the level that was
    just displayed, not always the coarsest level."""
    viewer = make_napari_viewer()
    layer = add_progressive_loading_image(multiscale_arrays, viewer=viewer)
    loader = layer.metadata['progressive_loader']
    _wait_for_idle_loader(qtbot, loader)

    # fully load level 1 (wait for chunks: the fetch pass itself only
    # starts after the debounced check fires)
    layer.locked_data_level = 1
    qtbot.waitUntil(
        lambda: len(loader._data[1].loaded_chunks) > 0, timeout=10000
    )
    _wait_for_idle_loader(qtbot, loader)

    min_coord = np.zeros(2, dtype=np.int64)
    max_coord = np.asarray(loader._data[0].shape, dtype=np.int64)
    assert loader._backdrop_level(0, min_coord, max_coord) == 1


def test_level_switch_keeps_canvas_filled(
    qtbot, make_napari_viewer, multiscale_arrays
):
    """Right after switching to a not-yet-fetched level, the level's data
    must already contain (upsampled) content from a previously displayed
    level — the canvas is never empty while chunks stream in."""
    viewer = make_napari_viewer()
    layer = add_progressive_loading_image(multiscale_arrays, viewer=viewer)
    loader = layer.metadata['progressive_loader']
    _wait_for_idle_loader(qtbot, loader)

    layer.locked_data_level = 0
    # as soon as the interval exists (set by the fast backdrop path or the
    # fetch pass), it must be filled with backdrop content — without
    # waiting for the fetch to complete
    qtbot.waitUntil(
        lambda: loader._data[0].interval is not None, timeout=10000
    )
    hyperslice = loader._data[0].hyperslice
    # source data has no zeros, so any zeros would be unfilled regions
    assert (hyperslice == 0).mean() < 0.05
    _wait_for_idle_loader(qtbot, loader)
