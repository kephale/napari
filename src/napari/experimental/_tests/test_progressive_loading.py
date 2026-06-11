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
    # strictly front-to-back: the chunk closest to the viewer loads first,
    # and depth along the view direction never decreases
    assert queue[0][0].start == 0
    depths = [key[0].start for key in queue]
    assert depths == sorted(depths)


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
    assert loader._worker is None
    # cancellation is cooperative: give in-flight fetch threads a moment
    # to drain before teardown checks for live thread pools
    qtbot.wait(300)


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
    # zoomed OUT, the viewport covers the whole volume: level 0 (64^3,
    # tiled to 32^3 minimum) and level 1 (32^3) exceed the byte budget;
    # level 2 (16^3) fits. (Zoomed in, the viewport bound can make finer
    # levels affordable - see test_zoom_target_respects_chunk_budget.)
    viewer.camera.zoom = 0.1
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
        },
        # huge-but-finite center: overflows the priority arithmetic
        {
            'camera_center': (1e308, -1e308, 1e308),
            'view_direction': (1, 0, 0),
        },
    ]
    for camera in degenerate_cameras:
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            queue = chunk_priority_3D(keys, (0, 0, 0), (64, 64, 64), **camera)
        assert len(queue) == 64
        # fallback ordering: the most central chunks come first
        first_center = np.array([(sl.start + sl.stop) / 2 for sl in queue[0]])
        assert np.all(np.abs(first_center - 32) <= 16)
    # a valid camera with a junk zoom value must not warn either (zoom is
    # unused by the front-to-back ordering)
    for zoom in (0.0, np.nan, 1e308):
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            queue = chunk_priority_3D(
                keys,
                (0, 0, 0),
                (64, 64, 64),
                camera_center=(32, 32, 32),
                view_direction=(1, 0, 0),
                zoom=zoom,
            )
        assert queue[0][0].start == 0  # front-to-back


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


# ---------- 3D sub-volume tiles ----------


def test_corners_for_locked_level_subvolume(
    qtbot, make_napari_viewer, multiscale_3d_arrays
):
    """With _max_tile_extent_3d set, a locked 3D level larger than the
    extent renders a centered sub-volume tile instead of the full level."""
    viewer = make_napari_viewer()
    viewer.dims.ndisplay = 3
    layer = viewer.add_image(
        list(multiscale_3d_arrays), multiscale=True, contrast_limits=(0, 255)
    )
    displayed = layer._slice_input.displayed

    # default: full extent
    corners = layer._corners_for_locked_level(0, displayed)
    assert (corners[1] - corners[0] + 1).max() == 64

    layer._max_tile_extent_3d = 32
    bbox = np.array([[40, 40, 40], [40, 40, 40]])
    corners = layer._corners_for_locked_level(0, displayed, bbox)
    extent = corners[1] - corners[0] + 1
    np.testing.assert_array_equal(extent[list(displayed)], 32)
    center = corners.mean(axis=0)[list(displayed)]
    np.testing.assert_allclose(center, 39.5, atol=1)

    # small level: unaffected
    corners = layer._corners_for_locked_level(2, displayed, bbox)
    assert (corners[1] - corners[0] + 1)[list(displayed)].max() == 16


def test_locked_tile_hysteresis(
    qtbot, make_napari_viewer, multiscale_3d_arrays
):
    viewer = make_napari_viewer()
    viewer.dims.ndisplay = 3
    layer = viewer.add_image(
        list(multiscale_3d_arrays), multiscale=True, contrast_limits=(0, 255)
    )
    layer._max_tile_extent_3d = 32
    displayed = layer._slice_input.displayed
    bbox = np.array([[32, 32, 32], [32, 32, 32]])
    corners = layer._corners_for_locked_level(0, displayed, bbox)
    layer.corner_pixels = corners

    # small movement (< extent/4): no re-slice
    near = layer._corners_for_locked_level(
        0, displayed, np.array([[36, 36, 36]] * 2)
    )
    assert not layer._locked_tile_moved(near, displayed)
    # large movement: re-slice
    far = layer._corners_for_locked_level(
        0, displayed, np.array([[60, 60, 60]] * 2)
    )
    assert layer._locked_tile_moved(far, displayed)


def test_progressive_loading_3d_subvolume_tile(
    qtbot, make_napari_viewer, multiscale_3d_arrays
):
    """The finest level of a volume larger than the tile budget is
    selectable and loads only the tile's chunks."""
    viewer = make_napari_viewer()
    viewer.dims.ndisplay = 3
    layer = add_progressive_loading_image(
        multiscale_3d_arrays,
        viewer=viewer,
        # 32^3 voxels (uint8): level 0 (64^3) must be tiled
        interval_max_bytes=32**3,
    )
    loader = layer.metadata['progressive_loader']
    assert loader._tile_extent_3d == 32
    assert layer._max_tile_extent_3d == 32
    _wait_for_idle_loader(qtbot, loader)

    layer.locked_data_level = 0
    qtbot.waitUntil(
        lambda: len(loader._data[0].loaded_chunks) > 0, timeout=10000
    )
    _wait_for_idle_loader(qtbot, loader)

    extent = layer.corner_pixels[1] - layer.corner_pixels[0] + 1
    displayed = list(layer._slice_input.displayed)
    assert np.all(extent[displayed] <= 32)
    interval = loader._data[0].interval
    interval_extent = np.array(interval[1]) - np.array(interval[0])
    # the resident interval covers the tile, not the whole level
    assert np.all(interval_extent <= 48)  # tile + chunk alignment slack
    # tile content matches the source data
    key = tuple(slice(mn, mx) for mn, mx in zip(*interval, strict=True))
    np.testing.assert_array_equal(
        np.asarray(loader._data[0][key]),
        np.asarray(multiscale_3d_arrays[0][key]),
    )


def test_chunk_priority_3d_closest_visible_first():
    """Among chunks at the same depth, on-axis chunks lead; across depths,
    closer-to-viewer always wins."""
    arr = da.zeros((64, 64, 64), chunks=(16, 16, 16), dtype=np.uint8)
    keys = chunk_slices(VirtualData(arr))
    queue = chunk_priority_3D(
        keys,
        (0, 0, 0),
        (64, 64, 64),
        camera_center=(32, 32, 32),
        view_direction=(1, 0, 0),
    )
    # first chunk: front slab, on the center line
    first = queue[0]
    assert first[0].start == 0
    assert first[1] == slice(16, 32) or first[1] == slice(32, 48)
    assert first[2] == slice(16, 32) or first[2] == slice(32, 48)


def test_auto_label_shows_current_level(
    qtbot, make_napari_viewer, multiscale_arrays
):
    """The resolution selector's Auto entry indicates the rendered level."""
    viewer = make_napari_viewer()
    layer = add_progressive_loading_image(multiscale_arrays, viewer=viewer)
    loader = layer.metadata['progressive_loader']
    _wait_for_idle_loader(qtbot, loader)

    control = viewer.window._qt_viewer.controls.widgets[layer]
    combo = control._multiscale_level_control.level_combobox
    assert combo.itemText(0) == f'Auto ({layer.data_level})'

    layer.locked_data_level = 0
    qtbot.waitUntil(lambda: combo.itemText(0) == 'Auto (0)', timeout=10000)
    _wait_for_idle_loader(qtbot, loader)


# ---------- deep pyramids (zoomed-out economics, world-size limits) ----------


def test_zoom_target_respects_chunk_budget(
    qtbot, make_napari_viewer, multiscale_3d_arrays
):
    """Auto level selection coarsens until the viewport tile can be
    fetched within max_chunks_per_pass."""
    from napari.experimental._virtual_data import MultiScaleVirtualData

    viewer = make_napari_viewer()
    viewer.dims.ndisplay = 3
    data = MultiScaleVirtualData(multiscale_3d_arrays)
    layer = viewer.add_image(
        data._data, multiscale=True, contrast_limits=(0, 255)
    )
    loader = ProgressiveLoader(
        viewer, layer, data, max_chunks_per_pass=8, auto_level_3d=True
    )
    layer.metadata['progressive_loader'] = loader
    _wait_for_idle_loader(qtbot, loader)
    # close first so the camera change cannot start fetch passes; the
    # level computation itself is a pure function of the camera state
    loader.close()
    viewer.camera.zoom = 0.1  # zoomed out: viewport covers the volume
    target = loader._zoom_target_level_3d()
    # level 0 = 4^3 = 64 chunks > 8; level 1 = 2^3 = 8 chunks fits
    assert target >= 1


def test_fill_unloaded_from_repairs_backdrop():
    """Backdrop repair fills only chunks without real data."""
    from napari.experimental._virtual_data import MultiScaleVirtualData

    base = np.full((64, 64), 7, dtype=np.uint8)
    coarse = np.full((32, 32), 9, dtype=np.uint8)
    msvd = MultiScaleVirtualData(
        [
            da.from_array(base, chunks=(16, 16)),
            da.from_array(coarse, chunks=(16, 16)),
        ]
    )
    # resident coarse level fully loaded
    msvd[1].set_interval((0, 0), (32, 32))
    msvd[1].set_offset((slice(0, 32), slice(0, 32)), coarse)
    msvd[1].loaded_chunks.add(((0, 16), (0, 16)))

    # fine level: interval initialized to zeros (the race), one real chunk
    msvd[0].set_interval((0, 0), (64, 64))
    msvd[0].set_offset((slice(0, 16), slice(0, 16)), base[:16, :16])
    msvd[0].loaded_chunks.add(((0, 16), (0, 16)))

    assert msvd.fill_unloaded_from(0, 1)
    hyperslice = msvd[0].hyperslice
    # the loaded chunk keeps its real data
    assert (hyperslice[:16, :16] == 7).all()
    # everything else now shows the upsampled coarse value
    assert (hyperslice[16:, 16:] == 9).all()
    assert (hyperslice[:16, 16:] == 9).all()


def test_huge_world_auto_normalized(qtbot, make_napari_viewer):
    """Pyramids whose extent exceeds float32 rendering precision get a
    normalizing layer scale (3D rendering goes blank past 2**22)."""
    base_shape = 2**23
    levels = []
    size = base_shape
    while size >= 64:
        levels.append(
            da.zeros((size, size), chunks=(min(size, 64),) * 2, dtype='u1')
        )
        size //= 2
    viewer = make_napari_viewer()
    layer = add_progressive_loading_image(
        levels, viewer=viewer, contrast_limits=(0, 255)
    )
    world_extent = base_shape * layer.scale[0]
    assert world_extent <= 2**21
    loader = layer.metadata['progressive_loader']
    _wait_for_idle_loader(qtbot, loader)


def test_texture_patching_used_in_3d(
    qtbot, make_napari_viewer, multiscale_3d_arrays
):
    """3D chunk arrivals go to the GPU as partial texture updates rather
    than full re-slice + re-upload refreshes."""
    viewer = make_napari_viewer()
    viewer.dims.ndisplay = 3
    layer = add_progressive_loading_image(multiscale_3d_arrays, viewer=viewer)
    loader = layer.metadata['progressive_loader']
    _wait_for_idle_loader(qtbot, loader)

    layer.locked_data_level = 0
    qtbot.waitUntil(
        lambda: len(loader._data[0].loaded_chunks) > 0, timeout=10000
    )
    _wait_for_idle_loader(qtbot, loader)
    assert loader._texture_patches > 0
    # patched texture content still matches the source at the end (the
    # final reconcile re-slices through the normal pipeline)
    np.testing.assert_array_equal(
        np.asarray(loader._data[0][0:64, 0:64, 0:64]),
        np.asarray(multiscale_3d_arrays[0]),
    )


# ---------- fetch rate limiting ----------


def test_rate_limiter_paces_bytes_per_second():
    import time as _time

    from napari.experimental._progressive_loading import _FetchRateLimiter

    limiter = _FetchRateLimiter(bytes_per_second=1e6)  # 1 MB/s
    start = _time.monotonic()
    # 5 x 100 KB: first acquire is free, the rest are paced -> >= ~0.4s
    for _ in range(5):
        limiter.acquire(100_000)
    elapsed = _time.monotonic() - start
    assert elapsed >= 0.35
    assert elapsed < 2.0


def test_rate_limiter_cancel_wakes_sleepers():
    import threading as _threading
    import time as _time

    from napari.experimental._progressive_loading import _FetchRateLimiter

    limiter = _FetchRateLimiter(bytes_per_second=1.0)  # absurdly slow
    limiter.acquire(10)  # next acquire would sleep ~10s
    done = _threading.Event()

    def sleeper():
        limiter.acquire(10)
        done.set()

    t = _threading.Thread(target=sleeper, daemon=True)
    t.start()
    _time.sleep(0.05)
    limiter.cancel()
    assert done.wait(1.0), 'cancel() did not wake the sleeping acquire'
    # cancelled limiter no longer paces at all
    start = _time.monotonic()
    limiter.acquire(10**9)
    assert _time.monotonic() - start < 0.1


def test_key_nbytes():
    from napari.experimental._progressive_loading import _key_nbytes

    key = (slice(0, 4), slice(8, 16), slice(0, 32))
    assert _key_nbytes(key, 2) == 4 * 8 * 32 * 2


def test_fetch_chunks_respects_limiter(multiscale_arrays):
    import time as _time

    from napari.experimental._progressive_loading import (
        _fetch_chunks,
        _FetchRateLimiter,
    )

    vdata = VirtualData(multiscale_arrays[2])  # 64x64 uint8
    vdata.set_interval((0, 0), (64, 64))
    keys = [
        (slice(i, i + 32), slice(j, j + 32)) for i in (0, 32) for j in (0, 32)
    ]  # 4 chunks x 1 KB
    limiter = _FetchRateLimiter(bytes_per_second=8_192)  # 8 chunks/s
    start = _time.monotonic()
    batches = list(
        _fetch_chunks.__wrapped__(
            vdata.array, keys, num_workers=2, limiter=limiter
        )
    )
    elapsed = _time.monotonic() - start
    fetched = [key for batch in batches for key in batch]
    assert sorted(map(str, fetched)) == sorted(map(str, keys))
    # 4 KB at 8 KB/s with the first chunk free -> >= ~0.375s
    assert elapsed >= 0.3


def test_loader_max_bytes_per_second_env_override(
    qtbot, make_napari_viewer, multiscale_arrays, monkeypatch
):
    monkeypatch.setenv('NAPARI_PROGRESSIVE_MAX_BPS', '12500000')
    viewer = make_napari_viewer()
    layer = add_progressive_loading_image(multiscale_arrays, viewer=viewer)
    loader = layer.metadata['progressive_loader']
    assert loader._max_bytes_per_second == 12_500_000
    _wait_for_idle_loader(qtbot, loader)


def test_loader_unlimited_by_default(
    qtbot, make_napari_viewer, multiscale_arrays
):
    viewer = make_napari_viewer()
    layer = add_progressive_loading_image(multiscale_arrays, viewer=viewer)
    loader = layer.metadata['progressive_loader']
    assert loader._max_bytes_per_second is None
    # a limiter is still created (it doubles as the interaction-hold
    # gate) but performs no rate pacing
    assert loader._make_limiter().bytes_per_second is None
    _wait_for_idle_loader(qtbot, loader)


# ---------- interaction hold ----------


def test_rate_limiter_pause_blocks_until_resume():
    import threading as _threading
    import time as _time

    from napari.experimental._progressive_loading import _FetchRateLimiter

    limiter = _FetchRateLimiter()  # unlimited rate, gate only
    limiter.pause()
    passed = _threading.Event()

    def worker():
        limiter.acquire(1)
        passed.set()

    t = _threading.Thread(target=worker, daemon=True)
    t.start()
    _time.sleep(0.05)
    assert not passed.is_set(), 'paused limiter let a fetch through'
    limiter.resume()
    assert passed.wait(1.0), 'resume() did not release the worker'


def test_rate_limiter_cancel_releases_paused_worker():
    import threading as _threading
    import time as _time

    from napari.experimental._progressive_loading import _FetchRateLimiter

    limiter = _FetchRateLimiter()
    limiter.pause()
    passed = _threading.Event()
    t = _threading.Thread(
        target=lambda: (limiter.acquire(1), passed.set()), daemon=True
    )
    t.start()
    _time.sleep(0.05)
    limiter.cancel()
    assert passed.wait(1.0), 'cancel() did not release a paused worker'


def test_interaction_hold_buffers_batches_and_refreshes(
    qtbot, make_napari_viewer, multiscale_arrays
):
    import time as _time

    viewer = make_napari_viewer()
    layer = add_progressive_loading_image(multiscale_arrays, viewer=viewer)
    loader = layer.metadata['progressive_loader']
    _wait_for_idle_loader(qtbot, loader)

    assert not loader._holding
    loader._on_interaction()
    assert loader._holding

    # chunk batches arriving during the hold are buffered, not patched
    generation = loader._generation
    vdata = loader._data[int(layer.data_level)]
    loader._on_chunks(generation, vdata, [])
    assert loader._held_batches == [(generation, vdata, [])]

    # throttled refreshes are deferred; forced ones still run
    loader._held_refresh = False
    loader._refresh()
    assert loader._held_refresh

    # the debounced check ends the hold and replays the buffer
    loader._check()
    assert not loader._holding
    assert loader._held_batches == []
    assert not loader._held_refresh
    assert loader._hold_until == 0.0
    # repeated interaction re-arms the hold
    loader._on_interaction()
    assert loader._holding
    assert loader._hold_until > _time.monotonic()
    loader._end_hold()
    _wait_for_idle_loader(qtbot, loader)


def test_interaction_hold_disabled_by_env(
    qtbot, make_napari_viewer, multiscale_arrays, monkeypatch
):
    monkeypatch.setenv('NAPARI_PROGRESSIVE_HOLD', '0')
    viewer = make_napari_viewer()
    layer = add_progressive_loading_image(multiscale_arrays, viewer=viewer)
    loader = layer.metadata['progressive_loader']
    loader._on_interaction()
    assert not loader._holding
    _wait_for_idle_loader(qtbot, loader)


def test_interaction_hold_pauses_fetch_limiter(
    qtbot, make_napari_viewer, multiscale_arrays
):
    viewer = make_napari_viewer()
    layer = add_progressive_loading_image(multiscale_arrays, viewer=viewer)
    loader = layer.metadata['progressive_loader']
    _wait_for_idle_loader(qtbot, loader)
    loader._limiter = loader._make_limiter()
    loader._on_interaction()
    assert not loader._limiter._go.is_set()
    loader._end_hold()
    assert loader._limiter._go.is_set()
    loader._limiter = None
    _wait_for_idle_loader(qtbot, loader)


def test_glir_hold_defers_all_uploads(monkeypatch):
    import time as _time

    from vispy.gloo import glir

    from napari.experimental import _glir_metering as gm

    monkeypatch.delenv('NAPARI_GLIR_METERING', raising=False)

    class FakeParser:
        def __init__(self):
            self._objects = {}
            self.executed = []

        def _parse(self, command):
            self.executed.append(command)

        def parse(self, commands):
            for c in commands:
                self._parse(c)

    parser = FakeParser()
    parser._objects[1] = glir.GlirTexture3D.__new__(glir.GlirTexture3D)
    try:
        assert gm.install(frame_budget_bytes=64 * 2**20, slab_bytes=2**20)
        gm.hold_uploads_until(_time.monotonic() + 60.0)
        queue = glir.GlirQueue()
        data = np.zeros((4, 64, 64), dtype=np.uint8)
        queue.command('DATA', 1, (0, 0, 0), data)
        queue.command('UNIFORM', 7, 'u_x', 'float', 1.0)
        queue.flush(parser)
        # the upload was held but other commands ran
        assert [c[0] for c in parser.executed] == ['UNIFORM']
        state = gm._states[parser]
        assert sum(c[3].nbytes for c in state.carry) == data.nbytes
        # hold expiry releases the carry on the next flush
        gm._upload_hold_until = 0.0
        state.reset_budget()
        queue.flush(parser)
        assert state.carry == []
        assert (
            sum(c[3].nbytes for c in parser.executed if c[0] == 'DATA')
            == data.nbytes
        )
    finally:
        gm._upload_hold_until = 0.0
        gm.uninstall()


# ---------- double-buffered texture streaming ----------


def test_double_buffer_swaps_and_content_correct(
    qtbot, make_napari_viewer, multiscale_3d_arrays
):
    """Chunk patches stream into the back texture and present() swaps;
    the rendered result still matches the source exactly."""
    viewer = make_napari_viewer()
    viewer.dims.ndisplay = 3
    layer = add_progressive_loading_image(multiscale_3d_arrays, viewer=viewer)
    loader = layer.metadata['progressive_loader']
    _wait_for_idle_loader(qtbot, loader)

    layer.locked_data_level = 0
    qtbot.waitUntil(
        lambda: len(loader._data[0].loaded_chunks) > 0, timeout=10000
    )
    _wait_for_idle_loader(qtbot, loader)
    assert loader._texture_patches > 0
    assert loader._dbuf is not None, 'double buffer never engaged'
    node = loader._get_volume_node()
    assert loader._dbuf.matches(node)
    # the shader samples the front texture, and the pair is distinct
    assert node._texture is loader._dbuf._front
    assert loader._dbuf._front is not loader._dbuf._back
    # patch log fully drained once idle
    assert not loader._dbuf.dirty
    # data correctness end to end (final reconcile path)
    np.testing.assert_array_equal(
        np.asarray(loader._data[0][0:64, 0:64, 0:64]),
        np.asarray(multiscale_3d_arrays[0]),
    )


def test_double_buffer_disabled_by_env(
    qtbot, make_napari_viewer, multiscale_3d_arrays, monkeypatch
):
    monkeypatch.setenv('NAPARI_PROGRESSIVE_DOUBLE_BUFFER', '0')
    viewer = make_napari_viewer()
    viewer.dims.ndisplay = 3
    layer = add_progressive_loading_image(multiscale_3d_arrays, viewer=viewer)
    loader = layer.metadata['progressive_loader']
    _wait_for_idle_loader(qtbot, loader)
    layer.locked_data_level = 0
    qtbot.waitUntil(
        lambda: len(loader._data[0].loaded_chunks) > 0, timeout=10000
    )
    _wait_for_idle_loader(qtbot, loader)
    assert loader._texture_patches > 0
    assert loader._dbuf is None


# ---------- interactive render quality (LOD) ----------


def test_interactive_step_degrades_and_restores(
    qtbot, make_napari_viewer, multiscale_3d_arrays
):
    viewer = make_napari_viewer()
    viewer.dims.ndisplay = 3
    layer = add_progressive_loading_image(multiscale_3d_arrays, viewer=viewer)
    loader = layer.metadata['progressive_loader']
    _wait_for_idle_loader(qtbot, loader)
    node = loader._get_volume_node()
    assert node is not None
    base_step = float(node.relative_step_size)

    loader._on_interaction()
    assert loader._saved_step is not None
    assert float(node.relative_step_size) == pytest.approx(
        base_step * loader._interactive_step_rate
    )
    # repeated interaction events do not compound the degradation
    loader._on_interaction()
    assert float(node.relative_step_size) == pytest.approx(
        base_step * loader._interactive_step_rate
    )

    loader._end_hold()
    assert loader._saved_step is None
    assert float(node.relative_step_size) == pytest.approx(base_step)
    _wait_for_idle_loader(qtbot, loader)


def test_interactive_step_disabled(
    qtbot, make_napari_viewer, multiscale_3d_arrays, monkeypatch
):
    monkeypatch.setenv('NAPARI_PROGRESSIVE_INTERACTIVE_STEP', '1')
    viewer = make_napari_viewer()
    viewer.dims.ndisplay = 3
    layer = add_progressive_loading_image(multiscale_3d_arrays, viewer=viewer)
    loader = layer.metadata['progressive_loader']
    _wait_for_idle_loader(qtbot, loader)
    node = loader._get_volume_node()
    base_step = float(node.relative_step_size)
    loader._on_interaction()
    assert loader._interactive_step_rate is None
    assert loader._saved_step is None
    assert float(node.relative_step_size) == pytest.approx(base_step)
    loader._end_hold()
    _wait_for_idle_loader(qtbot, loader)


def test_interactive_step_not_applied_in_2d(
    qtbot, make_napari_viewer, multiscale_arrays
):
    viewer = make_napari_viewer()
    layer = add_progressive_loading_image(multiscale_arrays, viewer=viewer)
    loader = layer.metadata['progressive_loader']
    _wait_for_idle_loader(qtbot, loader)
    loader._on_interaction()
    assert loader._saved_step is None
    loader._end_hold()
    _wait_for_idle_loader(qtbot, loader)
