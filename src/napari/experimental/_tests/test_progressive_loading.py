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
