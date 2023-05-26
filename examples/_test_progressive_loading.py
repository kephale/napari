import logging
import sys
import pytest 
import numpy as np
from numpy.testing import assert_array_equal, assert_raises

import napari

from napari.experimental._progressive_loading_datasets import (
    mandelbrot_dataset, MandlebrotStore
)
from napari.experimental import _progressive_loading

from _mandelbrot_vizarr import add_progressive_loading_image, get_and_process_chunk_2D


@pytest.fixture(
        params=[
            8, 
            14,
        ]
)
def max_level(request):
    """Parameterized fixture that supplies a max_level for testing.

    Parameters
    ----------
    request : _pytest.fixtures.SubRequest
        The pytest request object

    Returns
    -------
    int
        max_level for mandelbrot datasets
    """
    return request.param

@pytest.fixture
def mandelbrot_arrays(max_level):
    large_image = mandelbrot_dataset(max_levels=max_level)
    multiscale_img = large_image["arrays"]
    return multiscale_img

def test_add_progressive_loading_image(mandelbrot_arrays):
    viewer = napari.Viewer()
    add_progressive_loading_image(mandelbrot_arrays, viewer=viewer)


def test_add_progressive_loading_image_zoom_in(mandelbrot_arrays):
    viewer = napari.Viewer()
    viewer.camera.zoom = 0.0001
    add_progressive_loading_image(mandelbrot_arrays, viewer=viewer)
    viewer.camera.zoom = 0.001
    

def test_add_progressive_loading_image_zoom_out(mandelbrot_arrays):
    viewer = napari.Viewer()
    viewer.camera.zoom = 0.001
    add_progressive_loading_image(mandelbrot_arrays, viewer=viewer)
    viewer.camera.zoom = 0.0001

# TODO test for nothing visible on initial load and for nothing visible after camera movement
def test_add_progressive_loading_image_no_visible(mandelbrot_arrays):
    viewer = napari.Viewer()
    non_visible_center = (0.0, -3242614, -9247091)
    start_zoom = 0.00005

    add_progressive_loading_image(mandelbrot_arrays, viewer=viewer)
    viewer.camera.zoom = start_zoom
    viewer.camera.center = non_visible_center


def test_add_progressive_loading_image_zoom_before_load(mandelbrot_arrays):
    # TODO fails for max_levels=8
    viewer = napari.Viewer()
    start_center = (0.0, 4194303.5, 4194303.5)
    non_visible_center = (0.0, -3242614, -9247091)
    start_zoom = 0.00005
    viewer.camera.zoom = start_zoom
    viewer.camera.center = non_visible_center
    add_progressive_loading_image(mandelbrot_arrays, viewer=viewer)
    
def test_MultiScaleVirtualData_set_interval(mandelbrot_arrays):
    # viewer = napari.Viewer()
    # multi_data = _progressive_loading.MultiScaleVirtualData(mandelbrot_arrays)
    # # add_progressive_loading_image(mandelbrot_arrays, viewer=viewer)
    
    # # prev_max_coord = multi_data._max_coord
    # # prev_min_coord = multi_data._min_coord

    # # pan to reset coords
    # coords = tuple([
    #     slice(0, 1024, None),
    #     slice(0, 1024, None)
    # ])
    # # coords = tuple([
    # #     slice(0, 673, None),
    # #     slice(0, 0, None)
    # # ])
    # # coords = tuple([
    # #     slice(0, 1346, None),
    # #     slice(0, 0, None)
    # # ])
    # min_coord = [0]
    # max_coord = [1]
    # multi_data.set_interval(min_coord, max_coord)
    # TODO still working on this
    pass


def test_chunk_slices_0_1024(mandelbrot_arrays, max_level):
    scale = max_level - 1
    vdata = _progressive_loading.VirtualData(mandelbrot_arrays[scale], scale=scale)
    data_interval = np.array([[0, 0], [1024, 1024]])
    chunk_keys = _progressive_loading.chunk_slices(vdata, ndim=2, interval=data_interval)
    dims = len(vdata.array.shape)

    result = [
        [slice(0, 512, None), slice(512, 1024, None)],
        [slice(0, 512, None), slice(512, 1024, None)],
    ]
    assert len(chunk_keys) == dims
    assert chunk_keys == result

def test_chunk_slices_512_1024(mandelbrot_arrays, max_level):
    scale = max_level - 1
    vdata = _progressive_loading.VirtualData(mandelbrot_arrays[scale], scale=scale)
    data_interval = np.array([[512, 512], [1024, 1024]])
    chunk_keys = _progressive_loading.chunk_slices(vdata, ndim=2, interval=data_interval)
    dims = len(vdata.array.shape)

    result = [
        [slice(512, 1024, None)],
        [slice(512, 1024, None)],
    ]
    assert len(chunk_keys) == dims
    assert chunk_keys == result

def test_chunk_slices_600_1024(mandelbrot_arrays, max_level):
    scale = max_level - 1
    vdata = _progressive_loading.VirtualData(mandelbrot_arrays[scale], scale=scale)
    data_interval = np.array([[600, 512], [600, 1024]])
    chunk_keys = _progressive_loading.chunk_slices(vdata, ndim=2, interval=data_interval)
    dims = len(vdata.array.shape)

    result = [
        [slice(512, 1024, None)], 
        [slice(512, 1024, None)], 
    ]
    assert len(chunk_keys) == dims
    assert chunk_keys == result


def test_virtualdata_init(mandelbrot_arrays, max_level):
    scale = max_level - 1
    vdata = _progressive_loading.VirtualData(mandelbrot_arrays[scale], scale=scale)
    

def test_virtualdata_set_interval(mandelbrot_arrays, max_level):
    scale = max_level - 1
    vdata = _progressive_loading.VirtualData(mandelbrot_arrays[scale], scale=scale)
    coords = tuple([slice(512, 1024, None), slice(512, 1024, None)])
    vdata.set_interval(coords)

    min_coord = [st.start for st in coords]
    max_coord = [st.stop for st in coords]
    assert vdata._min_coord == min_coord
    assert vdata._max_coord == max_coord

def test_virtualdata_data_plane_reuse(mandelbrot_arrays, max_level):
    scale = max_level - 1
    vdata = _progressive_loading.VirtualData(mandelbrot_arrays[scale], scale=scale)
    coords = tuple([slice(0, 1024, None), slice(0, 1024, None)])
    vdata.set_interval(coords)
    first_data_plane = vdata.data_plane
    vdata.set_interval(coords)
    second_data_plane = vdata.data_plane
    assert_array_equal(first_data_plane, second_data_plane)


def test_virtualdata_data_plane(mandelbrot_arrays, max_level):
    scale = max_level - 1
    vdata = _progressive_loading.VirtualData(mandelbrot_arrays[scale], scale=scale)
    coords = tuple([slice(0, 1024, None), slice(0, 1024, None)])
    vdata.set_interval(coords)
    first_data_plane = vdata.data_plane
    coords = tuple([slice(512, 1024, None), slice(512, 1024, None)])
    vdata.set_interval(coords)
    second_data_plane = vdata.data_plane
    assert_raises(AssertionError, assert_array_equal, first_data_plane, second_data_plane)


def test_multiscalevirtualdata_init(mandelbrot_arrays):
    mvdata = _progressive_loading.MultiScaleVirtualData(mandelbrot_arrays)
    assert isinstance(mvdata, _progressive_loading.MultiScaleVirtualData)


@pytest.mark.parametrize('max_level', [8, 14])
def test_MandlebrotStore(max_level):
    store = MandlebrotStore(
        levels=max_level, tilesize=512, compressor=None, maxiter=255  
    ) 

def test_get_and_process_chunk_2D():
    large_image = mandelbrot_dataset(max_levels=14)
    mandelbrot_arrays = large_image["arrays"]
    scale = 12
    virtual_data = _progressive_loading.VirtualData(mandelbrot_arrays[scale], scale=scale)
    chunk_slice = tuple([slice(1024, 1536, None), slice(512, 1024, None)])
    full_shape = None

    chunk_widths = tuple([chunk_slice[0].stop - chunk_slice[0].start, chunk_slice[1].stop - chunk_slice[1].start])
    chunk_slices, scale, real_array = get_and_process_chunk_2D(chunk_slice, scale, virtual_data, full_shape)

    assert chunk_widths == real_array.shape
    

if __name__ == "__main__":
    viewer = napari.Viewer()
    large_image = mandelbrot_dataset(max_levels=14)
    mandelbrot_arrays = large_image["arrays"]

    # scale = 7
    # vdata = _progressive_loading.VirtualData(mandelbrot_arrays[scale], scale=scale)

    start_center = (0.0, 4194303.5, 4194303.5)
    non_visible_center = (0.0, -3242614, -9247091)
    start_zoom = 0.00005
    viewer.camera.zoom = start_zoom
    viewer.camera.center = non_visible_center
    add_progressive_loading_image(mandelbrot_arrays, viewer=viewer)
