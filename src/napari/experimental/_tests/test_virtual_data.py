import dask.array as da
import numpy as np
import pytest

pytest.importorskip('lodstone', reason='requires the progressive extra')

from napari.experimental._virtual_data import (
    MultiScaleVirtualData,
    VirtualData,
)


def test_lodstone_virtual_data_satisfies_layer_data_protocol():
    """The compatibility export remains valid napari layer data."""
    from napari.layers._data_protocols import assert_protocol

    array = da.zeros((100, 120), chunks=(32, 40), dtype=np.uint16)
    vdata = VirtualData(array)

    assert_protocol(vdata)
    assert vdata.shape == (100, 120)
    assert vdata.ndim == 2
    assert vdata.dtype == np.uint16
    assert vdata.size == 100 * 120


def test_multiscale_data_wrapper_accepts_lodstone_virtual_data():
    from napari.layers._multiscale_data import MultiScaleData

    array = da.zeros((100, 120), chunks=(32, 40), dtype=np.uint16)
    data = MultiScaleVirtualData([array, array[::2, ::2]])
    wrapped = MultiScaleData(data._data)

    assert wrapped.shape == (100, 120)
    assert wrapped.dtype == np.uint16
