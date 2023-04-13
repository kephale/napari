"""
nD multiscale image
===================

Displays an nD multiscale image

.. tags:: visualization-advanced
"""

import numpy as np
import dask.array as da
from skimage.transform import pyramid_gaussian

from napari.experimental._progressive_loading import ChunkCacheManager

import napari

# create multiscale from random data
img_shape = (128, 256, 256)
base = np.random.random(img_shape[1:]) * 255
depth = img_shape[0]
base = np.array([base * (depth - i) / depth for i in range(depth)])
print('base shape', base.shape)
multiscale = list(
    pyramid_gaussian(base, downscale=2, max_layer=2, channel_axis=-1)
)
# TODO note the dtype, not all have been tested
multiscale = [da.array(a, dtype=np.uint8).rechunk((64, 64, 64)) for a in multiscale]
print('multiscale level shapes: ', [p.shape for p in multiscale])

# add image multiscale
# viewer = napari.view_image(multiscale, contrast_limits=[0, 1], multiscale=True)
viewer = napari.Viewer()
layer = viewer.add_image(multiscale, contrast_limits=[0, 255], multiscale=True)
layer.metadata["zarr_container"] = "kyle.zarr"
layer.metadata["zarr_dataset"] = "brain"
layer.metadata["cache_manager"] = ChunkCacheManager(8e9)
layer.metadata["viewer"] = viewer


if __name__ == '__main__':
    napari.run()
