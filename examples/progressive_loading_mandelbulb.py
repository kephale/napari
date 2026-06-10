"""
Progressive loading: 3D Mandelbulb
==================================

Display a multiscale 3D volume whose chunks are computed on the fly, using
napari's experimental progressive loading in 3D.

In 3D, napari renders the coarsest level of a multiscale image. Chunks are
prioritized by distance to the camera, so the volume fills in from the
front of the view. Use the resolution selector in the layer controls
(``locked_data_level``) to force a finer level — its chunks will stream in
progressively, with coarser data shown as a backdrop in the meantime.

.. tags:: experimental
"""

import napari
from napari.experimental._progressive_loading import (
    add_progressive_loading_image,
)
from napari.experimental._progressive_loading_datasets import (
    mandelbulb_dataset,
)

dataset = mandelbulb_dataset(max_levels=5, tilesize=32, maxiter=64)

viewer = napari.Viewer()
viewer.dims.ndisplay = 3
layer = add_progressive_loading_image(
    dataset['arrays'],
    viewer=viewer,
    contrast_limits=(0, 64),
    colormap='magma',
    rendering='attenuated_mip',
)

if __name__ == '__main__':
    napari.run()
