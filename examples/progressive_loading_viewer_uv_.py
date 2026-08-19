# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "napari[pyqt5]",
#     "zarr>=3.1.6",
#     "fsspec>=2023.10.0",
#     "aiohttp",
#     "requests",
#     "s3fs",
#     "dask[array]",
#     "lodstone>=0.1.0a0",
# ]
#
# [tool.uv.sources]
# napari = { git = "https://github.com/kephale/napari", branch = "lodstone-integration" }
# lodstone = { git = "https://github.com/kephale/lodstone", branch = "main" }
# ///
"""
Progressive loading dataset viewer
==================================

Open any local or remote OME-Zarr with progressive loading, or select one of
the bundled dataset presets::

    uv run examples/progressive_loading_viewer_uv_.py --list
    uv run examples/progressive_loading_viewer_uv_.py zebrahub
    uv run examples/progressive_loading_viewer_uv_.py cardiac
    uv run examples/progressive_loading_viewer_uv_.py hela-labels
    uv run examples/progressive_loading_viewer_uv_.py mandelbulb
    uv run examples/progressive_loading_viewer_uv_.py /local/image.zarr
    uv run examples/progressive_loading_viewer_uv_.py s3://bucket/image.zarr
    uv run examples/progressive_loading_viewer_uv_.py https://host/image.zarr

Command-line options override preset defaults. Use ``--labels`` to treat a
custom Zarr as labels and ``--3d`` or ``--2d`` to select the initial view.

.. tags:: experimental
"""

from __future__ import annotations

import argparse
from pathlib import Path

from lodstone.datasets import (
    local_zarr_dataset,
    mandelbrot_dataset,
    mandelbulb_dataset,
    mandelbulb_rgb_dataset,
    open_ome_zarr,
)

import napari
from napari.experimental._lodstone_loading import (
    add_lodstone_loading_image,
    add_lodstone_loading_labels,
)

MIB = 1024**2


def _remote(
    description,
    path,
    *,
    levels=None,
    contrast=None,
    colormap='gray',
    threed=False,
    rendering='attenuated_mip',
    zarr_format=None,
    name=None,
    transform=None,
):
    return {
        'description': description,
        'threed': threed,
        'layers': [
            {
                'path': path,
                'levels': levels,
                'contrast': contrast,
                'colormap': colormap,
                'rendering': rendering,
                'zarr_format': zarr_format,
                'name': name,
                'transform': transform,
            }
        ],
    }


PRESETS = {
    'cardiac': _remote(
        'Zebrafish heart FIB-SEM — 16 B voxels, 15 levels (COSEM)',
        's3://janelia-cosem-datasets/jrc_zf-cardiac-1/'
        'jrc_zf-cardiac-1.zarr/recon-1/em/fibsem-uint8',
        levels=15,
        contrast=(0, 255),
        colormap='turbo',
        threed=True,
        name='Zebrafish heart (FIB-SEM)',
    ),
    'zebrafish-em': _remote(
        'Zebrafish embryo sagittal EM — 350 Gpx, 8 levels (IDR)',
        'https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.1/4495402.zarr/',
        levels=8,
        contrast=(0, 255),
        colormap='cyan',
        zarr_format=2,
        name='Zebrafish embryo EM (350 Gpx)',
    ),
    'platynereis': _remote(
        'Platynereis whole-worm serial EM — 10 levels (EMBL)',
        'https://s3.embl.de/i2k-2020/platy-raw.ome.zarr',
        levels=10,
        contrast=(0, 255),
        colormap='green',
        threed=True,
        name='Platynereis (serial-section EM)',
    ),
    'zebrahub': _remote(
        'Zebrafish embryo light-sheet timelapse (CZ Biohub)',
        'https://public.czbiohub.org/royerlab/zebrahub/imaging/'
        'single-objective/ZSNS002.ome.zarr/',
        levels=4,
        contrast=(0, 1000),
        name='Zebrahub ZSNS002',
    ),
    'hela': _remote(
        'HeLa cell FIB-SEM — 6 levels (COSEM)',
        's3://janelia-cosem-datasets/jrc_hela-2/'
        'jrc_hela-2.zarr/recon-1/em/fibsem-uint8',
        levels=6,
        contrast=(0, 255),
        threed=True,
        name='HeLa cell (FIB-SEM)',
    ),
    'covid': _remote(
        'SARS-CoV-2 infected cell FIB-SEM, uint16 (COSEM)',
        's3://janelia-cosem-datasets/jrc_ccl81-covid-1/'
        'jrc_ccl81-covid-1.zarr/recon-1/em/fibsem-uint16',
        levels=5,
        contrast=(0, 65535),
        colormap='inferno',
        threed=True,
        name='SARS-CoV-2 cell (FIB-SEM)',
    ),
    'mouse-kidney': _remote(
        'Mouse kidney FIB-SEM (COSEM)',
        's3://janelia-cosem-datasets/jrc_mus-kidney/'
        'jrc_mus-kidney.zarr/recon-1/em/fibsem-uint8',
        levels=5,
        contrast=(0, 255),
        colormap='magma',
        threed=True,
        name='Mouse kidney (FIB-SEM)',
    ),
    'idr-em-2d': _remote(
        'SARS-CoV-2 intestinal organoid TEM — 13 Gpx (IDR)',
        'https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.4/idr0083A/9822152.zarr/',
        levels=11,
        contrast=(0, 65535),
        name='SARS-CoV-2 organoid TEM',
    ),
    'wsi': _remote(
        'CMU-1 whole-slide RGB image (Glencoe)',
        's3://gs-public-zarr-archive/CMU-1.ome.zarr',
        levels=5,
        contrast=(0, 255),
        zarr_format=2,
        name='CMU-1 (WSI)',
        transform='channels-first-rgb',
    ),
    'liver-labels': {
        'description': 'Mouse liver FIB-SEM with organelle labels (COSEM)',
        'threed': True,
        'layers': [
            {
                'path': 's3://janelia-cosem-datasets/jrc_mus-liver/'
                'jrc_mus-liver.zarr/recon-1/em/fibsem-uint8',
                'levels': 5,
                'contrast': (0, 255),
                'name': 'EM (FIB-SEM)',
            },
            {
                'path': 's3://janelia-cosem-datasets/jrc_mus-liver/'
                'jrc_mus-liver.zarr/recon-1/labels/groundtruth/crop124/all',
                'levels': 5,
                'layer_type': 'labels',
                'name': 'organelles (crop124)',
            },
        ],
    },
    'hela-labels': {
        'description': 'HeLa FIB-SEM with organelle labels (COSEM)',
        'threed': True,
        'layers': [
            {
                'path': 's3://janelia-cosem-datasets/jrc_hela-2/'
                'jrc_hela-2.zarr/recon-1/em/fibsem-uint8',
                'levels': 6,
                'contrast': (0, 255),
                'name': 'HeLa EM (FIB-SEM)',
            },
            {
                'path': 's3://janelia-cosem-datasets/jrc_hela-2/'
                'jrc_hela-2.zarr/recon-1/labels/groundtruth/crop155/all',
                'levels': 6,
                'layer_type': 'labels',
                'name': 'organelles (crop155)',
            },
        ],
    },
    'mandelbrot': {
        'description': 'Generative Mandelbrot set (local, lazy)',
        'layers': [
            {
                'generator': 'mandelbrot',
                'contrast': (0, 255),
                'colormap': 'twilight_shifted',
            }
        ],
    },
    'mandelbulb': {
        'description': 'Generative Mandelbulb (local, lazy)',
        'threed': True,
        'layers': [
            {
                'generator': 'mandelbulb',
                'contrast': (0, 64),
                'colormap': 'magma',
            }
        ],
    },
    'mandelbulb-rgb': {
        'description': 'Generative RGB Mandelbulb (local, lazy)',
        'threed': True,
        'layers': [{'generator': 'mandelbulb-rgb', 'rgb': True}],
    },
    'local-zarr': {
        'description': 'Materialized local Mandelbulb Zarr (builds once)',
        'threed': True,
        'layers': [
            {
                'generator': 'local-zarr',
                'contrast': (0, 64),
                'colormap': 'magma',
            }
        ],
    },
}


def list_presets() -> None:
    print('\nAvailable presets:\n')
    width = max(map(len, PRESETS))
    for name, preset in PRESETS.items():
        mode = '3D' if preset.get('threed') else '2D'
        print(f'  {name:<{width}}  [{mode}]  {preset["description"]}')
    print()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Open a preset or Zarr path with progressive loading',
    )
    parser.add_argument('source', nargs='?', help='preset name, URL, or path')
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--levels', type=int, help='number of pyramid levels')
    parser.add_argument(
        '--contrast', type=float, nargs=2, metavar=('LO', 'HI')
    )
    parser.add_argument('--colormap')
    parser.add_argument('--name')
    parser.add_argument('--cache-mb', type=int, default=4000)
    display = parser.add_mutually_exclusive_group()
    display.add_argument('--3d', dest='threed', action='store_true')
    display.add_argument('--2d', dest='threed', action='store_false')
    parser.set_defaults(threed=None)
    parser.add_argument('--rendering')
    parser.add_argument('--zarr-format', type=int, choices=(2, 3))
    parser.add_argument('--no-squeeze', action='store_true')
    parser.add_argument('--labels', action='store_true')
    parser.add_argument('--rgb', action='store_true')
    parser.add_argument('--tile-mib', type=int)
    parser.add_argument('--interval-mib', type=int)
    parser.add_argument('--rate-mib', type=float)
    parser.add_argument(
        '--screenshot',
        type=Path,
        help='save a screenshot after loading, then close the viewer',
    )
    parser.add_argument(
        '--screenshot-delay',
        type=float,
        default=30,
        metavar='SECONDS',
        help='time to progressively load before taking --screenshot',
    )
    return parser


def _generated(spec, cache_bytes):
    name = spec['generator']
    if name == 'mandelbrot':
        return mandelbrot_dataset(max_levels=14, cache_bytes=cache_bytes)[
            'arrays'
        ]
    if name == 'mandelbulb':
        return mandelbulb_dataset(
            max_levels=5,
            tilesize=32,
            maxiter=64,
            cache_bytes=cache_bytes,
        )['arrays']
    if name == 'mandelbulb-rgb':
        return mandelbulb_rgb_dataset(
            max_levels=5,
            tilesize=32,
            maxiter=64,
            cache_bytes=cache_bytes,
        )['arrays']
    return local_zarr_dataset(
        Path('mandelbulb.zarr'), cache_bytes=cache_bytes
    )['arrays']


def _open_layer(spec, arguments, cache_bytes):
    if 'generator' in spec:
        return _generated(spec, cache_bytes), None, None

    arrays, scale, translate = open_ome_zarr(
        spec['path'],
        num_levels=arguments.levels or spec.get('levels'),
        cache_bytes=cache_bytes,
        zarr_format=arguments.zarr_format or spec.get('zarr_format'),
        squeeze=not arguments.no_squeeze,
    )
    if spec.get('transform') == 'channels-first-rgb':
        import dask.array as da

        arrays = [da.asarray(array).transpose(1, 2, 0) for array in arrays]
        if scale is not None:
            scale = scale[-2:]
        if translate is not None:
            translate = translate[-2:]
        spec['rgb'] = True
    return arrays, scale, translate


def main(argv=None) -> None:
    parser = _parser()
    arguments = parser.parse_args(argv)
    if arguments.list:
        list_presets()
        return
    if arguments.source is None:
        parser.print_help()
        print('\nUse --list to see available presets.')
        return

    preset = PRESETS.get(arguments.source)
    if preset is None:
        preset = {
            'description': arguments.source,
            'layers': [
                {
                    'path': arguments.source,
                    'layer_type': 'labels' if arguments.labels else 'image',
                    'rgb': arguments.rgb,
                }
            ],
        }
    threed = (
        arguments.threed
        if arguments.threed is not None
        else preset.get('threed', False)
    )
    cache_bytes = arguments.cache_mb * 1_000_000
    viewer = napari.Viewer(title=f'Progressive: {arguments.source}')

    for original_spec in preset['layers']:
        spec = dict(original_spec)
        print(f'Opening {spec.get("path", spec.get("generator"))} ...')
        arrays, scale, translate = _open_layer(spec, arguments, cache_bytes)
        print(
            f'  {len(arrays)} levels, level 0: shape={arrays[0].shape} '
            f'dtype={arrays[0].dtype}'
        )
        layer_type = spec.get('layer_type', 'image')
        kwargs = {'name': arguments.name or spec.get('name')}
        if layer_type == 'image':
            contrast = arguments.contrast or spec.get('contrast')
            if contrast is not None:
                kwargs['contrast_limits'] = tuple(contrast)
            kwargs['colormap'] = arguments.colormap or spec.get(
                'colormap', 'gray'
            )
            kwargs['rgb'] = arguments.rgb or spec.get('rgb', False)
            if threed:
                kwargs['rendering'] = arguments.rendering or spec.get(
                    'rendering', 'attenuated_mip'
                )
        if scale is not None:
            kwargs['scale'] = scale
        if translate is not None:
            kwargs['translate'] = translate
        if arguments.tile_mib is not None:
            kwargs['tile_max_bytes_3d'] = arguments.tile_mib * MIB
        if arguments.interval_mib is not None:
            kwargs['interval_max_bytes'] = arguments.interval_mib * MIB
        if arguments.rate_mib is not None:
            kwargs['max_bytes_per_second'] = arguments.rate_mib * MIB

        factory = (
            add_lodstone_loading_labels
            if layer_type == 'labels'
            else add_lodstone_loading_image
        )
        factory(arrays, viewer=viewer, **kwargs)

    # An empty viewer has only two dimensions, so requesting 3-D before
    # adding the first layer is clamped back to 2-D.
    viewer.dims.ndisplay = 3 if threed else 2
    viewer.reset_view()
    if arguments.screenshot is not None:
        from qtpy.QtCore import QTimer

        screenshot = arguments.screenshot.expanduser().resolve()
        screenshot.parent.mkdir(parents=True, exist_ok=True)

        def save_screenshot() -> None:
            viewer.screenshot(screenshot, canvas_only=False)
            print(f'Saved screenshot to {screenshot}')
            for layer in tuple(viewer.layers):
                loader = layer.metadata.get('progressive_loader')
                if loader is not None:
                    loader.close()
            # Let callbacks queued before loader disconnection drain before
            # Qt tears down the viewer and its dimensions widgets.
            QTimer.singleShot(0, viewer.close)

        QTimer.singleShot(
            max(0, round(arguments.screenshot_delay * 1000)),
            save_screenshot,
        )
    napari.run()


if __name__ == '__main__':
    main()
