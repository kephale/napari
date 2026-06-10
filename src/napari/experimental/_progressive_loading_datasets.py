"""Example multiscale datasets for progressive loading.

Each function returns a dictionary describing a multiscale image with (at
least) the keys ``'arrays'`` (list of array-like levels, highest resolution
first) and ``'scale_factors'``. The ``'arrays'`` entry can be passed
directly to
:func:`napari.experimental._progressive_loading.add_progressive_loading_image`.

The generative datasets (:func:`mandelbrot_dataset`,
:func:`mandelbulb_dataset`) have no dependencies beyond zarr (numba is
used when available). The remote datasets require optional extras noted in
their docstrings and are imported lazily.
"""

from __future__ import annotations

import logging

import zarr
from zarr.experimental.cache_store import CacheStore
from zarr.storage import MemoryStore

from napari.experimental._generative_zarr import (
    MandelbrotStore,
    MandelbulbStore,
)

LOGGER = logging.getLogger(__name__)

#: Default size of the in-memory chunk cache for generative datasets.
DEFAULT_CACHE_BYTES = int(4e9)


def _open_cached_multiscale(store, levels: int, cache_bytes: int):
    """Open a multiscale group through an in-memory chunk cache."""
    cached = CacheStore(store, cache_store=MemoryStore(), max_size=cache_bytes)
    group = zarr.open_group(cached, mode='r')
    return [group[str(level)] for level in range(levels)]


def mandelbrot_dataset(
    max_levels: int = 14,
    tilesize: int = 512,
    maxiter: int = 255,
    cache_bytes: int = DEFAULT_CACHE_BYTES,
    cpu_relief: float = 0.5,
):
    """Generate a multiscale 2D image of the Mandelbrot set.

    Scale 0 is the highest resolution (``tilesize * 2 ** max_levels``
    pixels wide); chunks are computed on demand and cached in memory.

    >>> large_image = mandelbrot_dataset(max_levels=14)
    >>> add_progressive_loading_image(large_image['arrays'], viewer=viewer)

    Parameters
    ----------
    max_levels : int
        Number of levels (scales) to generate.
    tilesize : int
        Chunk edge length.
    maxiter : int
        Maximum escape-time iterations (determines dtype).
    cache_bytes : int
        Size of the in-memory chunk cache.

    Returns
    -------
    dict
        Multiscale metadata with keys ``['container', 'dataset',
        'scale_levels', 'scale_factors', 'chunk_size', 'arrays']``.
    """
    store = MandelbrotStore(
        levels=max_levels,
        tilesize=tilesize,
        maxiter=maxiter,
        cpu_relief=cpu_relief,
    )
    arrays = _open_cached_multiscale(store, max_levels, cache_bytes)
    return {
        'container': 'mandelbrot.zarr/',
        'dataset': '',
        'scale_levels': max_levels,
        'scale_factors': [(2**level, 2**level) for level in range(max_levels)],
        'chunk_size': (tilesize, tilesize),
        'arrays': arrays,
    }


def mandelbulb_dataset(
    max_levels: int = 6,
    tilesize: int = 32,
    maxiter: int = 255,
    order: int = 8,
    cache_bytes: int = DEFAULT_CACHE_BYTES,
    cpu_relief: float = 0.5,
):
    """Generate a multiscale 3D image of a Mandelbulb.

    Parameters
    ----------
    max_levels : int
        Number of levels (scales) to generate.
    tilesize : int
        Chunk edge length (chunks are ``tilesize**3`` voxels).
    maxiter : int
        Maximum escape-time iterations (determines dtype).
    order : int
        Order of the Mandelbulb equation.
    cache_bytes : int
        Size of the in-memory chunk cache.

    Returns
    -------
    dict
        Multiscale metadata with keys ``['container', 'dataset',
        'scale_levels', 'scale_factors', 'chunk_size', 'arrays']``.
    """
    store = MandelbulbStore(
        levels=max_levels,
        tilesize=tilesize,
        maxiter=maxiter,
        order=order,
        cpu_relief=cpu_relief,
    )
    arrays = _open_cached_multiscale(store, max_levels, cache_bytes)
    return {
        'container': 'mandelbulb.zarr/',
        'dataset': '',
        'scale_levels': max_levels,
        'scale_factors': [
            (2**level, 2**level, 2**level) for level in range(max_levels)
        ],
        'chunk_size': (tilesize, tilesize, tilesize),
        'arrays': arrays,
    }


def openorganelle_mouse_kidney_em():
    """Mouse kidney FIB-SEM data from OpenOrganelle (remote, ~TB scale).

    Requires the optional ``fibsem_tools`` package.
    """
    try:
        from fibsem_tools import read_xarray
    except ModuleNotFoundError as e:  # pragma: no cover - optional dep
        raise ModuleNotFoundError(
            'openorganelle_mouse_kidney_em requires fibsem_tools: '
            'pip install fibsem_tools'
        ) from e

    large_image = {
        'container': 's3://janelia-cosem-datasets/jrc_mus-kidney/jrc_mus-kidney.n5',
        'dataset': 'em/fibsem-uint8',
        'scale_levels': 5,
        'scale_factors': [
            (1, 1, 1),
            (2, 2, 2),
            (4, 4, 4),
            (8, 8, 8),
            (16, 16, 16),
        ],
    }
    large_image['arrays'] = [
        read_xarray(
            f'{large_image["container"]}/{large_image["dataset"]}/s{scale}/',
            storage_options={'anon': True},
        ).data
        for scale in range(large_image['scale_levels'])
    ]
    return large_image


def luethi_zenodo_7144919(cache_bytes: int = DEFAULT_CACHE_BYTES):
    """Multiscale OME-Zarr of cardiomyocyte differentiation (Zenodo 7144919).

    Downloads ~600 MB on first use (cached by pooch). Requires the optional
    ``pooch`` package.
    """
    import os

    try:
        import pooch
    except ModuleNotFoundError as e:  # pragma: no cover - optional dep
        raise ModuleNotFoundError(
            'luethi_zenodo_7144919 requires pooch: pip install pooch'
        ) from e

    # Downloaded from https://zenodo.org/record/7144919
    dest_dir = pooch.retrieve(
        url='https://zenodo.org/record/7144919/files/20200812-CardiomyocyteDifferentiation14-Cycle1.zarr.zip?download=1',
        known_hash='e6773fc97dcf3689e2f42e6504e0d4f4d0845c329dfbdfe92f61c2f3f1a4d55d',
        processor=pooch.Unzip(),
    )
    local_container = os.path.split(dest_dir[0])[0]

    large_image = {
        'container': local_container,
        'dataset': 'B/03/0',
        'scale_levels': 5,
        'scale_factors': [
            (1, 0.1625, 0.1625),
            (1, 0.325, 0.325),
            (1, 0.65, 0.65),
            (1, 1.3, 1.3),
            (1, 2.6, 2.6),
        ],
        'chunk_size': (1, 10, 256, 256),
    }

    store = CacheStore(
        zarr.storage.LocalStore(local_container),
        cache_store=MemoryStore(),
        max_size=cache_bytes,
    )
    group = zarr.open_group(store, mode='r')
    multiscale_data = group[large_image['dataset']]
    large_image['arrays'] = [
        multiscale_data[str(scale)]
        for scale in range(large_image['scale_levels'])
    ]
    return large_image
