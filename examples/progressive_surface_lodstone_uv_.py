# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "napari[pyqt6]",
#     "lodstone[mesh]>=0.1.0a3",
# ]
#
# [tool.uv.sources]
# napari = { git = "https://github.com/kephale/napari", branch = "lodstone-integration" }
# ///
"""
Progressive multiscale surface
==============================

Stream a surface stored at several resolutions. Lodstone chooses the level
from the 3D camera, culls chunks outside the view and replaces the whole
level at once, so different resolutions never meet at a seam::

    uv run examples/progressive_surface_lodstone_uv_.py

Zoom in to refine the surface and zoom out to coarsen it again; when zoomed
in, chunks outside the view are not loaded. The window title shows the
displayed level and triangle count.

.. tags:: experimental
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from lodstone.biological_mesh import write_surface_levels
from lodstone.sources.zarr_vectors import ZarrVectorsMeshSource

import napari
from napari.experimental._lodstone_meshes import add_lodstone_surface

RADIUS = 48.0
PIXEL_ERROR = 8.0


def _midpoint(
    points: list[np.ndarray], cache: dict[tuple[int, int], int], a: int, b: int
) -> int:
    """Index of the unit-sphere midpoint of edge (a, b), adding it once."""
    key = (min(a, b), max(a, b))
    if key not in cache:
        middle = points[a] + points[b]
        points.append(middle / np.linalg.norm(middle))
        cache[key] = len(points) - 1
    return cache[key]


def icosphere(subdivisions: int) -> tuple[np.ndarray, np.ndarray]:
    """A unit icosphere, subdividing each triangle into four per step."""
    phi = (1 + np.sqrt(5)) / 2
    vertices = [
        (-1, phi, 0), (1, phi, 0), (-1, -phi, 0), (1, -phi, 0),
        (0, -1, phi), (0, 1, phi), (0, -1, -phi), (0, 1, -phi),
        (phi, 0, -1), (phi, 0, 1), (-phi, 0, -1), (-phi, 0, 1),
    ]  # fmt: skip
    faces = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
    ]  # fmt: skip
    points = [np.asarray(v, float) / np.linalg.norm(v) for v in vertices]
    for _ in range(subdivisions):
        midpoints: dict[tuple[int, int], int] = {}
        refined = []
        for a, b, c in faces:
            ab, bc, ca = (
                _midpoint(points, midpoints, i, j)
                for i, j in ((a, b), (b, c), (c, a))
            )
            refined += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        faces = refined
    return np.asarray(points), np.asarray(faces, dtype=np.int64)


def bumpy_sphere(subdivisions: int) -> tuple[np.ndarray, np.ndarray]:
    """Fine ridges that only the finer levels can resolve, in XYZ order."""
    unit, faces = icosphere(subdivisions)
    x, y, z = unit.T
    ridges = 0.08 * np.sin(9 * x) * np.sin(11 * y) * np.sin(7 * z)
    ripples = 0.02 * np.sin(41 * x + 37 * y + 29 * z)
    vertices = unit * RADIUS * (1 + ridges + ripples)[:, None] + RADIUS * 1.5
    return vertices.astype(np.float32), faces


def main() -> None:
    subdivisions = (6, 4, 2)  # finest first
    # Geometric error of each level: about one triangle edge, in world units.
    errors = tuple(RADIUS * 1.1 / 2**s for s in subdivisions)
    scratch = tempfile.TemporaryDirectory(prefix='lodstone-surface-')
    root = write_surface_levels(
        Path(scratch.name), [[(*bumpy_sphere(s), 1)] for s in subdivisions]
    )
    source = ZarrVectorsMeshSource(root, level_errors=errors)

    viewer = napari.Viewer(ndisplay=3)
    layer = add_lodstone_surface(
        source,
        viewer,
        # A level is used until its error spans this many screen pixels.
        pixel_error=PIXEL_ERROR,
        name='bumpy sphere',
        colormap='viridis',
        shading='smooth',
    )
    loader = layer.metadata['lodstone_loader']
    # Start a little zoomed out, on the middle level.
    viewer.scene.camera.zoom *= 0.25

    def show_level(*_args) -> None:
        level = loader.target.level
        if level is not None:
            viewer.title = f'level {level}: {loader.target.faces:,} triangles'

    layer.events.data.connect(show_level)
    try:
        napari.run()
    finally:
        loader.close()
        source.close()
        scratch.cleanup()


if __name__ == '__main__':
    main()
