"""Compatibility exports for LodStone's bounded virtual array model."""

from lodstone import (
    MultiScaleVirtualData,
    VirtualArrayView,
    VirtualData,
    chunk_boundaries,
    chunk_ids_in_region,
    chunk_shape_for,
    chunk_sizes_for,
)

__all__ = [
    'MultiScaleVirtualData',
    'VirtualArrayView',
    'VirtualData',
    'chunk_boundaries',
    'chunk_ids_in_region',
    'chunk_shape_for',
    'chunk_sizes_for',
]
