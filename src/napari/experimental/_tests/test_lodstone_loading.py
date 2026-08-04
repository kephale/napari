from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from lodstone import Region, TileKey, Update

from napari.experimental._lodstone_loading import _NapariTarget


class _VirtualData:
    def __init__(self) -> None:
        self.values = np.zeros((2, 4, 4), dtype=np.uint16)
        self.loaded_chunks = set()
        self.chunk_source = {}

    def set_offset(self, key, value) -> None:
        self.values[key] = value


def test_target_applies_full_nd_chunk_before_render_callback() -> None:
    vdata = _VirtualData()
    delivered = []
    loader = SimpleNamespace(
        _data=[vdata],
        _on_chunks=lambda generation, level, batch: delivered.append(
            (generation, level, batch),
        ),
    )
    target = _NapariTarget(loader)
    target.generation = 7
    region = Region((1, 0, 0), (2, 4, 4))
    update = Update(
        TileKey(0, (1, 0, 0), ()),
        region,
        np.ones(region.shape, dtype=np.uint16),
        np.eye(4),
    )

    target.apply([update])

    assert np.all(vdata.values[1] == 1)
    assert ((1, 2), (0, 4), (0, 4)) in vdata.loaded_chunks
    assert delivered == [(7, vdata, [region.slices()])]
