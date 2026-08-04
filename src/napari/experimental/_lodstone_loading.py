"""Lodstone execution backend for napari's progressive renderer.

The progressive renderer remains responsible for camera tracking, bounded
``VirtualData`` intervals, backdrops, and GPU texture updates. Lodstone owns
chunk reads, decoded caching, batching, cancellation, stale-pass rejection,
and interaction pause/resume.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import numpy as np
from lodstone import Layout, Plan, Region, Stream, Tile, TileKey, View
from lodstone.sources import ArrayPyramidSource
from qtpy.QtCore import QObject, Qt, Signal, Slot

from napari.experimental._progressive_loading import (
    DEFAULT_INTERVAL_MAX_BYTES,
    DEFAULT_TILE_MAX_BYTES_3D,
    ProgressiveLoader,
    _attach_progressive_loader,
    _estimate_contrast_limits,
    _normalize_scale_for_float32,
    _resolve_viewer_and_tile_cap,
)
from napari.experimental._virtual_data import MultiScaleVirtualData

LOGGER = logging.getLogger(__name__)


class _QtDispatcher(QObject):
    requested = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self.requested.connect(self._run, Qt.ConnectionType.QueuedConnection)

    @Slot(object)
    def _run(self, callback) -> None:
        callback()

    def dispatch(self, callback) -> None:
        self.requested.emit(callback)


class _PassPlanner:
    def __init__(self) -> None:
        self.next_plan: Plan | None = None

    def plan(self, *_args, **_kwargs) -> Plan:
        if self.next_plan is None:
            raise RuntimeError('napari did not prepare a Lodstone pass')
        return self.next_plan


class _WorkerProxy:
    def __init__(self, stream: Stream) -> None:
        self._stream = stream

    def quit(self) -> None:
        self._stream.cancel()


class _NapariTarget:
    def __init__(self, loader: LodstoneProgressiveLoader) -> None:
        self.loader = loader
        self.generation = 0

    def layout(self, view, pyramid) -> Layout:
        return Layout(kind='bricked', squeeze_hidden=False)

    def prepare(self, view, plan) -> None:
        return None

    def apply(self, updates) -> None:
        grouped: dict[int, list[tuple[slice, ...]]] = defaultdict(list)
        for update in updates:
            vdata = self.loader._data[update.level]
            key = update.region.slices()
            vdata.set_offset(key, update.data)
            chunk_id = tuple(
                (int(start), int(stop))
                for start, stop in zip(
                    update.region.start,
                    update.region.stop,
                    strict=True,
                )
            )
            vdata.loaded_chunks.add(chunk_id)
            vdata.chunk_source[chunk_id] = update.level
            grouped[update.level].append(key)
        for level, batch in grouped.items():
            self.loader._on_chunks(
                self.generation,
                self.loader._data[level],
                batch,
            )

    def discard(self, keys) -> None:
        # VirtualData interval changes own physical residency. Lodstone's
        # discard only retires logical pass keys and decoded cache entries.
        return None

    def complete(self, view, plan) -> None:
        self.loader._on_fetch_finished(self.generation)

    def redraw(self) -> None:
        return None


class LodstoneProgressiveLoader(ProgressiveLoader):
    """ProgressiveLoader whose fetch passes execute through Lodstone."""

    def __init__(self, viewer, layer, data, **kwargs) -> None:
        self._lodstone_enabled = False
        self._lodstone_stream: Stream | None = None
        kwargs['_start_immediately'] = False
        super().__init__(viewer, layer, data, **kwargs)

        source = ArrayPyramidSource(
            data.arrays,
            chunks=[level.chunk_shape for level in data],
        )
        self._lodstone_dispatcher = _QtDispatcher()
        self._lodstone_planner = _PassPlanner()
        self._lodstone_target = _NapariTarget(self)
        self._lodstone_stream = Stream(
            source,
            self._lodstone_target,
            planner=self._lodstone_planner,
            dispatch=self._lodstone_dispatcher.dispatch,
            workers=self._fetch_workers,
            bytes_per_second=self._max_bytes_per_second,
        )
        self._lodstone_stream.on_status_changed(self._on_lodstone_status)
        self._lodstone_enabled = True
        self._check()

    def _start_stage(self, generation: int, index: int) -> None:
        if not self._lodstone_enabled or self._lodstone_stream is None:
            super()._start_stage(generation, index)
            return

        tiles: list[Tile] = []
        target = self._stages[-1][0]
        for phase, (level, queue) in enumerate(self._stages):
            for priority, key in enumerate(queue):
                region = Region(
                    tuple(int(item.start) for item in key),
                    tuple(int(item.stop) for item in key),
                )
                tile_key = TileKey(
                    level,
                    tuple(int(item.start) for item in key),
                    (),
                )
                tiles.append(Tile(tile_key, region, float(priority), phase))
        plan = Plan(
            wanted=tuple(tiles),
            retain=frozenset(tile.key for tile in tiles if tile.level == target),
            target_level=target,
            desired=tuple(tiles),
        )
        self._lodstone_planner.next_plan = plan
        self._lodstone_target.generation = generation
        self._worker = _WorkerProxy(self._lodstone_stream)
        self._lodstone_stream.update(self._pass_view(tiles))

    def _pass_view(self, tiles: list[Tile]) -> View:
        displayed = tuple(int(axis) for axis in self._layer._slice_input.displayed)
        ndim = self._data.ndim
        first = tiles[-1].region if tiles else None
        index = tuple(
            None
            if axis in displayed
            else int(first.start[axis])
            if first is not None
            else 0
            for axis in range(ndim)
        )
        return View(
            displayed_axes=displayed,
            index=index,
            viewport=(1, 1),
            world_to_clip=np.eye(4, dtype=np.float64),
        )

    def _on_lodstone_status(self, status) -> None:
        if status.state == 'failed':
            LOGGER.error(
                'Lodstone progressive pass failed: %s',
                status.error,
                exc_info=(
                    type(status.error),
                    status.error,
                    status.error.__traceback__,
                ),
            )

    def _on_interaction(self, event=None) -> None:
        super()._on_interaction(event)
        if self._lodstone_enabled and self._lodstone_stream is not None:
            self._lodstone_stream.pause()

    def _end_hold(self) -> None:
        super()._end_hold()
        if self._lodstone_enabled and self._lodstone_stream is not None:
            self._lodstone_stream.resume()

    def close(self) -> None:
        stream = self._lodstone_stream
        self._lodstone_stream = None
        self._lodstone_enabled = False
        if stream is not None:
            stream.close()
        super().close()


def add_lodstone_loading_image(
    img,
    viewer=None,
    contrast_limits=None,
    colormap='gray',
    rendering='attenuated_mip',
    name=None,
    auto_level_3d=True,
    max_pixel_size_3d=2.0,
    interval_max_bytes=DEFAULT_INTERVAL_MAX_BYTES,
    tile_max_bytes_3d=DEFAULT_TILE_MAX_BYTES_3D,
    max_bytes_per_second=None,
    interaction_hold=True,
    interactive_step_rate=4.0,
    coarse_first=True,
    debug_overlay=None,
    **layer_kwargs: Any,
):
    """Add a progressive image rendered by napari and loaded by Lodstone."""
    viewer, tile_max_bytes_3d = _resolve_viewer_and_tile_cap(
        viewer,
        tile_max_bytes_3d,
    )
    data = MultiScaleVirtualData(img)
    _normalize_scale_for_float32(data, layer_kwargs, 'image')
    if contrast_limits is None:
        contrast_limits = _estimate_contrast_limits(data.arrays[-1])

    from napari.layers import Image

    layer = Image(
        data._data,
        multiscale=True,
        contrast_limits=contrast_limits,
        colormap=colormap,
        rendering=rendering,
        name=name,
        **layer_kwargs,
    )
    _attach_progressive_loader(
        layer,
        data,
        viewer,
        interval_max_bytes=interval_max_bytes,
        tile_max_bytes_3d=tile_max_bytes_3d,
        auto_level_3d=auto_level_3d,
        max_pixel_size_3d=max_pixel_size_3d,
        max_bytes_per_second=max_bytes_per_second,
        interaction_hold=interaction_hold,
        interactive_step_rate=interactive_step_rate,
        coarse_first=coarse_first,
        debug_overlay=debug_overlay,
        loader_class=LodstoneProgressiveLoader,
    )
    return layer
