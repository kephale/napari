"""Lodstone planning and execution for napari's progressive renderer.

Napari captures camera events and retains ownership of bounded ``VirtualData``
intervals, backdrops, and GPU texture updates. Each prepared interval is
represented as a real Lodstone ``View`` and planned by Lodstone. The previous
napari plan is retained as a comparison trace while Lodstone owns tile order,
chunk reads, decoded caching, batching, cancellation, and stale-pass rejection.
"""

from __future__ import annotations

import itertools
import logging
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

import numpy as np
from lodstone import (
    Layout,
    Plan,
    Planner,
    Region,
    Stream,
    Tile,
    TileKey,
    View,
)
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
    chunk_priority_2D,
    chunk_slices,
)
from napari.experimental._virtual_data import MultiScaleVirtualData

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PlanTrace:
    """Renderer-neutral summary used to compare two planners."""

    target_level: int
    tiles: tuple[tuple[int, tuple[int, ...], tuple[int, ...], int], ...]
    wanted: tuple[tuple[int, tuple[int, ...], tuple[int, ...], int], ...]

    @staticmethod
    def _tiles(
        tiles: tuple[Tile, ...],
    ) -> tuple[tuple[int, tuple[int, ...], tuple[int, ...], int], ...]:
        return tuple(
            (tile.level, tile.region.start, tile.region.stop, tile.phase)
            for tile in tiles
        )

    @classmethod
    def from_plan(cls, plan: Plan) -> PlanTrace:
        tiles = plan.desired or plan.wanted
        return cls(
            target_level=plan.target_level,
            tiles=cls._tiles(tiles),
            wanted=cls._tiles(plan.wanted),
        )


@dataclass(frozen=True)
class PlanComparison:
    """One napari/Lodstone planning comparison for a captured view."""

    view: View
    napari: PlanTrace
    lodstone: PlanTrace

    @property
    def matches(self) -> bool:
        return self.napari == self.lodstone

    @property
    def geometry_matches(self) -> bool:
        """Whether level, regions, and ladder phases agree, ignoring order."""
        return (
            self.napari.target_level == self.lodstone.target_level
            and frozenset(self.napari.tiles) == frozenset(self.lodstone.tiles)
            and frozenset(self.napari.wanted)
            == frozenset(self.lodstone.wanted)
        )


def _level_transforms(layer, data) -> list[np.ndarray]:
    """Map every pyramid level into napari world coordinates."""
    data_to_world = np.asarray(layer._data_to_world.affine_matrix, dtype=float)
    transforms = []
    for factors in data._scale_factors:
        level_to_finest = np.eye(data.ndim + 1, dtype=float)
        level_to_finest[:-1, :-1] = np.diag(np.asarray(factors, dtype=float))
        transforms.append(data_to_world @ level_to_finest)
    return transforms


def _camera_view(
    viewer,
    layer,
    shape: tuple[int, ...],
    depth_span: float,
) -> View:
    """Capture napari's orthographic camera as a Lodstone ``View``."""
    ndim = len(shape)
    displayed = tuple(int(axis) for axis in layer._slice_input.displayed)
    ndisplay = len(displayed)
    viewport = tuple(max(int(value), 1) for value in viewer.canvas.size)
    camera = viewer.camera
    center = np.asarray(camera.center, dtype=float)[-ndisplay:]
    zoom = float(camera.zoom)
    if not np.isfinite(zoom) or zoom <= 0 or not np.all(np.isfinite(center)):
        zoom = np.finfo(float).eps
        center = np.zeros(ndisplay, dtype=float)

    matrix = np.eye(4, dtype=float)
    matrix[:3, :] = 0.0
    if ndisplay == 2:
        scales = 2.0 * zoom / np.asarray(viewport, dtype=float)
        matrix[0, 0] = scales[0]
        matrix[1, 1] = scales[1]
        matrix[0, 3] = -scales[0] * center[0]
        matrix[1, 3] = -scales[1] * center[1]
        eye = None
    else:
        view_direction = np.asarray(camera.view_direction, dtype=float)
        up_direction = np.asarray(camera.up_direction, dtype=float)
        view_direction /= max(np.linalg.norm(view_direction), 1e-12)
        up_direction /= max(np.linalg.norm(up_direction), 1e-12)
        right_direction = np.cross(up_direction, view_direction)
        right_direction /= max(np.linalg.norm(right_direction), 1e-12)
        rows = (up_direction, right_direction, view_direction)
        scales = (
            2.0 * zoom / viewport[0],
            2.0 * zoom / viewport[1],
            2.0 / max(float(depth_span), 1.0),
        )
        for row, (direction, scale) in enumerate(
            zip(rows, scales, strict=True)
        ):
            matrix[row, :3] = direction * scale
            matrix[row, 3] = -float(np.dot(direction, center)) * scale
        eye = tuple(float(value) for value in center)

    index = tuple(None if axis in displayed else 0 for axis in range(ndim))
    try:
        data_point = np.asarray(
            layer.world_to_data(viewer.dims.point), dtype=float
        )
        index = tuple(
            None
            if axis in displayed
            else min(max(round(data_point[axis]), 0), shape[axis] - 1)
            for axis in range(ndim)
        )
    except (IndexError, TypeError, ValueError):
        pass
    return View(displayed, index, viewport, matrix, eye=eye)


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
        return Layout(
            kind='bricked',
            memory_limit=self.loader._interval_max_bytes,
            squeeze_hidden=False,
        )

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
        lod_bias = float(kwargs.get('max_pixel_size_3d', 2.0))
        kwargs['_start_immediately'] = False
        super().__init__(viewer, layer, data, **kwargs)

        source = ArrayPyramidSource(
            data.arrays,
            chunks=[level.chunk_shape for level in data],
            transforms=_level_transforms(layer, data),
        )
        self._lodstone_source = source
        self._shared_planners = {
            2: Planner(lod_bias=1.0),
            3: Planner(lod_bias=lod_bias),
        }
        self._plan_comparisons: deque[PlanComparison] = deque(maxlen=32)
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

        target = self._stages[-1][0]
        desired_stages = self._desired_stages(target)
        phases = {
            level: phase
            for phase, (level, _queue) in enumerate(desired_stages)
        }
        wanted = tuple(
            tile
            for level, queue in self._stages
            for tile in self._plan_tiles(level, queue, phases[level])
        )
        desired = tuple(
            tile
            for phase, (level, queue) in enumerate(desired_stages)
            for tile in self._plan_tiles(level, queue, phase)
        )
        plan = Plan(
            wanted=wanted,
            retain=frozenset(
                tile.key for tile in desired if tile.level == target
            ),
            target_level=target,
            desired=desired,
        )
        view = self._pass_view(desired)
        shared = self._shared_planners[len(view.displayed_axes)].plan(
            self._lodstone_source.pyramid,
            view,
            self._lodstone_target.layout(view, self._lodstone_source.pyramid),
            available=self._available_keys(view),
        )
        comparison = PlanComparison(
            view,
            PlanTrace.from_plan(plan),
            PlanTrace.from_plan(shared),
        )
        self._plan_comparisons.append(comparison)
        LOGGER.debug(
            'planner trace: matches=%s geometry_matches=%s '
            'napari=(level=%d tiles=%d) '
            'lodstone=(level=%d tiles=%d)',
            comparison.matches,
            comparison.geometry_matches,
            comparison.napari.target_level,
            len(comparison.napari.tiles),
            comparison.lodstone.target_level,
            len(comparison.lodstone.tiles),
        )
        # Planning authority transfers here. Napari's plan above remains in
        # the comparison trace while all renderer/residency behavior stays on
        # the existing ProgressiveLoader path.
        self._lodstone_planner.next_plan = shared
        self._lodstone_target.generation = generation
        self._worker = _WorkerProxy(self._lodstone_stream)
        self._lodstone_stream.update(view)

    @staticmethod
    def _plan_tiles(level: int, queue, phase: int) -> tuple[Tile, ...]:
        result = []
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
            result.append(Tile(tile_key, region, float(priority), phase))
        return tuple(result)

    def _desired_stages(self, target: int) -> list[tuple[int, list]]:
        """Reconstruct napari's complete ladder, including cached chunks."""
        active = self._active
        if active is None:
            return list(self._stages)
        _active_level, target_min, target_max = active
        factors = self._data._scale_factors
        stages = []
        for level in range(self._resident_level, target - 1, -1):
            vdata = self._data[level]
            if level == target:
                region_min = np.asarray(target_min, dtype=np.int64)
                region_max = np.asarray(target_max, dtype=np.int64)
            else:
                ratio = np.asarray(factors[target]) / np.asarray(
                    factors[level]
                )
                region_min = np.floor(np.asarray(target_min) * ratio).astype(
                    np.int64
                )
                region_max = np.ceil(np.asarray(target_max) * ratio).astype(
                    np.int64
                )
                region_min = np.clip(region_min, 0, vdata.shape)
                region_max = np.clip(region_max, 0, vdata.shape)
                region_min, region_max = self._clamp_interval(
                    vdata, region_min, region_max
                )
            keys = chunk_slices(
                vdata,
                interval=(region_min, region_max),
            )
            if self._viewer.dims.ndisplay == 3:
                queue = self._prioritize_3d(
                    level,
                    keys,
                    (region_min, region_max),
                )
            else:
                queue = chunk_priority_2D(keys, region_min, region_max)
            stages.append((level, queue))
        return stages

    def _available_keys(self, view: View) -> frozenset[TileKey]:
        """Translate napari's loaded-chunk bookkeeping to Lodstone keys."""
        selection = tuple(
            -1 if value is None else int(value) for value in view.index
        )
        keys = set()
        for level, vdata in enumerate(self._data):
            chunks = self._lodstone_source.pyramid.levels[level].chunks
            chunk_ids = set(vdata.loaded_chunks)
            if level == self._resident_level:
                # The PR's dedicated resident worker owns these reads. Treat
                # its complete target interval as available even while the
                # worker is still filling it, so Lodstone does not duplicate
                # coarsest-level I/O in the foreground pass.
                target = self._resident_target_interval()
                if target is not None:
                    resident_keys = chunk_slices(vdata, interval=target)
                    chunk_ids.update(
                        tuple(
                            (int(item.start), int(item.stop)) for item in key
                        )
                        for key in itertools.product(*resident_keys)
                    )
            for chunk_id in chunk_ids:
                starts = tuple(int(start) for start, _stop in chunk_id)
                grid_index = tuple(
                    starts[axis] // chunks[axis]
                    for axis in view.displayed_axes
                )
                keys.add(TileKey(level, grid_index, selection))
        return frozenset(keys)

    def _pass_view(self, tiles: list[Tile]) -> View:
        transforms = self._lodstone_source.pyramid.levels[0].voxel_to_world
        extent = np.asarray(self._data.shape, dtype=float)
        world_extent = np.abs(transforms[:-1, :-1]) @ extent
        displayed = self._layer._slice_input.displayed
        depth_span = float(np.linalg.norm(np.take(world_extent, displayed)))
        return _camera_view(
            self._viewer,
            self._layer,
            self._data.shape,
            depth_span,
        )

    @property
    def plan_comparisons(self) -> tuple[PlanComparison, ...]:
        """Recent PR/shared-planner traces, oldest first."""
        return tuple(self._plan_comparisons)

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
