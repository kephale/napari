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
    StreamDiagnostics,
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
    _chunk_id,
    _estimate_contrast_limits,
    _normalize_scale_for_float32,
    _pack_upload_block,
    _resolve_viewer_and_tile_cap,
    chunk_priority_2D,
    chunk_slices,
)
from napari.experimental._virtual_data import (
    MultiScaleVirtualData,
    chunk_sizes_for,
)

LOGGER = logging.getLogger(__name__)

MISSING_LEVEL_LABEL = 1
LEVEL_LABEL_OFFSET = 2
LEVEL_DIAGNOSTIC_COLORS = (
    '#28d34f',  # L0: green
    '#f5d90a',  # L1: yellow
    '#ff851b',  # L2: orange
    '#26c6da',  # L3: cyan
    '#7e57c2',  # L4: purple
    '#ef5350',  # L5: red
    '#8d6e63',  # L6: brown
    '#bdbdbd',  # L7: grey
)
LEVEL_DIAGNOSTIC_COLOR_NAMES = (
    'green',
    'yellow',
    'orange',
    'cyan',
    'purple',
    'red',
    'brown',
    'grey',
)


class _LevelDiagnosticArray:
    """Read a real array but return a solid categorical level label."""

    def __init__(self, array, level: int) -> None:
        self._array = array
        self.level = int(level)
        self.shape = tuple(int(value) for value in array.shape)
        self.ndim = len(self.shape)
        self.size = int(np.prod(self.shape, dtype=np.int64))
        self.dtype = np.dtype(np.uint8)
        self.fill_value = MISSING_LEVEL_LABEL

    def __getattr__(self, name):
        return getattr(self._array, name)

    def __getitem__(self, key):
        result = self._array[key]
        if hasattr(result, 'read'):
            result = result.read().result()
        if hasattr(result, 'compute'):
            result = result.compute()
        shape = np.asarray(result).shape
        return np.full(
            shape,
            self.level + LEVEL_LABEL_OFFSET,
            dtype=self.dtype,
        )


def _level_diagnostic_color_map(levels: int) -> dict[int | None, str]:
    colors: dict[int | None, str] = {
        None: '#00000000',
        MISSING_LEVEL_LABEL: '#ff00cc',
    }
    for level in range(levels):
        colors[level + LEVEL_LABEL_OFFSET] = LEVEL_DIAGNOSTIC_COLORS[
            level % len(LEVEL_DIAGNOSTIC_COLORS)
        ]
    return colors


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
    scene = getattr(viewer, 'scene', None)
    camera = scene.camera if scene is not None else viewer.camera
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

    def stage(self, updates):
        """Write and pack chunks on Lodstone's worker thread."""
        grouped: dict[int, list[tuple[slice, ...]]] = defaultdict(list)
        for update in updates:
            vdata = self.loader._data[update.level]
            key = update.region.slices()
            if vdata.set_chunk(key, update.data, update.level):
                grouped[update.level].append(key)
        active = getattr(self.loader, '_active', None)
        target = active[0] if active is not None else None
        return tuple(
            (
                level,
                batch,
                _pack_upload_block(self.loader._data[level], batch)
                if level == target
                else None,
            )
            for level, batch in grouped.items()
        )

    def apply(self, staged) -> None:
        # Keep direct calls useful for adapters and tests; Stream calls
        # ``stage`` before dispatching here, so production CPU work remains
        # off the Qt thread.
        if staged and not isinstance(staged[0], tuple):
            staged = self.stage(staged)
        for level, batch, block in staged:
            self.loader._on_chunks(
                self.generation,
                self.loader._data[level],
                (batch, block) if block is not None else batch,
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
            chunks=[chunk_sizes_for(array) for array in data.arrays],
            transforms=_level_transforms(layer, data),
        )
        self._lodstone_source = source
        self._shared_planners = {
            2: Planner(lod_bias=1.0),
            3: Planner(lod_bias=lod_bias),
        }
        self._plan_comparisons: deque[PlanComparison] = deque(maxlen=32)
        self._execution_diagnostics: deque[StreamDiagnostics] = deque(
            maxlen=32
        )
        self._lodstone_dispatcher = _QtDispatcher()
        self._submitted_plan: Plan | None = None
        self._lodstone_target = _NapariTarget(self)
        self._lodstone_stream = Stream(
            source,
            self._lodstone_target,
            dispatch=self._lodstone_dispatcher.dispatch,
            workers=self._fetch_workers,
            bytes_per_second=self._max_bytes_per_second,
        )
        self._lodstone_stream.on_status_changed(self._on_lodstone_status)
        self._lodstone_enabled = True
        self._check()

    def _ensure_persistent_overview(self) -> None:
        """Attach the full coarsest level as the volume's fallback texture."""
        if self._viewer.dims.ndisplay != 3 or self._data.ndim != 3:
            return
        node = self._get_volume_node()
        if node is None or node.clipmap_enabled:
            return
        target = self._resident_target_interval()
        if target is None:
            return
        overview = self._data[self._resident_level]
        low, high = target
        if np.any(low) or np.any(high != np.asarray(overview.shape)):
            return
        with overview.lock:
            if overview.interval is None:
                data = np.full(
                    overview.shape,
                    overview.fill_value,
                    dtype=overview.dtype,
                )
            else:
                data = np.ascontiguousarray(overview.hyperslice)
                if data.shape != overview.shape:
                    full = np.full(
                        overview.shape,
                        overview.fill_value,
                        dtype=overview.dtype,
                    )
                    low, high = overview.interval
                    full[
                        tuple(
                            slice(a, b) for a, b in zip(low, high, strict=True)
                        )
                    ] = data
                    data = full
        node.enable_clipmap(data, self._data[0].shape)
        displayed = list(self._layer._slice_input.displayed)
        if len(displayed) == 3:
            scale = self._layer.downsample_factors[self._layer.data_level][
                displayed
            ]
            node.set_clipmap_detail_bounds(
                self._layer.corner_pixels[0, displayed],
                self._layer.corner_pixels[1, displayed] + 1,
                scale=scale,
            )
        try:
            visual = self._viewer.window._qt_viewer.layer_to_visual[
                self._layer
            ]
            visual._on_matrix_change()
        except (KeyError, AttributeError, RuntimeError):
            pass

    def _pack_resident_batch(self, vdata, keys):
        if (
            self._closed
            or self._viewer is None
            or not isinstance(self._data, MultiScaleVirtualData)
            or self._viewer.dims.ndisplay != 3
            or self._data.ndim != 3
        ):
            return keys
        return keys, _pack_upload_block(vdata, keys)

    def _on_resident_chunks(self, vdata, batch) -> None:
        block = None
        if isinstance(batch, tuple):
            _keys, block = batch
        node = self._get_volume_node()
        if node is not None and node.clipmap_enabled and block is not None:
            low, _high, data = block
            node.set_overview_data(data, offset=low)
        if self._layer.data_level == self._resident_level:
            self._refresh()

    def _start_resident_fill(self, min_coord, max_coord) -> None:
        vdata = self._data[self._resident_level]
        dimensions = chunk_slices(vdata, interval=(min_coord, max_coord))
        desired = 0
        wanted = 0
        for key in itertools.product(*dimensions):
            desired += 1
            wanted += _chunk_id(key) not in vdata.loaded_chunks
        LOGGER.info(
            'napari resident bootstrap trace: level=%d '
            'desired_chunks=%d wanted_chunks=%d',
            self._resident_level,
            desired,
            wanted,
        )
        super()._start_resident_fill(min_coord, max_coord)

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
        # The renderer-specific napari plan is authoritative. In particular,
        # its 3-D plan is a chunk-aligned, memory-bounded cuboid and is
        # intentionally broader than strict view-frustum intersection on
        # large anisotropic volumes such as Zebrahub. Lodstone executes these
        # exact regions while the shared planner remains a diagnostic trace.
        self._submitted_plan = plan
        self._lodstone_target.generation = generation
        self._worker = _WorkerProxy(self._lodstone_stream)
        self._lodstone_stream.submit(view, plan)
        diagnostics = self._lodstone_stream.diagnostics
        LOGGER.info(
            'Lodstone submitted exact napari plan: generation=%d '
            'target_level=%d desired_tiles=%d wanted_tiles=%d '
            'native_chunks=%d',
            diagnostics.generation,
            plan.target_level,
            diagnostics.desired_tiles,
            diagnostics.wanted_tiles,
            diagnostics.unique_native_chunks,
        )

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
            source_level = self._lodstone_source.pyramid.levels[level]
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
                    source_level.chunk_index(axis, starts[axis])
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

    @property
    def execution_diagnostics(self) -> tuple[StreamDiagnostics, ...]:
        """Recent exact-plan native-read traces, oldest first."""
        return tuple(self._execution_diagnostics)

    def _on_lodstone_status(self, status) -> None:
        stream = self._lodstone_stream
        if status.state in {'complete', 'failed'} and stream is not None:
            diagnostics = stream.diagnostics
            if diagnostics.generation == status.generation:
                self._record_execution_diagnostics(diagnostics, status.state)
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

    def _record_execution_diagnostics(
        self, diagnostics: StreamDiagnostics, state: str
    ) -> None:
        if diagnostics.generation == 0:
            LOGGER.info(
                'Lodstone execution trace: no exact plan was submitted; '
                'napari resident bootstrap remained active'
            )
            return
        if (
            self._execution_diagnostics
            and self._execution_diagnostics[-1].generation
            == diagnostics.generation
        ):
            return
        self._execution_diagnostics.append(diagnostics)
        LOGGER.info(
            'Lodstone execution trace: state=%s generation=%d '
            'desired_tiles=%d wanted_tiles=%d native_chunks=%d '
            'cache_hits=%d joined_reads=%d source_reads=%d '
            'evictions=%d cache_chunks=%d cache_bytes=%d',
            state,
            diagnostics.generation,
            diagnostics.desired_tiles,
            diagnostics.wanted_tiles,
            diagnostics.unique_native_chunks,
            diagnostics.cache_hits,
            diagnostics.joined_reads,
            diagnostics.source_reads,
            diagnostics.evictions,
            diagnostics.cache_chunks,
            diagnostics.cache_bytes,
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
        node = self._get_volume_node()
        if node is not None and node.clipmap_enabled:
            node.disable_clipmap()
        stream = self._lodstone_stream
        self._lodstone_stream = None
        self._lodstone_enabled = False
        if stream is not None:
            stream.close()
            self._record_execution_diagnostics(stream.diagnostics, 'closed')
        super().close()


def add_lodstone_loading_image(
    img,
    viewer=None,
    fill_value=0,
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
    data = MultiScaleVirtualData(img, fill_value=fill_value)
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


def add_lodstone_level_diagnostics(
    arrays,
    viewer=None,
    name=None,
    opacity=1.0,
    **kwargs: Any,
):
    """Show actual progressive reads as solid labels by source level.

    The wrapped arrays still perform their normal reads. Returned values are
    replaced with a constant categorical label after materialization, so the
    layer exercises the same resident bootstrap, exact Lodstone plans, cache,
    and delivery paths as the image while making coverage unambiguous.

    Label 1 (magenta) is missing/unfilled content. Labels 2 onward identify
    L0, L1, and subsequent pyramid levels using stable distinct colors.
    """
    diagnostic_arrays = [
        _LevelDiagnosticArray(array, level)
        for level, array in enumerate(arrays)
    ]
    metadata = dict(kwargs.pop('metadata', {}))
    metadata['level_diagnostic_labels'] = {
        MISSING_LEVEL_LABEL: 'missing',
        **{
            level + LEVEL_LABEL_OFFSET: f'L{level}'
            for level in range(len(diagnostic_arrays))
        },
    }
    metadata['level_diagnostic_legend'] = {
        'missing': 'magenta',
        **{
            f'L{level}': LEVEL_DIAGNOSTIC_COLOR_NAMES[
                level % len(LEVEL_DIAGNOSTIC_COLOR_NAMES)
            ]
            for level in range(len(diagnostic_arrays))
        },
    }
    kwargs.setdefault(
        'colormap', _level_diagnostic_color_map(len(diagnostic_arrays))
    )
    kwargs.setdefault('debug_overlay', True)
    return add_lodstone_loading_labels(
        diagnostic_arrays,
        viewer=viewer,
        fill_value=MISSING_LEVEL_LABEL,
        name=name or 'Lodstone level provenance',
        opacity=opacity,
        metadata=metadata,
        **kwargs,
    )


def add_lodstone_loading_labels(
    labels,
    viewer=None,
    fill_value=0,
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
    """Add progressive Labels rendered by napari and loaded by Lodstone."""

    viewer, tile_max_bytes_3d = _resolve_viewer_and_tile_cap(
        viewer,
        tile_max_bytes_3d,
    )
    data = MultiScaleVirtualData(labels, fill_value=fill_value)
    _normalize_scale_for_float32(data, layer_kwargs, 'label')

    from napari.layers import Labels

    layer = Labels(
        data._data,
        multiscale=True,
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
