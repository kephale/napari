"""Lodstone planning and execution for napari's progressive renderer.

Napari captures camera events and retains ownership of bounded ``VirtualData``
intervals, backdrops, and GPU texture updates. Each prepared interval is
represented as a real Lodstone ``View`` and planned by Lodstone. The previous
napari plan is retained as a comparison trace while Lodstone owns tile order,
chunk reads, decoded caching, batching, cancellation, and stale-pass rejection.
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from threading import Lock
from typing import TYPE_CHECKING, Any

import numpy as np
from lodstone import (
    Layout,
    LevelDiagnosticArray,
    Plan,
    PlanComparison as _PlanComparison,
    Planner,
    PlanTrace,
    Region,
    Stream,
    StreamDiagnostics,
    TileKey,
    View,
    available_tile_keys,
    merge_plans,
    plan_from_slices,
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
    _pack_upload_block,
    _resolve_viewer_and_tile_cap,
    chunk_priority_2D,
    chunk_slices,
)
from napari.experimental._virtual_data import (
    MultiScaleVirtualData,
    chunk_sizes_for,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

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


_LevelDiagnosticArray = LevelDiagnosticArray


class PlanComparison(_PlanComparison):
    """Compatibility view naming the reference and candidate planners."""

    @property
    def napari(self) -> PlanTrace:
        return self.reference

    @property
    def lodstone(self) -> PlanTrace:
        return self.candidate


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


def _level_transforms(layer, data) -> list[np.ndarray]:
    """Map every pyramid level into napari world coordinates."""
    layer_to_world = np.asarray(
        layer._data_to_world.affine_matrix, dtype=float
    )
    data_to_world = np.eye(data.ndim + 1, dtype=float)
    spatial_ndim = layer_to_world.shape[0] - 1
    data_to_world[:spatial_ndim, :spatial_ndim] = layer_to_world[:-1, :-1]
    data_to_world[:spatial_ndim, -1] = layer_to_world[:-1, -1]
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
    depth_center: Sequence[float] | None = None,
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
        projection_center = center
        if depth_center is not None:
            projection_center = np.asarray(depth_center, dtype=float)
            if projection_center.shape != (3,) or not np.all(
                np.isfinite(projection_center)
            ):
                projection_center = center
        scales = (
            2.0 * zoom / viewport[0],
            2.0 * zoom / viewport[1],
            2.0 / max(float(depth_span), 1.0),
        )
        for row, (direction, scale) in enumerate(
            zip(rows, scales, strict=True)
        ):
            matrix[row, :3] = direction * scale
            row_center = center if row < 2 else projection_center
            matrix[row, 3] = -float(np.dot(direction, row_center)) * scale
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
        try:
            self.requested.emit(callback)
        except RuntimeError:
            # Qt may destroy the dispatcher before a late Stream.close()
            # status delivery during application teardown. There is no
            # event loop left to receive the callback in that state.
            LOGGER.debug('dropping callback after Qt dispatcher teardown')


class _WorkerProxy:
    def __init__(self, stream: Stream) -> None:
        self._stream = stream

    def quit(self) -> None:
        self._stream.cancel()


class _NapariTarget:
    def __init__(self, loader: LodstoneProgressiveLoader) -> None:
        self.loader = loader
        self.generation = 0
        self._prepare_lock = Lock()
        self.prepared_regions: dict[int, Region] = {}

    def layout(self, view, pyramid) -> Layout:
        return Layout(
            kind='bricked',
            memory_limit=self.loader._interval_max_bytes,
            squeeze_hidden=False,
            max_axis_extent=self.loader._gl_max_texture_size_2d,
        )

    def stage_prepare(self, view, plan) -> dict[int, Region]:
        """Compute pass intervals on Lodstone's runtime thread."""
        bounds: dict[int, tuple[list[int], list[int]]] = {}
        for tile in plan.desired:
            if tile.level not in bounds:
                bounds[tile.level] = [
                    list(tile.region.start),
                    list(tile.region.stop),
                ]
                continue
            start, stop = bounds[tile.level]
            for axis in range(tile.region.ndim):
                start[axis] = min(start[axis], tile.region.start[axis])
                stop[axis] = max(stop[axis], tile.region.stop[axis])
        return {
            level: Region(tuple(start), tuple(stop))
            for level, (start, stop) in bounds.items()
        }

    def prepare(self, view, plan, prepared=None) -> None:
        if prepared is None:
            prepared = self.stage_prepare(view, plan)
        with self._prepare_lock:
            self.prepared_regions = prepared

    def stage(self, updates):
        """Write and pack chunks on Lodstone's worker thread."""
        with self._prepare_lock:
            prepared_regions = self.prepared_regions.copy()
        grouped: dict[int, list[tuple[slice, ...]]] = defaultdict(list)
        for update in updates:
            vdata = self.loader._data[update.level]
            prepared = prepared_regions.get(update.level)
            if prepared is not None and not vdata.covers(
                prepared.start, prepared.stop
            ):
                vdata.set_interval(prepared.start, prepared.stop)
            key = update.region.slices()
            if vdata.set_chunk(key, update.data, update.level):
                grouped[update.level].append(key)
        active = getattr(self.loader, '_active', None)
        target = active[0] if active is not None else None
        resident = getattr(self.loader, '_resident_level', target)
        return tuple(
            (
                level,
                batch,
                _pack_upload_block(self.loader._data[level], batch)
                if level in {target, resident}
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
            resident = getattr(self.loader, '_resident_level', None)
            on_resident = getattr(self.loader, '_on_resident_chunks', None)
            if level == resident and on_resident is not None:
                on_resident(
                    self.loader._data[level],
                    (batch, block) if block is not None else batch,
                )
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
        self._bounded_planners = {
            2: Planner(lod_bias=1.0),
            3: Planner(lod_bias=lod_bias),
        }
        self._plan_comparisons: deque[PlanComparison] = deque(maxlen=32)
        self._bounded_plan_comparisons: deque[PlanComparison] = deque(
            maxlen=32
        )
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
        self._disconnect_lodstone_status = (
            self._lodstone_stream.on_status_changed(self._on_lodstone_status)
        )
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

    def _start_stage(self, generation: int, index: int) -> None:
        if not self._lodstone_enabled or self._lodstone_stream is None:
            return

        target = self._stages[-1][0]
        desired_stages = self._desired_stages(target)
        view = self._pass_view()
        available = self._available_keys(view)
        plan = plan_from_slices(
            self._lodstone_source.pyramid,
            view,
            desired_stages,
            target_level=target,
            available=available,
            fetch_levels=None,
        )
        plan_source = 'napari-fallback'
        shared = self._shared_planners[len(view.displayed_axes)].plan(
            self._lodstone_source.pyramid,
            view,
            self._lodstone_target.layout(view, self._lodstone_source.pyramid),
            available=available,
        )
        comparison = PlanComparison(
            view,
            PlanTrace.from_plan(plan),
            PlanTrace.from_plan(shared),
        )
        self._plan_comparisons.append(comparison)
        active = self._active
        if active is not None:
            _active_level, active_min, active_max = active
            bounded = self._bounded_planners[
                len(view.displayed_axes)
            ].plan_region(
                self._lodstone_source.pyramid,
                view,
                self._lodstone_target.layout(
                    view, self._lodstone_source.pyramid
                ),
                target_level=target,
                target_region=Region(active_min, active_max),
                available=available,
                fetch_intermediate=True,
            )
            bounded_comparison = PlanComparison(
                view,
                PlanTrace.from_plan(plan),
                PlanTrace.from_plan(bounded),
            )
            self._bounded_plan_comparisons.append(bounded_comparison)
            if bounded_comparison.geometry_matches:
                plan = bounded
                plan_source = 'lodstone-region'
        overview = self._bounded_planners[
            len(view.displayed_axes)
        ].plan_overview(
            self._lodstone_source.pyramid,
            view,
            level_index=self._resident_level,
            memory_limit=self._resident_max_bytes,
            available=available,
        )
        plan = merge_plans(plan, overview)
        self._chunks_total = len(plan.wanted)
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
        # Use Lodstone's renderer-constrained region plan only after checking
        # that it describes the same target geometry. Keep the reconstructed
        # napari plan as a compatibility fallback while the strict frustum
        # planner remains useful as an independent diagnostic.
        self._submitted_plan = plan
        self._lodstone_target.generation = generation
        self._worker = _WorkerProxy(self._lodstone_stream)
        self._lodstone_stream.submit(view, plan)
        diagnostics = self._lodstone_stream.diagnostics
        LOGGER.info(
            'Lodstone submitted progressive plan: source=%s generation=%d '
            'target_level=%d desired_tiles=%d wanted_tiles=%d '
            'native_chunks=%d',
            plan_source,
            diagnostics.generation,
            plan.target_level,
            diagnostics.desired_tiles,
            diagnostics.wanted_tiles,
            diagnostics.unique_native_chunks,
        )

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
        return available_tile_keys(
            self._lodstone_source.pyramid,
            view,
            {
                level: vdata.loaded_chunks
                for level, vdata in enumerate(self._data)
            },
        )

    def _ensure_resident(self) -> None:
        """The LodStone plan carries persistent overview residency."""

    def _resident_target_interval(
        self,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        view = self._pass_view()
        plan = self._bounded_planners[len(view.displayed_axes)].plan_overview(
            self._lodstone_source.pyramid,
            view,
            level_index=self._resident_level,
            memory_limit=self._resident_max_bytes,
        )
        if not plan.desired:
            return None
        start = np.min([tile.region.start for tile in plan.desired], axis=0)
        stop = np.max([tile.region.stop for tile in plan.desired], axis=0)
        return np.asarray(start), np.asarray(stop)

    def _pass_view(self) -> View:
        transforms = self._lodstone_source.pyramid.levels[0].voxel_to_world
        extent = np.asarray(self._data.shape, dtype=float)
        world_extent = np.abs(transforms[:-1, :-1]) @ extent
        displayed = self._layer._slice_input.displayed
        depth_span = float(np.linalg.norm(np.take(world_extent, displayed)))
        data_center = np.concatenate(
            [np.asarray(self._data.shape, dtype=float) / 2.0, [1.0]]
        )
        world_center = (transforms @ data_center)[:-1]
        depth_center = tuple(float(world_center[axis]) for axis in displayed)
        return _camera_view(
            self._viewer,
            self._layer,
            self._data.shape,
            depth_span,
            depth_center,
        )

    @property
    def plan_comparisons(self) -> tuple[PlanComparison, ...]:
        """Recent PR/shared-planner traces, oldest first."""
        return tuple(self._plan_comparisons)

    @property
    def bounded_plan_comparisons(self) -> tuple[PlanComparison, ...]:
        """Recent established/bounded-region traces, oldest first."""
        return tuple(self._bounded_plan_comparisons)

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
        if self._closed:
            return
        node = self._get_volume_node()
        if node is not None and node.clipmap_enabled:
            node.disable_clipmap()
        stream = self._lodstone_stream
        self._lodstone_stream = None
        self._lodstone_enabled = False
        disconnect_status = self._disconnect_lodstone_status
        self._disconnect_lodstone_status = lambda: None
        disconnect_status()
        # Disconnect viewer callbacks and mark the loader closed before
        # shutting down the stream. Stream cancellation can deliver queued
        # Qt work, which must not be allowed to start another fetch pass.
        super().close()
        if stream is not None:
            stream.close()
            self._record_execution_diagnostics(stream.diagnostics, 'closed')


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
