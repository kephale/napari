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
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import Lock
from typing import TYPE_CHECKING, Any

import numpy as np
from lodstone import (
    Layout,
    LevelDiagnosticArray,
    PerformanceRecorder,
    Plan,
    PlanComparison as _PlanComparison,
    Planner,
    PlanTrace,
    Region,
    Runtime,
    Stream,
    StreamDiagnostics,
    TargetDiagnostics,
    TileKey,
    View,
    available_tile_keys,
    merge_plans,
    plan_from_slices,
)
from lodstone.sources import ArrayPyramidSource
from qtpy.QtCore import QObject, Qt, Signal, Slot

from napari._vispy.visuals.volume import CLIPMAP_MAX_RINGS
from napari.experimental._progressive_loading import (
    DEFAULT_INTERVAL_MAX_BYTES,
    DEFAULT_TILE_MAX_BYTES_3D,
    ProgressiveLoader,
    _attach_progressive_loader,
    _drawn_camera_axes,
    _estimate_contrast_limits,
    _normalize_scale_for_float32,
    _orientation_signs,
    _pack_upload_block,
    _resolve_viewer_and_tile_cap,
    _texture_dtype_3d,
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
_RUNTIME_LOCK = Lock()
_RUNTIME_ATTRIBUTE = '_lodstone_runtime'
_RUNTIME_REFS_ATTRIBUTE = '_lodstone_runtime_refs'
# Decoded bytes Lodstone may gather into one Qt delivery and upload block.
LODSTONE_GPU_BATCH_BYTES = 32 << 20


# clipmap rings: levels between a ring and the next, and the least
# detail-to-overview level gap that uses them by default
_RING_LEVEL_STEP = 2
_AUTO_RING_MIN_GAP = 5


def _append_plans(primary: Plan, *suffixes: Plan) -> Plan:
    """``primary`` followed by tiles of ``suffixes`` it does not hold."""

    def unique(tiles):
        seen: set[TileKey] = set()
        result = []
        for tile in tiles:
            if tile.key not in seen:
                seen.add(tile.key)
                result.append(tile)
        return tuple(result)

    ordered = (primary, *suffixes)
    return Plan(
        unique(tile for plan in ordered for tile in plan.wanted),
        frozenset(key for plan in ordered for key in plan.retain),
        primary.target_level,
        unique(tile for plan in ordered for tile in plan.desired),
    )


def _compact_key_groups(keys, max_waste: float = 2.0) -> list[list]:
    """Split chunk keys into groups whose union boxes stay compact.

    A batch is uploaded as the union box of its chunks. Parallel reads
    deliver chunks from different parts of the tile together, and their
    union can be a hundred times their own volume: one 16 MiB upload, and
    again through the double buffer, for 256 KiB of new data. Grow each
    group only while its union box is at most ``max_waste`` times the
    volume of its chunks.
    """
    groups: list[tuple[list, np.ndarray, np.ndarray, int]] = []
    for key in keys:
        low = np.array([k.start for k in key], dtype=np.int64)
        high = np.array([k.stop for k in key], dtype=np.int64)
        volume = int(np.prod(high - low))
        for index, (members, g_low, g_high, g_volume) in enumerate(groups):
            u_low, u_high = np.minimum(g_low, low), np.maximum(g_high, high)
            if np.prod(u_high - u_low) <= max_waste * (g_volume + volume):
                groups[index] = (
                    [*members, key],
                    u_low,
                    u_high,
                    g_volume + volume,
                )
                break
        else:
            groups.append(([key], low, high, volume))
    return [members for members, *_ in groups]


try:
    # a huge regular axis is a compact sequence whose max() would iterate
    # every chunk; Lodstone reads it arithmetically
    from lodstone.grid import max_chunk_size as _max_chunk_size
except ImportError:  # pragma: no cover - Lodstone before compact grids
    _max_chunk_size = max


def _gpu_batch_size(chunks, dtype, workers: int) -> int:
    """Tiles per Lodstone read window and Qt delivery.

    Lodstone gathers each window's reads concurrently and delivers it as
    one batch, so the batch size bounds read parallelism as well as
    delivery size. A decoded ZebraHub chunk is roughly 32 MiB, and core's
    default of eight tiles would make 256+ MiB deliveries, so large chunks
    go one at a time. Small chunks (the 32^3 Mandelbulb's are 32 KiB) would
    then read strictly one after another, leaving every worker but one
    idle: allow enough of them to keep the workers busy.
    """
    itemsize = np.dtype(dtype).itemsize
    largest = max(
        int(np.prod([_max_chunk_size(axis) for axis in level], dtype=np.int64))
        for level in chunks
    )
    fit = LODSTONE_GPU_BATCH_BYTES // max(largest * itemsize, 1)
    return int(max(1, min(2 * workers, fit)))


def _acquire_viewer_runtime(viewer) -> Runtime:
    """Share bounded Lodstone scheduling resources across viewer layers."""
    with _RUNTIME_LOCK:
        runtime = getattr(viewer, _RUNTIME_ATTRIBUTE, None)
        if runtime is None or runtime.closed:
            runtime = Runtime(compute_workers=2)
            setattr(viewer, _RUNTIME_ATTRIBUTE, runtime)
            setattr(viewer, _RUNTIME_REFS_ATTRIBUTE, 0)
        refs = int(getattr(viewer, _RUNTIME_REFS_ATTRIBUTE, 0)) + 1
        setattr(viewer, _RUNTIME_REFS_ATTRIBUTE, refs)
        return runtime


def _release_viewer_runtime(viewer, runtime: Runtime) -> None:
    with _RUNTIME_LOCK:
        if getattr(viewer, _RUNTIME_ATTRIBUTE, None) is not runtime:
            return
        refs = max(0, int(getattr(viewer, _RUNTIME_REFS_ATTRIBUTE, 1)) - 1)
        if refs:
            setattr(viewer, _RUNTIME_REFS_ATTRIBUTE, refs)
            return
        runtime.close()
        delattr(viewer, _RUNTIME_ATTRIBUTE)
        delattr(viewer, _RUNTIME_REFS_ATTRIBUTE)


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
    # napari reports canvas size as (height, width); Lodstone uses (width, height).
    viewport = tuple(max(int(value), 1) for value in viewer.canvas.size[::-1])
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
        _, row_sign, column_sign = _orientation_signs(camera)
        x_scale, y_scale = 2.0 * zoom / np.asarray(viewport, dtype=float)
        x_scale *= column_sign
        # Screen Y points against the first displayed axis by default, because
        # napari draws rows downward.
        y_scale *= row_sign
        matrix[0, 1] = x_scale
        matrix[1, 0] = -y_scale
        matrix[0, 3] = -x_scale * center[1]
        matrix[1, 3] = y_scale * center[0]
        eye = None
    else:
        view_direction, up_direction, right_direction = _drawn_camera_axes(
            camera
        )
        view_direction = view_direction / max(
            np.linalg.norm(view_direction), 1e-12
        )
        up_direction = up_direction / max(np.linalg.norm(up_direction), 1e-12)
        right_direction = right_direction / max(
            np.linalg.norm(right_direction), 1e-12
        )
        rows = (right_direction, up_direction, view_direction)
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


@dataclass(frozen=True)
class _PlanInputs:
    """Qt-thread state a pass is planned from, captured at pass start."""

    target: int
    view: View
    desired_stages: list
    available: frozenset
    layout: Layout
    active: tuple | None
    ring_stages: list = ()  # type: ignore[assignment]


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
        self._performance_lock = Lock()
        self._submitted_bytes = 0
        self._presentations = 0
        self.prepared_regions: dict[int, Region] = {}

    def performance_metrics(self) -> TargetDiagnostics:
        """Report texture submissions observable at napari's upload boundary."""
        with self._performance_lock:
            return TargetDiagnostics(
                submitted_bytes=self._submitted_bytes,
                presentations=self._presentations,
            )

    def _record_submission(self, level: int, batch, block) -> None:
        if block is not None:
            submitted = block[2].nbytes
        else:
            data = self.loader._data[level]
            dtype = getattr(data, 'dtype', None)
            if dtype is None:
                dtype = data.values.dtype
            itemsize = np.dtype(dtype).itemsize
            submitted = sum(
                int(np.prod([part.stop - part.start for part in key]))
                * itemsize
                for key in batch
            )
        with self._performance_lock:
            self._submitted_bytes += submitted
            self._presentations += 1

    def layout(self, view, pyramid) -> Layout:
        return Layout(
            kind='bricked',
            memory_limit=self.loader._interval_max_bytes,
            squeeze_hidden=False,
            max_axis_extent=self.loader._gl_max_texture_size_2d,
            # Match volume refinement to the crosshair instead of exhausting
            # each near depth slab before reaching central interior content.
            focus_depth_weight=(
                0.5 if len(view.displayed_axes) == 3 else None
            ),
            focus_depth_target=(
                0.5 if len(view.displayed_axes) == 3 else None
            ),
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
        uploaded = {target, resident} | {
            ring[0] for ring in getattr(self.loader, '_rings', ())
        }
        return tuple(
            (
                level,
                group,
                _pack_upload_block(self.loader._data[level], group)
                if level in uploaded
                else None,
            )
            for level, batch in grouped.items()
            for group in (
                _compact_key_groups(batch) if level in uploaded else (batch,)
            )
        )

    def apply(self, staged) -> None:
        # Keep direct calls useful for adapters and tests; Stream calls
        # ``stage`` before dispatching here, so production CPU work remains
        # off the Qt thread.
        if staged and not isinstance(staged[0], tuple):
            staged = self.stage(staged)
        on_ring = getattr(self.loader, '_on_ring_chunks', None)
        for level, batch, block in staged:
            if on_ring is not None:
                batch = on_ring(level, batch, block)
                if not batch:
                    continue
            resident = getattr(self.loader, '_resident_level', None)
            on_resident = getattr(self.loader, '_on_resident_chunks', None)
            if level == resident and on_resident is not None:
                self._record_submission(level, batch, block)
                on_resident(
                    self.loader._data[level],
                    (batch, block) if block is not None else batch,
                )
            self._record_submission(level, batch, block)
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
    """ProgressiveLoader whose fetch passes execute through Lodstone.

    ``trace_plans`` also runs Lodstone's strict frustum planner on every
    pass and records it against the submitted plan in
    :attr:`plan_comparisons`. That plan is diagnostic only and is planned
    on the Qt thread, which stalls the UI for seconds on large pyramids,
    so it is off unless requested (or debug logging is enabled).
    """

    trace_plans: bool = False

    def __init__(
        self,
        viewer,
        layer,
        data,
        *,
        trace_plans: bool | None = None,
        clipmap_rings: int | None = None,
        **kwargs,
    ) -> None:
        if trace_plans is not None:
            self.trace_plans = trace_plans
        self._clipmap_ring_limit = clipmap_rings
        # (level, low, high) per clipmap ring, finest first, and each ring
        # level's share of the coarse-first ladder (see _configure_rings)
        self._rings: tuple[tuple[int, tuple, tuple], ...] = ()
        self._ring_footprints: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._lodstone_enabled = False
        self._lodstone_stream: Stream | None = None
        self._lodstone_runtime: Runtime | None = None
        lod_bias = float(kwargs.get('max_pixel_size_3d', 2.0))
        kwargs['_start_immediately'] = False
        super().__init__(viewer, layer, data, **kwargs)

        level_chunks = [chunk_sizes_for(array) for array in data.arrays]
        source = ArrayPyramidSource(
            data.arrays,
            chunks=level_chunks,
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
        self._plan_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix='napari-lodstone-plan'
        )
        self._submitted_plan: Plan | None = None
        self._lodstone_target = _NapariTarget(self)
        runtime = _acquire_viewer_runtime(viewer)
        self._lodstone_runtime = runtime
        try:
            self._lodstone_stream = Stream(
                source,
                self._lodstone_target,
                dispatch=self._lodstone_dispatcher.dispatch,
                workers=self._fetch_workers,
                batch_size=_gpu_batch_size(
                    level_chunks, data.dtype, self._fetch_workers
                ),
                # The movable interval already bounds napari's CPU working
                # set.  Do not retain a second, multi-gigabyte decoded cache
                # behind it merely because Stream's viewer-neutral default is
                # sized for reuse-heavy non-GPU consumers.
                cpu_cache=max(
                    self._interval_max_bytes,
                    self._resident_max_bytes,
                ),
                bytes_per_second=self._max_bytes_per_second,
                runtime=runtime,
            )
        except Exception:
            self._lodstone_runtime = None
            _release_viewer_runtime(viewer, runtime)
            raise
        self._disconnect_lodstone_status = (
            self._lodstone_stream.on_status_changed(self._on_lodstone_status)
        )
        self._performance_recorder = PerformanceRecorder(
            self._lodstone_stream,
            host='napari',
            backend='vispy',
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
        """Plan the pass off the Qt thread, then submit it from the Qt thread.

        Planning walks every tile of every ladder level in Python (hundreds
        of milliseconds for a 3D view), which stalled the viewer each time
        the camera settled. Inputs that read Qt, camera and layer state are
        captured here; the plans are pure functions of them and are built on
        a planning thread. Submission returns to this thread so the target's
        generation still changes in step with the stream's.
        """
        if not self._lodstone_enabled or self._lodstone_stream is None:
            return
        target = self._stages[-1][0]
        view = self._pass_view()
        inputs = _PlanInputs(
            target=target,
            view=view,
            desired_stages=self._desired_stages(target),
            available=self._available_keys(view),
            layout=self._lodstone_target.layout(
                view, self._lodstone_source.pyramid
            ),
            active=self._active,
            ring_stages=self._ring_stages(),
        )
        # the pass is in flight from now on, even while it is being planned
        self._worker = _WorkerProxy(self._lodstone_stream)
        future = self._plan_executor.submit(self._build_plan, inputs)
        future.add_done_callback(
            lambda done: self._lodstone_dispatcher.dispatch(
                lambda: self._submit_planned(generation, inputs, done)
            )
        )

    def _build_plan(self, inputs: _PlanInputs) -> tuple[Plan, str]:
        """Build the pass plan from captured inputs (planning thread)."""
        pyramid = self._lodstone_source.pyramid
        view, target = inputs.view, inputs.target
        planners = self._bounded_planners[len(view.displayed_axes)]
        plan = plan_from_slices(
            pyramid,
            view,
            inputs.desired_stages,
            target_level=target,
            available=inputs.available,
            fetch_levels=None,
        )
        plan_source = 'napari-fallback'
        if self.trace_plans or LOGGER.isEnabledFor(logging.DEBUG):
            self._trace_shared_plan(view, plan, inputs.available)
        if inputs.active is not None:
            _active_level, active_min, active_max = inputs.active
            bounded = planners.plan_region(
                pyramid,
                view,
                inputs.layout,
                target_level=target,
                target_region=Region(active_min, active_max),
                available=inputs.available,
                fetch_intermediate=True,
            )
            bounded_comparison = PlanComparison(
                view,
                PlanTrace.from_plan(plan),
                PlanTrace.from_plan(bounded),
            )
            self._bounded_plan_comparisons.append(bounded_comparison)
            # Use Lodstone's renderer-constrained region plan only after
            # checking that it describes the same target geometry. Keep the
            # reconstructed napari plan as a compatibility fallback while
            # the strict frustum planner remains useful as a diagnostic.
            if bounded_comparison.geometry_matches:
                plan = bounded
                plan_source = 'lodstone-region'
        overview = planners.plan_overview(
            pyramid,
            view,
            level_index=self._resident_level,
            memory_limit=self._resident_max_bytes,
            available=inputs.available,
        )
        plan = merge_plans(plan, overview)
        if inputs.ring_stages:
            # rings sharpen the surroundings: after the view's own tiles
            rings = plan_from_slices(
                pyramid,
                view,
                inputs.ring_stages,
                target_level=target,
                available=inputs.available,
                fetch_levels=None,
            )
            plan = _append_plans(plan, rings)
        return plan, plan_source

    def _submit_planned(self, generation: int, inputs: _PlanInputs, done):
        """Submit a finished plan if its pass is still current (Qt thread)."""
        stream = self._lodstone_stream
        if (
            self._closed
            or not self._lodstone_enabled
            or stream is None
            or generation != self._generation
        ):
            return
        try:
            plan, plan_source = done.result()
        except Exception:
            LOGGER.exception('Lodstone planning failed; planning inline')
            plan, plan_source = self._build_plan(inputs)
        # ring-only chunks never reach the progress count (_on_ring_chunks)
        self._chunks_total = len(plan.wanted) - sum(
            1
            for tile in plan.wanted
            if self._ring_only(tile.level, tile.region.slices())
        )
        self._submitted_plan = plan
        self._lodstone_target.generation = generation
        stream.submit(inputs.view, plan)
        diagnostics = stream.diagnostics
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

    def _trace_shared_plan(self, view, plan, available) -> None:
        """Record Lodstone's strict frustum plan beside the submitted one."""
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

    # -- clipmap rings --

    def _ring_layout(
        self, target: int, min_coord, max_coord
    ) -> tuple[tuple[int, tuple, tuple], ...]:
        """(level, low, high) of each ring around a target region.

        Rings are cubes of the 3D tile budget every _RING_LEVEL_STEP levels
        coarser than the target, centered on it, finest first, stopping
        before the resident overview or once a ring holds its whole level.
        With ``clipmap_rings=None`` they are used only when the overview
        is _AUTO_RING_MIN_GAP or more levels coarser than the target.
        """
        node = self._get_volume_node()
        if (
            self._viewer.dims.ndisplay != 3
            or self._data.ndim != 3
            or node is None
            or not getattr(node, 'clipmap_enabled', False)
            or not hasattr(node, 'set_clipmap_ring')
        ):
            return ()
        gap = self._resident_level - target
        limit = self._clipmap_ring_limit
        if limit is None:
            limit = CLIPMAP_MAX_RINGS if gap >= _AUTO_RING_MIN_GAP else 0
        limit = max(0, min(int(limit), CLIPMAP_MAX_RINGS))
        tile_bytes = getattr(self._layer, '_tile_max_bytes_3d', None)
        if not limit or not tile_bytes:
            return ()
        itemsize = _texture_dtype_3d(self._data.dtype).itemsize
        # the detail tile's cube, rounded to whole chunks below
        edge = (tile_bytes / itemsize) ** (1 / 3)
        factors = self._data._scale_factors
        center = (
            np.asarray(min_coord, dtype=float)
            + np.asarray(max_coord, dtype=float)
        ) / 2
        rings = []
        for index in range(1, limit + 1):
            level = target + _RING_LEVEL_STEP * index
            if level >= self._resident_level:
                break
            vdata = self._data[level]
            shape = np.asarray(vdata.shape, dtype=np.int64)
            chunks = np.asarray(vdata.chunk_shape, dtype=np.int64)
            ratio = np.asarray(factors[target], float) / np.asarray(
                factors[level], float
            )
            extent = np.minimum(
                np.maximum(np.round(edge / chunks), 1).astype(np.int64)
                * chunks,
                shape,
            )
            low = np.floor(center * ratio - extent / 2).astype(np.int64)
            low = np.clip(low // chunks * chunks, 0, shape - extent)
            rings.append(
                (
                    level,
                    tuple(int(v) for v in low),
                    tuple(int(v) for v in low + extent),
                )
            )
            if np.all(extent >= shape):
                break
        return tuple(rings)

    def _configure_rings(self, level: int, min_coord, max_coord) -> bool:
        """Place the clipmap rings around this pass's target region.

        Each ring level's resident interval grows to hold its ring (new
        voxels start as the upsampled overview), and the ring's texture
        is (re)built from it; chunks then stream in after the target's
        (see _build_plan) and patch it (_on_ring_chunks).
        """
        rings = self._ring_layout(level, min_coord, max_coord)
        node = self._get_volume_node()
        factors = self._data._scale_factors
        footprints = {}
        pending = False
        resident_loaded = bool(self._data[self._resident_level].loaded_chunks)
        for index, (ring_level, low, high) in enumerate(rings):
            vdata = self._data[ring_level]
            low_a, high_a = np.asarray(low), np.asarray(high)
            ratio = np.asarray(factors[level], float) / np.asarray(
                factors[ring_level], float
            )
            footprints[ring_level] = (
                np.floor(np.asarray(min_coord) * ratio).astype(np.int64),
                np.ceil(np.asarray(max_coord) * ratio).astype(np.int64),
            )
            if not vdata.covers(low_a, high_a):
                new_low, new_high = low_a, high_a
                existing = vdata.interval
                if existing is not None:
                    union_low, union_high = self._clamp_interval(
                        vdata,
                        np.minimum(low_a, existing[0]),
                        np.maximum(high_a, existing[1]),
                    )
                    if np.all(union_low <= low_a) and np.all(
                        high_a <= union_high
                    ):
                        new_low, new_high = union_low, union_high
                self._data.set_interval(
                    ring_level,
                    new_low,
                    new_high,
                    backdrop_level=self._resident_level
                    if resident_loaded
                    else None,
                )
            previous = self._rings[index] if index < len(self._rings) else None
            if previous != (ring_level, low, high) or index not in (
                node.clipmap_rings
            ):
                with vdata.lock:
                    start = np.asarray(vdata.interval[0])
                    crop = tuple(
                        slice(int(a - s), int(b - s))
                        for a, b, s in zip(low_a, high_a, start, strict=True)
                    )
                    data = np.array(vdata.hyperslice[crop])
                node.set_clipmap_ring(
                    index,
                    data,
                    low,
                    high,
                    scale=self._layer.downsample_factors[ring_level],
                )
            pending = pending or bool(
                self._stage_queue(ring_level, region=(low_a, high_a))
            )
        if node is not None and hasattr(node, 'clear_clipmap_rings'):
            node.clear_clipmap_rings(len(rings))
        self._rings = rings
        self._ring_footprints = footprints
        return pending

    def _ring_stages(self) -> list[tuple[int, list]]:
        """Every chunk of every ring, nearest the view first."""
        stages = []
        for level, low, high in self._rings:
            region = (np.asarray(low), np.asarray(high))
            keys = chunk_slices(self._data[level], interval=region)
            stages.append((level, self._prioritize_3d(level, keys, region)))
        return stages

    def _ring_only(self, level: int, key) -> bool:
        """A ring chunk outside the ladder's share of its level."""
        footprint = self._ring_footprints.get(level)
        if footprint is None:
            return False
        low, high = footprint
        return any(
            part.stop <= lo or part.start >= hi
            for part, lo, hi in zip(key, low, high, strict=False)
        )

    def _on_ring_chunks(self, level: int, batch, block):
        """Patch ring textures with fetched chunks (Qt thread).

        Returns the chunks the rest of the pipeline handles: those in the
        ladder's share of the level (they also refine the target's
        backdrop); ring-only ones end here.
        """
        index = next(
            (i for i, ring in enumerate(self._rings) if ring[0] == level),
            None,
        )
        if index is None:
            return batch
        node = self._get_volume_node()
        if (
            block is not None
            and node is not None
            and index in getattr(node, 'clipmap_rings', ())
        ):
            _level, ring_low, ring_high = self._rings[index]
            block_low, block_high, data = block
            low = np.maximum(np.asarray(block_low), ring_low)
            high = np.minimum(np.asarray(block_high), ring_high)
            if np.all(high > low):
                crop = tuple(
                    slice(int(a - b), int(c - b))
                    for a, b, c in zip(low, block_low, high, strict=True)
                )
                node.set_clipmap_ring_data(
                    index, data[crop], offset=low - np.asarray(ring_low)
                )
        return [key for key in batch if not self._ring_only(level, key)]

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

    @property
    def performance_records(self) -> tuple[dict[str, Any], ...]:
        """Comparable Lodstone stream and renderer-boundary measurements."""
        return self._performance_recorder.records()

    def _on_lodstone_status(self, status) -> None:
        stream = self._lodstone_stream
        if status.state in {'complete', 'failed'} and stream is not None:
            diagnostics = stream.diagnostics
            if diagnostics.generation == status.generation:
                self._record_execution_diagnostics(diagnostics, status.state)
                node = self._get_volume_node()
                if node is not None and node.clipmap_enabled:
                    LOGGER.info(
                        'Lodstone clipmap trace: level=%d active=%s '
                        'corners=%s texture_shape=%s detail_bounds=%s',
                        self._layer.data_level,
                        self._active,
                        self._layer.corner_pixels.tolist(),
                        tuple(node._texture.shape[:3]),
                        node._clipmap_detail_bounds,
                    )
                    LOGGER.info(
                        'Lodstone tile budget trace: dtype=%s '
                        'tile_bytes=%s interval_bytes=%s extent_cap=%s '
                        'camera_center=%s view_direction=%s',
                        self._layer.dtype,
                        self._layer._tile_max_bytes_3d,
                        self._layer._interval_max_bytes_3d,
                        self._layer._max_tile_extent_3d,
                        self._viewer.scene.camera.center,
                        self._viewer.scene.camera.view_direction,
                    )
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
        # a plan still being built is dropped on delivery (loader closed)
        self._plan_executor.shutdown(wait=False, cancel_futures=True)
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
        self._performance_recorder.close()
        runtime = self._lodstone_runtime
        self._lodstone_runtime = None
        if runtime is not None:
            _release_viewer_runtime(self._viewer, runtime)


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
    clipmap_rings=None,
    **layer_kwargs: Any,
):
    """Add a progressive image rendered by napari and loaded by Lodstone.

    ``clipmap_rings`` sets how many 3D clipmap rings (boxes of
    intermediate levels around the detail tile, up to 3) to draw: ``None``
    uses them only for deep pyramids, where the overview is five or more
    levels coarser than the detail, and ``0`` never does.
    """
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
        loader_kwargs={'clipmap_rings': clipmap_rings},
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
        # Preserve the source dtype so this diagnostic exercises the same
        # texture allocation and tiling decisions as the real image.  The
        # generic wrapper defaults to uint8, which can make a 295 MiB level
        # appear to fit while the corresponding uint16 image becomes a
        # 1.18 GiB float32 texture in vispy.
        _LevelDiagnosticArray(array, level, dtype=array.dtype)
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
