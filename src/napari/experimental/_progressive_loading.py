"""Progressive (chunk-wise) loading for multiscale images.

This module implements progressive loading on top of napari's standard
multiscale ``Image`` layer. A single layer is added to the viewer; its
per-level data objects are :class:`~napari.experimental._virtual_data.VirtualData`
instances that present the full array shape while only keeping the visible,
chunk-aligned region in memory.

A :class:`ProgressiveLoader` watches the camera and dims, and on every view
change it:

1. reads the data level napari selected (respecting
   ``layer.locked_data_level``) and the visible ``corner_pixels``,
2. moves the level's resident interval to cover the view, initializing
   newly exposed regions from a coarser resident level (so the canvas is
   never empty),
3. fetches the missing chunks on a background thread in priority order
   (view-center first in 2D; camera depth/center-line in 3D), writing each
   chunk into the virtual data and refreshing the layer through napari's
   normal slicing pipeline.

The lowest-resolution level is kept fully resident (up to a size limit),
which provides instant low-resolution context everywhere, powers layer
thumbnails, and serves as the backdrop source for finer levels.

Use :func:`add_progressive_loading_image` to add a progressively loading
image to a viewer.

This is experimental: expect breaking changes and rough edges, and please
report issues to https://github.com/napari/napari/issues.
"""

from __future__ import annotations

import contextlib
import itertools
import logging
import time
from typing import TYPE_CHECKING

import numpy as np
from psygnal import debounced
from superqt import ensure_main_thread

from napari.experimental._virtual_data import (
    MultiScaleVirtualData,
    VirtualData,
    chunk_boundaries,
)
from napari.qt.threading import thread_worker
from napari.utils import progress

if TYPE_CHECKING:
    import napari

LOGGER = logging.getLogger(__name__)

#: Maximum size of the always-resident coarsest level, in bytes.
DEFAULT_RESIDENT_MAX_BYTES = 256 * 1024**2
#: Maximum size of a single level's resident interval, in bytes.
DEFAULT_INTERVAL_MAX_BYTES = 512 * 1024**2
#: Show a progress bar (activity dock) for fetch passes with at least this
#: many chunks; short interactive passes stay silent.
PROGRESS_MIN_CHUNKS = 16


# ---------- chunk geometry ----------


def chunk_slices(data, interval: tuple | None = None) -> list[list[slice]]:
    """Per-dimension lists of chunk slices, optionally clipped to a region.

    Parameters
    ----------
    data : VirtualData or array-like
        Object whose chunk grid should be enumerated.
    interval : tuple of (min_coord, max_coord), optional
        Half-open bounds per dimension. Only chunks intersecting the
        interval are returned.

    Returns
    -------
    list of list of slice
        For each dimension, the slices of every chunk along it. The full
        set of chunk keys is the cartesian product across dimensions.
    """
    if isinstance(data, VirtualData):
        boundaries = data._boundaries
    else:
        boundaries = chunk_boundaries(data)

    result: list[list[slice]] = []
    for dim, bounds in enumerate(boundaries):
        starts, stops = bounds[:-1], bounds[1:]
        if interval is not None:
            min_c = int(interval[0][dim])
            max_c = int(interval[1][dim])
            first = int(np.searchsorted(stops, min_c, side='right'))
            last = int(np.searchsorted(starts, max_c, side='left'))
            starts, stops = starts[first:last], stops[first:last]
        result.append(
            [
                slice(int(start), int(stop))
                for start, stop in zip(starts, stops, strict=True)
            ]
        )
    return result


def get_chunk_center(chunk_key: tuple[slice, ...]) -> np.ndarray:
    """Return the center coordinate of a tuple-of-slices chunk key."""
    return np.array([(sl.start + sl.stop) * 0.5 for sl in chunk_key])


def visual_depth(points, camera) -> np.ndarray:
    """Compute visual depth from camera position to a(n array of) point(s).

    Parameters
    ----------
    points : (N, D) array of float
        An array of N points. This can be one point or many thanks to NumPy
        broadcasting.
    camera : napari.components.Camera
        A camera model specifying a view direction and a center or focus
        point.

    Returns
    -------
    projected_length : (N,) array of float
        Position of the points along the view vector of the camera. These
        can be negative (in front of the center) or positive (behind the
        center).
    """
    view_direction = camera.view_direction
    points_relative_to_camera = points - camera.center
    projected_length = points_relative_to_camera @ view_direction
    return projected_length


def distance_from_camera_center_line(points, camera) -> np.ndarray:
    """Compute distance from a point or array of points to camera center line.

    This is the line aligned to the camera view direction and passing
    through the camera's center point.

    Parameters
    ----------
    points : (N, D) array of float
        An array of N points. This can be one point or many thanks to NumPy
        broadcasting.
    camera : napari.components.Camera
        A camera model specifying a view direction and a center or focus
        point.

    Returns
    -------
    distances : (N,) array of float
        Distances from points to the center line of the camera.
    """
    view_direction = camera.view_direction
    projected_length = visual_depth(points, camera)
    projected = view_direction * np.reshape(projected_length, (-1, 1))
    points_relative_to_camera = points - camera.center
    distances = np.linalg.norm(projected - points_relative_to_camera, axis=-1)
    return distances


def _chunk_keys_product(
    chunk_keys: list[list[slice]],
) -> list[tuple[slice, ...]]:
    return list(itertools.product(*chunk_keys))


def chunk_priority_2D(
    chunk_keys: list[list[slice]], min_coord, max_coord
) -> list[tuple[slice, ...]]:
    """Order chunk keys by distance from the center of the view.

    Parameters
    ----------
    chunk_keys : list of list of slice
        Per-dimension chunk slices (see :func:`chunk_slices`).
    min_coord, max_coord : sequence of int
        The visible interval in this level's coordinates.

    Returns
    -------
    list of tuple of slice
        Chunk keys sorted with the most central chunks first.
    """
    view_center = (np.asarray(min_coord) + np.asarray(max_coord)) / 2
    keys = _chunk_keys_product(chunk_keys)
    centers = np.array([get_chunk_center(key) for key in keys])
    if centers.size == 0:
        return []
    distances = np.linalg.norm(centers - view_center, axis=1)
    return [keys[i] for i in np.argsort(distances, kind='stable')]


def chunk_priority_3D(
    chunk_keys: list[list[slice]],
    min_coord,
    max_coord,
    camera_center,
    view_direction,
    zoom: float = 1.0,
    center_weight: float = 5.0,
) -> list[tuple[slice, ...]]:
    """Order chunk keys for 3D rendering.

    Priority combines visual depth along the view direction, distance from
    the camera's center line, and distance from the view center, so chunks
    in front of and near the middle of the camera load first.

    Parameters
    ----------
    chunk_keys : list of list of slice
        Per-dimension chunk slices (see :func:`chunk_slices`).
    min_coord, max_coord : sequence of int
        The visible interval in this level's coordinates.
    camera_center : sequence of float
        Camera center in this level's coordinates (displayed dimensions,
        i.e. the last 3 dimensions of the chunk keys).
    view_direction : sequence of float
        Camera view direction (3-vector over the displayed dimensions).
    zoom : float
        Camera zoom; weights the center-line distance term.
    center_weight : float
        Weight of the view-center distance term.

    Returns
    -------
    list of tuple of slice
        Chunk keys sorted from highest to lowest priority.
    """
    keys = _chunk_keys_product(chunk_keys)
    if not keys:
        return []
    centers = np.array([get_chunk_center(key)[-3:] for key in keys])
    camera_center = np.asarray(camera_center, dtype=float)[-3:]
    view_direction = np.asarray(view_direction, dtype=float)[-3:]

    relative = centers - camera_center
    depth = relative @ view_direction
    projected = view_direction * depth[:, np.newaxis]
    center_line_dist = np.linalg.norm(projected - relative, axis=-1)

    view_center = (
        (np.asarray(min_coord, dtype=float) + np.asarray(max_coord)) / 2
    )[-3:]
    center_view_dist = np.linalg.norm(centers - view_center, axis=-1)

    priority = (
        depth + zoom * center_line_dist + center_weight * center_view_dist
    )
    return [keys[i] for i in np.argsort(priority, kind='stable')]


def _chunk_id(chunk_key: tuple[slice, ...]) -> tuple[tuple[int, int], ...]:
    """Hashable identifier for a chunk key."""
    return tuple((int(sl.start), int(sl.stop)) for sl in chunk_key)


# ---------- background fetching ----------


@thread_worker
def _fetch_chunks(array, chunk_queue: list[tuple[slice, ...]]):
    """Fetch chunks from ``array`` one at a time, yielding each result.

    Yields ``(chunk_key, ndarray)`` tuples. Runs on a background thread;
    cancellation is delivered at yield points by the thread worker
    machinery.
    """
    for chunk_key in chunk_queue:
        start = time.monotonic()
        chunk = np.asarray(array[chunk_key])
        LOGGER.debug(
            'fetched chunk %s in %.3fs', chunk_key, time.monotonic() - start
        )
        yield chunk_key, chunk


class ProgressiveLoader:
    """Stream visible chunks into a multiscale image layer.

    Connects to the viewer's camera and dims events and keeps the layer's
    per-level :class:`VirtualData` intervals in sync with the visible
    region, fetching missing chunks on a background thread in priority
    order. Respects ``layer.locked_data_level`` (the resolution selector in
    the layer controls) because it always loads the level napari selected
    for rendering.

    Normally constructed by :func:`add_progressive_loading_image`; the
    instance is stored in ``layer.metadata['progressive_loader']``.

    Parameters
    ----------
    viewer : napari.Viewer
        The viewer the layer belongs to.
    layer : napari.layers.Image
        A multiscale image layer whose levels are ``VirtualData`` objects.
    data : MultiScaleVirtualData
        The coordinating multiscale wrapper for the layer's levels.
    debounce_ms : int
        Debounce interval for camera/dims events.
    refresh_interval_s : float
        Minimum time between layer refreshes while chunks stream in.
    resident_max_bytes : int
        Keep the coarsest level fully in memory if it is at most this big.
    interval_max_bytes : int
        Upper bound for a single level's resident interval.
    auto_level_3d : bool
        In 3D, automatically select the data level from the camera zoom
        (napari itself always renders the coarsest level in 3D). The level
        is driven through the layer's internal level lock without emitting
        events, so the resolution selector still reads "Auto"; choosing an
        explicit level in the selector suspends automatic selection until
        it is set back to "Auto". Levels too large for
        ``interval_max_bytes`` are skipped in favor of coarser ones.
    """

    def __init__(
        self,
        viewer: napari.Viewer,
        layer,
        data: MultiScaleVirtualData,
        *,
        debounce_ms: int = 100,
        refresh_interval_s: float = 0.1,
        resident_max_bytes: int = DEFAULT_RESIDENT_MAX_BYTES,
        interval_max_bytes: int = DEFAULT_INTERVAL_MAX_BYTES,
        auto_level_3d: bool = True,
    ):
        self._viewer = viewer
        self._layer = layer
        self._data = data
        self._refresh_interval_s = refresh_interval_s
        self._interval_max_bytes = interval_max_bytes
        self._auto_level_3d = auto_level_3d
        # Level we set through layer._locked_data_level for 3D auto mode
        # (None when we are not driving the level).
        self._auto_locked: int | None = None
        # True while the user has pinned an explicit level in the
        # resolution selector; suspends 3D auto level selection.
        self._user_locked = layer.locked_data_level is not None
        self._closed = False

        self._worker = None
        self._generation = 0
        self._active: tuple | None = None
        self._chunks_done = 0
        self._chunks_total = 0
        self._last_refresh = 0.0
        self._pbar = None
        self._resident_pbar = None

        self._resident_worker = None
        self._resident_level = len(data) - 1
        self._resident_max_bytes = resident_max_bytes
        self._resident_disabled = False
        self._last_clamp_message: str | None = None

        # Debounced so that continuous camera motion only triggers a fetch
        # pass once interaction settles.
        self._debounced_check = debounced(
            ensure_main_thread(self._check), timeout=debounce_ms
        )
        self._connections = [
            (viewer.camera.events, self._debounced_check),
            (viewer.dims.events.current_step, self._debounced_check),
            (viewer.dims.events.ndisplay, self._debounced_check),
            (layer.events.locked_data_level, self._debounced_check),
            # the locked_data_level event only fires for user/API writes
            # (3D auto mode bypasses the setter), so it reliably tells us
            # whether the user has pinned a level
            (layer.events.locked_data_level, self._on_user_locked_change),
            (layer.events.visible, self._debounced_check),
            # set_data fires after every (re-)slice, including the ones
            # napari runs when the data level or corners change; _check is
            # a cheap no-op when the view is already covered.
            (layer.events.set_data, self._debounced_check),
        ]
        for emitter, callback in self._connections:
            emitter.connect(callback)
        viewer.layers.events.removed.connect(self._on_layer_removed)

        # 2D multiscale slicing caches a one-time materialization of the
        # thumbnail (coarsest) level, which would freeze this layer's
        # pre-load (all-zero) content. Disable it: materializing a resident
        # VirtualData is a plain memory copy.
        layer._level_materializer = None

        self._check()

    # -- lifecycle --

    def _on_layer_removed(self, event) -> None:
        if event.value is self._layer:
            self.close()

    def close(self) -> None:
        """Disconnect from the viewer and stop all background fetching."""
        if self._closed:
            return
        self._closed = True
        self._release_auto_level()
        self._cancel_active()
        if self._resident_worker is not None:
            self._resident_worker.quit()
            self._resident_worker = None
        self._close_progress(self._resident_pbar)
        self._resident_pbar = None
        for emitter, callback in self._connections:
            with contextlib.suppress(ValueError, TypeError):
                emitter.disconnect(callback)
        self._connections = []
        with contextlib.suppress(ValueError, TypeError):
            self._viewer.layers.events.removed.disconnect(
                self._on_layer_removed
            )

    # -- view tracking --

    def _level_interval(self, level: int) -> tuple[np.ndarray, np.ndarray]:
        """Visible half-open interval for ``level``, in level coordinates.

        Displayed dimensions come from the layer's ``corner_pixels`` (which
        napari maintains in the coordinates of the current data level);
        non-displayed dimensions cover only the current dims step.
        """
        layer = self._layer
        vdata = self._data[level]
        ndim = vdata.ndim
        shape = np.asarray(vdata.shape, dtype=np.int64)

        min_coord = np.zeros(ndim, dtype=np.int64)
        max_coord = shape.copy()

        displayed = set(layer._slice_input.displayed)
        if self._viewer.dims.ndisplay == 2:
            corners = layer.corner_pixels
            for d in displayed:
                min_coord[d] = corners[0, d]
                max_coord[d] = corners[1, d] + 1

        self._restrict_to_current_step(level, displayed, min_coord, max_coord)

        min_coord = np.clip(min_coord, 0, shape)
        max_coord = np.clip(max_coord, 0, shape)
        return self._clamp_interval(vdata, min_coord, max_coord)

    def _restrict_to_current_step(
        self, level: int, displayed: set, min_coord, max_coord
    ) -> None:
        """Restrict non-displayed dims to the current dims step (in place)."""
        layer = self._layer
        vdata = self._data[level]
        ndim = vdata.ndim
        factors = np.asarray(self._data._scale_factors[level])
        try:
            data_point = np.asarray(
                layer.world_to_data(self._viewer.dims.point), dtype=float
            )
        except (ValueError, IndexError, TypeError):
            # pragma: no cover - layer/viewer dims mismatch fallback
            data_point = np.asarray(self._viewer.dims.point, dtype=float)[
                -ndim:
            ]
        for d in range(ndim):
            if d not in displayed:
                point = int(np.round(data_point[d] / factors[d]))
                point = min(max(point, 0), int(vdata.shape[d]) - 1)
                min_coord[d] = point
                max_coord[d] = point + 1

    def _clamp_interval(
        self, vdata: VirtualData, min_coord, max_coord
    ) -> tuple[np.ndarray, np.ndarray]:
        """Shrink an interval around its center to respect the memory cap."""
        min_coord = np.array(min_coord, dtype=np.int64)
        max_coord = np.array(max_coord, dtype=np.int64)
        requested_min, requested_max = min_coord.copy(), max_coord.copy()
        itemsize = vdata.dtype.itemsize
        extent = np.maximum(max_coord - min_coord, 1)
        max_elements = self._interval_max_bytes // itemsize
        clamped = False
        while np.prod(extent, dtype=np.int64) > max_elements:
            widest = int(np.argmax(extent))
            center = (min_coord[widest] + max_coord[widest]) // 2
            half = max(extent[widest] // 4, 1)
            min_coord[widest] = max(center - half, min_coord[widest])
            max_coord[widest] = min(center + half, max_coord[widest])
            new_extent = max(max_coord[widest] - min_coord[widest], 1)
            if new_extent == extent[widest]:  # pragma: no cover - safety
                break
            extent[widest] = new_extent
            clamped = True
        if clamped:
            message = (
                f'progressive loading: visible interval '
                f'[{requested_min.tolist()}, {requested_max.tolist()}) '
                f'exceeds {self._interval_max_bytes} bytes; clamped to '
                f'[{min_coord.tolist()}, {max_coord.tolist()})'
            )
            if message != self._last_clamp_message:
                self._last_clamp_message = message
                LOGGER.warning(message)
        return min_coord, max_coord

    # -- 3D automatic level selection --

    def _on_user_locked_change(self, event=None) -> None:
        """Track explicit level pins made through the resolution selector.

        This only fires for writes through the public
        ``locked_data_level`` setter — 3D auto mode writes the private
        attribute directly — so a non-None value here is always a user
        (or API) choice.
        """
        self._user_locked = self._layer.locked_data_level is not None
        if self._user_locked:
            self._auto_locked = None

    def _zoom_target_level_3d(self, max_pixel_size: float = 4.0) -> int:
        """Pick the 3D data level appropriate for the current camera zoom.

        Chooses the coarsest level whose voxels project to at most
        ``max_pixel_size`` screen pixels (so the displayed resolution
        roughly matches the screen, like napari's 2D level selection),
        then falls back to coarser levels until the visible volume fits
        the memory budget.
        """
        layer = self._layer
        zoom = float(self._viewer.camera.zoom)
        displayed = list(layer._slice_input.displayed)
        n_levels = len(self._data)

        target = 0
        for level in range(n_levels - 1, -1, -1):
            factors = np.take(
                np.asarray(self._data._scale_factors[level]), displayed
            )
            pixel_size = zoom * float(np.max(factors))
            if pixel_size <= max_pixel_size:
                target = level
                break

        # Coarsen until the visible volume (full displayed extent at the
        # current step of other dims) fits the interval budget.
        for level in range(target, n_levels):
            vdata = self._data[level]
            extent = np.take(
                np.asarray(vdata.shape, dtype=np.int64), displayed
            )
            nbytes = np.prod(extent, dtype=np.int64) * vdata.dtype.itemsize
            if nbytes <= self._interval_max_bytes:
                return level
        return n_levels - 1

    def _apply_auto_level(self) -> None:
        """Drive the layer's data level from zoom while in 3D Auto mode.

        Writes ``layer._locked_data_level`` directly (not the public
        setter) so no ``locked_data_level`` event is emitted and the
        resolution selector keeps displaying "Auto".
        """
        layer = self._layer
        if not self._auto_level_3d or self._user_locked:
            return
        if self._viewer.dims.ndisplay != 3:
            self._release_auto_level()
            return
        target = self._zoom_target_level_3d()
        if target == self._auto_locked and layer._locked_data_level == target:
            return
        self._auto_locked = target
        layer._locked_data_level = target
        layer._data_level = target
        # Mirror the corner_pixels update of the locked_data_level setter.
        displayed_axes = layer._slice_input.displayed
        shape_at_level = np.array(layer.level_shapes[target])
        corners = np.zeros((2, layer.ndim), dtype=int)
        corners[1, displayed_axes] = shape_at_level[displayed_axes] - 1
        layer.corner_pixels = corners
        layer.refresh(extent=False)

    def _release_auto_level(self) -> None:
        """Give level control back to napari (e.g. on 2D or teardown)."""
        layer = self._layer
        if (
            self._auto_locked is not None
            and not self._user_locked
            and layer._locked_data_level == self._auto_locked
        ):
            layer._locked_data_level = None
            layer._reset_data_level()
            # _reset_data_level cleared the lock state; preserve our flag
            self._auto_locked = None
            layer.refresh(extent=False)
        else:
            self._auto_locked = None

    # -- fetch passes --

    def _check(self, event=None) -> None:
        """Start a fetch pass if the current view is not fully loaded."""
        if self._closed:
            return
        layer = self._layer
        if not layer.visible:
            self._cancel_active()
            return
        self._apply_auto_level()
        self._ensure_resident()
        level = int(layer.data_level)
        min_coord, max_coord = self._level_interval(level)
        if np.any(max_coord <= min_coord):
            return
        if level == self._resident_level and self._resident_worker is not None:
            # The coarsest level is being filled by the resident worker.
            return
        view_key = (level, tuple(min_coord), tuple(max_coord))
        if view_key == self._active:
            # A pass for exactly this view is in flight or already done.
            return
        self._start_fetch(level, min_coord, max_coord)

    def _backdrop_level(self, level: int) -> int | None:
        coarsest = len(self._data) - 1
        if level == coarsest:
            return None
        if self._data[coarsest].interval is None:
            return None
        return coarsest

    def _start_fetch(self, level: int, min_coord, max_coord) -> None:
        self._cancel_active()
        vdata = self._data[level]

        self._data.set_interval(
            level,
            min_coord,
            max_coord,
            backdrop_level=self._backdrop_level(level),
        )
        self._active = (level, tuple(min_coord), tuple(max_coord))

        interval = vdata.interval
        keys = chunk_slices(vdata, interval=interval)
        if self._viewer.dims.ndisplay == 3:
            queue = self._prioritize_3d(level, keys, interval)
        else:
            queue = chunk_priority_2D(keys, interval[0], interval[1])
        queue = [
            key for key in queue if _chunk_id(key) not in vdata.loaded_chunks
        ]

        if not queue:
            # Everything visible is already resident (e.g. carried over
            # from the previous interval); make sure the canvas shows it.
            self._refresh(final=True)
            return

        LOGGER.debug(
            'starting fetch pass: level=%d interval=%s chunks=%d',
            level,
            interval,
            len(queue),
        )

        self._generation += 1
        generation = self._generation
        self._chunks_done = 0
        self._chunks_total = len(queue)
        self._pbar = self._make_progress(
            len(queue), f'{self._layer.name}: loading level {level}'
        )

        worker = _fetch_chunks(vdata.array, queue)
        worker.yielded.connect(
            lambda result: self._on_chunk(generation, vdata, result)
        )
        worker.finished.connect(lambda: self._on_fetch_finished(generation))
        self._worker = worker
        worker.start()

    def _prioritize_3d(self, level, keys, interval):
        camera = self._viewer.camera
        factors = np.asarray(self._data._scale_factors[level])
        displayed = list(self._layer._slice_input.displayed)[-3:]
        camera_center = np.asarray(camera.center, dtype=float) / np.take(
            factors, displayed
        )
        return chunk_priority_3D(
            keys,
            interval[0],
            interval[1],
            camera_center=camera_center,
            view_direction=camera.view_direction,
            zoom=camera.zoom,
        )

    def _make_progress(self, total: int, description: str):
        """Best-effort progress bar shown in the napari activity dock."""
        if total < PROGRESS_MIN_CHUNKS:
            return None
        try:
            return progress(total=total, desc=description)
        except Exception:  # noqa: BLE001 # pragma: no cover - cosmetic
            return None

    @staticmethod
    def _close_progress(pbar) -> None:
        if pbar is not None:
            with contextlib.suppress(Exception):
                pbar.close()

    def _cancel_active(self) -> None:
        self._generation += 1
        if self._worker is not None:
            self._worker.quit()
            self._worker = None
        self._active = None
        self._close_progress(self._pbar)
        self._pbar = None

    def _on_chunk(self, generation: int, vdata: VirtualData, result) -> None:
        if generation != self._generation or self._closed:
            return
        chunk_key, chunk = result
        vdata.set_offset(chunk_key, chunk)
        vdata.loaded_chunks.add(_chunk_id(chunk_key))
        self._chunks_done += 1
        if self._pbar is not None:
            self._pbar.update(1)
        final = self._chunks_done >= self._chunks_total
        if final:
            self._close_progress(self._pbar)
            self._pbar = None
        self._refresh(final=final)

    def _on_fetch_finished(self, generation: int) -> None:
        if generation != self._generation or self._closed:
            return
        self._worker = None

    def _refresh(self, final: bool = False) -> None:
        now = time.monotonic()
        if not final and now - self._last_refresh < self._refresh_interval_s:
            return
        self._last_refresh = now
        self._layer.refresh(extent=False, highlight=False, thumbnail=final)

    # -- resident coarsest level --

    def _resident_target_interval(
        self,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """The interval of the coarsest level worth keeping resident.

        Prefers the full level (so backdrops/thumbnails work everywhere);
        if that exceeds the memory budget — e.g. the coarsest level of a
        large timelapse spans every timepoint — falls back to the full
        *displayed* extent at the current step of the other dimensions.
        Returns ``None`` if even that is too large.
        """
        vdata = self._data[self._resident_level]
        itemsize = vdata.dtype.itemsize
        if vdata.size * itemsize <= self._resident_max_bytes:
            return (
                np.zeros(vdata.ndim, dtype=np.int64),
                np.asarray(vdata.shape, dtype=np.int64),
            )

        min_coord = np.zeros(vdata.ndim, dtype=np.int64)
        max_coord = np.asarray(vdata.shape, dtype=np.int64)
        displayed = set(self._layer._slice_input.displayed)
        self._restrict_to_current_step(
            self._resident_level, displayed, min_coord, max_coord
        )
        nbytes = np.prod(max_coord - min_coord, dtype=np.int64) * itemsize
        if nbytes > self._resident_max_bytes:
            if not self._resident_disabled:
                self._resident_disabled = True
                LOGGER.warning(
                    'coarsest level slice is %.0f MB (> %.0f MB); not '
                    'keeping it resident. Backdrops and thumbnails will be '
                    'limited to the visible region.',
                    nbytes / 1e6,
                    self._resident_max_bytes / 1e6,
                )
            return None
        return min_coord, max_coord

    def _ensure_resident(self) -> None:
        """Keep the coarsest level resident around the current view.

        The resident level provides instant low-resolution context when
        panning/zooming, backs the layer thumbnail, and is the backdrop
        source for finer levels. When the resident interval no longer
        covers the current dims step (e.g. the time slider moved), it is
        refilled.
        """
        target = self._resident_target_interval()
        if target is None:
            return
        min_coord, max_coord = target
        vdata = self._data[self._resident_level]
        if vdata.covers(min_coord, max_coord):
            if self._resident_worker is not None:
                return
            keys = chunk_slices(vdata, interval=(min_coord, max_coord))
            if all(
                _chunk_id(key) in vdata.loaded_chunks
                for key in itertools.product(*keys)
            ):
                return
        self._start_resident_fill(min_coord, max_coord)

    def _start_resident_fill(self, min_coord, max_coord) -> None:
        if self._resident_worker is not None:
            self._resident_worker.quit()
            self._resident_worker = None
        self._close_progress(self._resident_pbar)
        self._resident_pbar = None
        vdata = self._data[self._resident_level]
        vdata.set_interval(min_coord, max_coord)
        interval = vdata.interval
        keys = chunk_slices(vdata, interval=interval)
        queue = [
            key
            for key in chunk_priority_2D(keys, interval[0], interval[1])
            if _chunk_id(key) not in vdata.loaded_chunks
        ]
        if not queue:
            return

        self._resident_pbar = self._make_progress(
            len(queue), f'{self._layer.name}: loading overview'
        )
        worker = _fetch_chunks(vdata.array, queue)

        def on_chunk(result, vdata=vdata):
            if self._closed or self._resident_worker is not worker:
                return
            chunk_key, chunk = result
            vdata.set_offset(chunk_key, chunk)
            vdata.loaded_chunks.add(_chunk_id(chunk_key))
            if self._resident_pbar is not None:
                self._resident_pbar.update(1)
            if self._layer.data_level == self._resident_level:
                self._refresh()

        def on_finished():
            if self._closed or self._resident_worker is not worker:
                return
            self._resident_worker = None
            self._close_progress(self._resident_pbar)
            self._resident_pbar = None
            self._refresh(final=True)
            # Re-evaluate coverage now that backdrops are available.
            self._check()

        worker.yielded.connect(on_chunk)
        worker.finished.connect(on_finished)
        self._resident_worker = worker
        worker.start()


# ---------- public entry point ----------


def _estimate_contrast_limits(array) -> tuple[float, float] | None:
    """Estimate contrast limits from a central sample of an array."""
    try:
        key = []
        for size in array.shape:
            center = int(size) // 2
            half = min(int(size), 256) // 2
            key.append(slice(max(center - half, 0), center + half + 1))
        sample = np.asarray(array[tuple(key)])
    except Exception:  # pragma: no cover - estimation is best-effort
        LOGGER.exception('contrast limit estimation failed')
        return None
    if sample.size == 0:
        return None
    low = float(np.min(sample))
    high = float(np.max(sample))
    if low == high:
        high = low + 1
    return low, high


def add_progressive_loading_image(
    img,
    viewer: napari.Viewer | None = None,
    contrast_limits: tuple[float, float] | None = None,
    colormap: str = 'gray',
    rendering: str = 'attenuated_mip',
    name: str | None = None,
    auto_level_3d: bool = True,
    **layer_kwargs,
):
    """Add a progressively loading multiscale image to a viewer.

    The image is added as a *single* multiscale layer. Chunks of the level
    napari selects for rendering are streamed in on a background thread,
    nearest to the view center first, while coarser data is shown as a
    backdrop. The layer's resolution selector (``locked_data_level``) is
    respected.

    Parameters
    ----------
    img : sequence of array-like
        Multiscale image data, highest resolution first. Levels may be
        zarr arrays, dask arrays, or anything implementing ``shape``,
        ``dtype``, ``chunks``/``chunksize`` and ``__getitem__``.
    viewer : napari.Viewer, optional
        The viewer to add the image to. A new one is created if not given.
    contrast_limits : tuple of float, optional
        Contrast limits for the layer. If not given, they are estimated
        from a central sample of the coarsest level.
    colormap : str
        Colormap for the layer.
    rendering : str
        3D rendering mode for the layer.
    name : str, optional
        Layer name.
    auto_level_3d : bool
        In 3D, automatically pick the rendered data level from the camera
        zoom (napari itself always uses the coarsest level in 3D). The
        resolution selector stays on "Auto"; pinning an explicit level
        there suspends automatic selection.
    **layer_kwargs
        Additional keyword arguments passed to ``viewer.add_image``.

    Returns
    -------
    napari.layers.Image
        The created layer. The active :class:`ProgressiveLoader` is stored
        in ``layer.metadata['progressive_loader']``.
    """
    if viewer is None:
        from napari import Viewer

        viewer = Viewer()

    data = MultiScaleVirtualData(img)

    if contrast_limits is None:
        contrast_limits = _estimate_contrast_limits(data.arrays[-1])

    layer = viewer.add_image(
        data._data,
        multiscale=True,
        contrast_limits=contrast_limits,
        colormap=colormap,
        rendering=rendering,
        name=name,
        **layer_kwargs,
    )
    loader = ProgressiveLoader(
        viewer, layer, data, auto_level_3d=auto_level_3d
    )
    layer.metadata['progressive_loader'] = loader
    return layer
