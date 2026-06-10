"""Per-frame metering of vispy GLIR texture uploads.

vispy drains its entire GLIR command queue inside whichever draw happens
next — including interaction frames. With progressive loading, dozens of
megabytes of ``glTexSubImage3D`` traffic can accumulate between draws and
the drain then blocks the main thread for hundreds of milliseconds to
seconds (macOS GL-over-Metal is the worst case: ~125 MB/s effective with
multi-second outliers for large single uploads).

This module patches :meth:`vispy.gloo.glir._GlirQueueShare.flush` so that

- a single large texture ``DATA`` command is split into small slab
  sub-uploads (contiguous views along the leading axes, no copies), and
- each frame spends at most ``frame_budget_bytes`` on texture uploads;
  the remainder is carried to the *next* frame and a redraw is scheduled
  so the carry keeps draining even without interaction.

GLIR ordering semantics are preserved per object: once a command for a
texture is deferred, every later command for that same texture id is
deferred behind it. ``SIZE`` (re-specification) and ``DELETE`` commands
cancel any earlier carried uploads for their id, mirroring vispy's own
queue filtering.

The patch is installed lazily by progressive loading
(:func:`napari.experimental.add_progressive_loading_image`); vanilla
napari is unaffected. Tunables (env vars override arguments):

- ``NAPARI_GLIR_METERING=0`` disables installation entirely.
- ``NAPARI_GLIR_TEX_BYTES_PER_FRAME`` per-frame upload budget (bytes).
- ``NAPARI_GLIR_TEX_SLAB_BYTES`` maximum size of a single sub-upload.

Run ``python -m napari.experimental._glir_metering`` for a standalone
vispy-only benchmark of draw time vs. texture upload size (no napari),
with and without metering.
"""

from __future__ import annotations

import logging
import os
import time
import weakref

import numpy as np

LOGGER = logging.getLogger('napari.experimental._glir_metering')

DEFAULT_FRAME_BUDGET_BYTES = 4 * 2**20
DEFAULT_SLAB_BYTES = 1 * 2**20

# Commands that operate on a specific GLIR object id and therefore must
# stay ordered behind any deferred command for the same id.
_OBJECT_COMMANDS = frozenset(
    {'DATA', 'SIZE', 'WRAPPING', 'INTERPOLATION', 'DELETE'}
)

_original_flush = None
_hooked_canvases: weakref.WeakSet = weakref.WeakSet()


class _ParserState:
    """Per-GlirParser metering state (carry survives across flushes)."""

    def __init__(self, frame_budget_bytes: int, slab_bytes: int):
        self.frame_budget = int(frame_budget_bytes)
        self.slab_bytes = min(int(slab_bytes), self.frame_budget)
        self.budget_left = self.frame_budget
        self.carry: list[tuple] = []
        self.last_reset = time.perf_counter()

    def reset_budget(self):
        self.budget_left = self.frame_budget
        self.last_reset = time.perf_counter()


# keyed by GlirParser so canvases sharing a context share one budget
_states: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _state_for(parser) -> _ParserState:
    state = _states.get(parser)
    if state is None:
        # read the module globals at call time: install() retunes them
        state = _ParserState(DEFAULT_FRAME_BUDGET_BYTES, DEFAULT_SLAB_BYTES)
        _states[parser] = state
    return state


def _is_metered_texture(ob) -> bool:
    """Whether uploads to this GLIR object count against the budget.

    Only 3D textures are metered: they are the pathological path on
    macOS, and 2D texture uploads have profiled fine.
    """
    from vispy.gloo import glir

    return isinstance(ob, glir.GlirTexture3D)


def _split_slabs(offset, data, slab_bytes):
    """Split a texture DATA payload into <= slab_bytes sub-uploads.

    Splits along the leading texel axis first (contiguous views of a
    C-contiguous source), recursing into the second axis when a single
    leading slice is itself too large. Yields ``(offset, subarray)``.
    """
    if data.nbytes <= slab_bytes or data.ndim < 2 or data.shape[0] == 0:
        yield tuple(offset), data
        return
    bytes_per_slice = data.nbytes // data.shape[0]
    step = max(1, int(slab_bytes // max(1, bytes_per_slice)))
    for start in range(0, data.shape[0], step):
        sub = data[start : start + step]
        sub_offset = list(offset)
        sub_offset[0] += start
        if sub.nbytes > slab_bytes and sub.shape[0] == 1 and sub.ndim >= 3:
            # one leading slice is still too big: split its rows
            for in_off, in_sub in _split_slabs(
                offset=list(sub_offset[1:]),
                data=sub[0],
                slab_bytes=slab_bytes,
            ):
                yield (sub_offset[0], *in_off), in_sub[np.newaxis]
        else:
            yield tuple(sub_offset), sub


def _drop_deleted_carry(carry, new_commands):
    """Drop carried uploads whose texture is deleted in this flush.

    The carry only ever holds commands for metered textures, so this
    cannot affect other object types (notably shaders, whose DATA must
    survive even though vispy DELETEs them right after LINK).
    """
    deleted = {c[1] for c in new_commands if c[0] == 'DELETE'}
    if not deleted:
        return carry
    return [c for c in carry if c[1] not in deleted]


def _metered_parse(parser, commands, state):
    """Execute commands under the upload budget; return the leftovers."""
    # mirror GlirParser.parse's deferred deletion bookkeeping, which we
    # bypass by calling _parse directly
    from vispy.gloo.glir import JUST_DELETED

    for id_ in [
        id_ for id_, val in parser._objects.items() if val == JUST_DELETED
    ]:
        parser._objects.pop(id_)

    deferred_ids = set()
    leftover = []
    for command in commands:
        cmd = command[0]
        id_ = command[1] if len(command) > 1 else None
        if id_ in deferred_ids and cmd in _OBJECT_COMMANDS:
            leftover.append(command)
            continue
        if cmd == 'DATA':
            ob = parser._objects.get(id_, None)
            if _is_metered_texture(ob):
                offset, data = command[2], command[3]
                executed_any = False
                for sub_offset, sub in _split_slabs(
                    offset, data, state.slab_bytes
                ):
                    # always make progress: with a full budget, upload at
                    # least one slab even if it alone exceeds the budget
                    fresh = state.budget_left >= state.frame_budget
                    if state.budget_left <= 0 or (
                        sub.nbytes > state.budget_left
                        and not (fresh and not executed_any)
                    ):
                        deferred_ids.add(id_)
                        leftover.append(('DATA', id_, sub_offset, sub))
                        continue
                    if not sub.flags['C_CONTIGUOUS']:
                        sub = np.ascontiguousarray(sub)
                    parser._parse(('DATA', id_, sub_offset, sub))
                    state.budget_left -= sub.nbytes
                    executed_any = True
                continue
        parser._parse(command)
    return leftover


def _attach_reset_hook(canvas):
    """Reset the upload budget at the start of each canvas draw."""
    if canvas is None or canvas in _hooked_canvases:
        return
    canvas_ref = weakref.ref(canvas)

    def _on_draw(event=None):
        c = canvas_ref()
        if c is None:
            return
        try:
            parser = c.context.shared.parser
        except AttributeError:
            return
        _state_for(parser).reset_budget()

    # position='first': run before the scene draw issues any flushes
    canvas.events.draw.connect(_on_draw, position='first')
    _hooked_canvases.add(canvas)


def _metered_flush(self, parser):
    """Replacement for _GlirQueueShare.flush with per-frame metering."""
    from vispy.gloo.context import get_current_canvas

    if self._verbose:
        show = self._verbose if isinstance(self._verbose, str) else None
        self.show(show)

    state = _state_for(parser)
    canvas = get_current_canvas()
    _attach_reset_hook(canvas)
    # fallback when no canvas draw hook fires (offscreen / bare gloo):
    # never let the carry starve for more than 0.25 s
    if (
        state.budget_left < state.frame_budget
        and time.perf_counter() - state.last_reset > 0.25
    ):
        state.reset_budget()

    carry, state.carry = state.carry, []
    new_commands = self.clear()
    carry = _drop_deleted_carry(carry, new_commands)
    commands = self._filter(carry + new_commands, parser)
    state.carry = _metered_parse(parser, commands, state)

    if state.carry and canvas is not None:
        # keep draining even without interaction
        canvas.update()


def install(
    frame_budget_bytes: int | None = None,
    slab_bytes: int | None = None,
) -> bool:
    """Install GLIR texture-upload metering (idempotent).

    Returns True if metering is active after the call.
    """
    global _original_flush, DEFAULT_FRAME_BUDGET_BYTES, DEFAULT_SLAB_BYTES

    if os.environ.get('NAPARI_GLIR_METERING', '1') in ('0', 'false', ''):
        return False

    frame_budget_bytes = int(
        os.environ.get(
            'NAPARI_GLIR_TEX_BYTES_PER_FRAME',
            frame_budget_bytes or DEFAULT_FRAME_BUDGET_BYTES,
        )
    )
    slab_bytes = int(
        os.environ.get(
            'NAPARI_GLIR_TEX_SLAB_BYTES', slab_bytes or DEFAULT_SLAB_BYTES
        )
    )

    # re-parameterize existing states (and set defaults for new ones)
    DEFAULT_FRAME_BUDGET_BYTES = frame_budget_bytes
    DEFAULT_SLAB_BYTES = slab_bytes
    for state in _states.values():
        state.frame_budget = frame_budget_bytes
        state.slab_bytes = min(slab_bytes, frame_budget_bytes)

    from vispy.gloo import glir

    if _original_flush is None:
        _original_flush = glir._GlirQueueShare.flush
        glir._GlirQueueShare.flush = _metered_flush
        LOGGER.info(
            'GLIR texture upload metering installed '
            '(budget=%d B/frame, slab=%d B)',
            frame_budget_bytes,
            slab_bytes,
        )
    return True


def uninstall():
    """Remove the patch and flush any carried uploads on next draw."""
    global _original_flush
    if _original_flush is None:
        return
    from vispy.gloo import glir

    glir._GlirQueueShare.flush = _original_flush
    _original_flush = None
    # re-queue carried uploads so they are not lost
    for parser, state in list(_states.items()):
        if state.carry:
            commands, state.carry = state.carry, []
            parser.parse(commands)
    _states.clear()


def is_installed() -> bool:
    return _original_flush is not None


def _benchmark():  # pragma: no cover - manual profiling tool
    """Pure-vispy benchmark: draw time vs. 3D texture upload size.

    Renders a vispy Volume and times the draw immediately after
    ``set_data`` calls of increasing size, with metering off and on.
    This is the minimal repro for the macOS GL-over-Metal upload stalls
    (run on Apple Silicon and compare the two curves).
    """
    import argparse

    parser_ = argparse.ArgumentParser(description=__doc__)
    parser_.add_argument('--size', type=int, default=512, help='volume edge')
    parser_.add_argument('--repeats', type=int, default=5)
    parser_.add_argument(
        '--budget', type=int, default=DEFAULT_FRAME_BUDGET_BYTES
    )
    parser_.add_argument('--slab', type=int, default=DEFAULT_SLAB_BYTES)
    args = parser_.parse_args()

    from vispy import app, scene

    canvas = scene.SceneCanvas(keys=None, size=(800, 600), show=True)
    view = canvas.central_widget.add_view()
    n = args.size
    rng = np.random.default_rng(0)
    base = rng.random((n, n, n), dtype=np.float32)
    volume = scene.visuals.Volume(base, parent=view.scene)
    view.camera = scene.cameras.TurntableCamera(parent=view.scene, fov=60.0)

    def timed_draw():
        t0 = time.perf_counter()
        canvas.render()  # forces a full draw + flush
        return time.perf_counter() - t0

    sub_mb_sizes = [1, 2, 4, 8, 16, 32, 64]
    for label, metered in (('unmetered', False), ('metered', True)):
        if metered:
            os.environ.pop('NAPARI_GLIR_METERING', None)
            install(frame_budget_bytes=args.budget, slab_bytes=args.slab)
        else:
            uninstall()
        # warm up
        for _ in range(3):
            timed_draw()
        print(f'--- {label} ---')  # noqa: T201
        for mb in sub_mb_sizes:
            nz = max(1, int(mb * 2**20 // (n * n * base.itemsize)))
            if nz > n:
                break
            sub = rng.random((nz, n, n), dtype=np.float32)
            times = []
            for i in range(args.repeats):
                z = (i * nz) % max(1, n - nz)
                t0 = time.perf_counter()
                volume._texture.set_data(sub, offset=(z, 0, 0))
                t_set = time.perf_counter() - t0
                t_draw = timed_draw()
                # with metering, drain the carry and count total time.
                # canvas.render() bypasses the draw event, so emulate the
                # per-frame budget reset a real paint event would trigger.
                t_drain = 0.0
                if metered:
                    while _states and any(s.carry for s in _states.values()):
                        for s in _states.values():
                            s.reset_budget()
                        t_drain += timed_draw()
                times.append((t_set, t_draw, t_drain))
            worst = max(t[1] for t in times)
            total = max(t[1] + t[2] for t in times)
            print(  # noqa: T201
                f'{mb:>4} MB sub-upload: worst single draw '
                f'{worst * 1e3:7.1f} ms, total incl. drain '
                f'{total * 1e3:7.1f} ms'
            )
    app.process_events()
    canvas.close()


if __name__ == '__main__':  # pragma: no cover
    _benchmark()
