"""Tests for per-frame GLIR texture upload metering.

These run without a GL context: a fake parser records what would be
executed and real ``GlirTexture3D`` instances are created via ``__new__``
(no GL calls) purely so isinstance checks pass.
"""

import numpy as np
import pytest
from vispy.gloo import glir

from napari.experimental import _glir_metering as gm


class FakeParser:
    """Records parsed commands instead of touching GL."""

    def __init__(self):
        self._objects = {}
        self.executed = []

    def add_texture3d(self, id_):
        tex = glir.GlirTexture3D.__new__(glir.GlirTexture3D)
        self._objects[id_] = tex
        return tex

    def _parse(self, command):
        self.executed.append(command)

    def parse(self, commands):
        for c in commands:
            self._parse(c)


@pytest.fixture
def parser():
    return FakeParser()


def make_state(budget, slab):
    return gm._ParserState(frame_budget_bytes=budget, slab_bytes=slab)


def data_bytes(commands):
    return sum(c[3].nbytes for c in commands if c[0] == 'DATA')


def test_small_commands_pass_through_in_order(parser):
    parser.add_texture3d(1)
    state = make_state(budget=2**20, slab=2**20)
    small = np.zeros((4, 4, 4), dtype=np.uint8)
    commands = [
        ('UNIFORM', 7, 'u_x', 'float', 1.0),
        ('DATA', 1, (0, 0, 0), small),
        ('DRAW', 7, 'triangles', None, 1),
    ]
    leftover = gm._metered_parse(parser, commands, state)
    assert leftover == []
    assert [c[0] for c in parser.executed] == ['UNIFORM', 'DATA', 'DRAW']


def test_large_data_split_into_slabs(parser):
    parser.add_texture3d(1)
    # 64 z-slices of 64 KiB each = 4 MiB total, 256 KiB slabs
    data = (
        np.arange(64 * 256 * 256, dtype=np.uint8)
        .reshape(64, 256, 256)
        .copy()  # own the buffer so the .base view check below holds
    )
    state = make_state(budget=64 * 2**20, slab=256 * 2**10)
    leftover = gm._metered_parse(parser, [('DATA', 1, (8, 0, 0), data)], state)
    assert leftover == []
    executed = parser.executed
    assert all(c[0] == 'DATA' for c in executed)
    assert len(executed) == 16  # 4 MiB / 256 KiB
    # offsets advance along z from the original offset
    assert [c[2][0] for c in executed] == [8 + 4 * i for i in range(16)]
    assert all(c[2][1:] == (0, 0) for c in executed)
    # reassembly reproduces the original payload
    reassembled = np.concatenate([c[3] for c in executed], axis=0)
    np.testing.assert_array_equal(reassembled, data)
    # slabs are views, not copies
    assert all(c[3].base is data for c in executed)


def test_single_slice_too_large_splits_rows(parser):
    parser.add_texture3d(1)
    # one z-slice of 1 MiB with a 256 KiB slab limit -> split along y
    data = np.zeros((1, 1024, 1024), dtype=np.uint8)
    state = make_state(budget=64 * 2**20, slab=256 * 2**10)
    leftover = gm._metered_parse(parser, [('DATA', 1, (3, 0, 0), data)], state)
    assert leftover == []
    assert len(parser.executed) == 4
    assert [c[2] for c in parser.executed] == [
        (3, 256 * i, 0) for i in range(4)
    ]
    assert all(c[3].shape == (1, 256, 1024) for c in parser.executed)


def test_budget_defers_remainder(parser):
    parser.add_texture3d(1)
    data = np.zeros((16, 256, 256), dtype=np.uint8)  # 1 MiB
    state = make_state(budget=512 * 2**10, slab=128 * 2**10)
    leftover = gm._metered_parse(parser, [('DATA', 1, (0, 0, 0), data)], state)
    assert data_bytes(parser.executed) == 512 * 2**10
    assert data_bytes(leftover) == 512 * 2**10
    assert state.budget_left == 0
    # next frame: fresh budget drains the carry
    state.reset_budget()
    leftover2 = gm._metered_parse(parser, leftover, state)
    assert leftover2 == []
    assert data_bytes(parser.executed) == data.nbytes


def test_non_texture_commands_run_past_deferred_uploads(parser):
    parser.add_texture3d(1)
    data = np.zeros((16, 256, 256), dtype=np.uint8)  # 1 MiB
    state = make_state(budget=256 * 2**10, slab=256 * 2**10)
    commands = [
        ('DATA', 1, (0, 0, 0), data),
        ('UNIFORM', 7, 'u_x', 'float', 1.0),
        ('DRAW', 7, 'triangles', None, 1),
        # but same-id commands stay ordered behind the deferred upload
        ('WRAPPING', 1, ('clamp_to_edge',) * 3),
    ]
    leftover = gm._metered_parse(parser, commands, state)
    executed_kinds = [c[0] for c in parser.executed]
    assert 'UNIFORM' in executed_kinds
    assert 'DRAW' in executed_kinds
    assert 'WRAPPING' not in executed_kinds
    assert leftover[-1][0] == 'WRAPPING'
    assert all(c[0] == 'DATA' for c in leftover[:-1])


def test_oversized_single_slab_still_makes_progress(parser):
    parser.add_texture3d(1)
    # a slab that alone exceeds the whole budget: with a fresh budget it
    # must execute anyway, otherwise the carry never drains
    data = np.zeros((1, 64, 64), dtype=np.uint8)
    state = make_state(budget=1024, slab=1024)
    state.slab_bytes = data.nbytes  # force a single indivisible slab
    leftover = gm._metered_parse(parser, [('DATA', 1, (0, 0, 0), data)], state)
    assert leftover == []
    assert data_bytes(parser.executed) == data.nbytes


def test_delete_drops_carried_uploads():
    data = np.zeros((4, 4, 4), dtype=np.uint8)
    carry = [
        ('DATA', 1, (0, 0, 0), data),
        ('DATA', 2, (0, 0, 0), data),
    ]
    new_commands = [
        ('DELETE', 1),
        # shader-style DATA-then-DELETE in the new commands must be
        # left alone (vispy deletes shaders right after LINK)
        ('DATA', 3, 0, 'shader code'),
        ('DELETE', 3),
    ]
    out = gm._drop_deleted_carry(carry, new_commands)
    assert out == [('DATA', 2, (0, 0, 0), data)]
    assert new_commands == [
        ('DELETE', 1),
        ('DATA', 3, 0, 'shader code'),
        ('DELETE', 3),
    ]


def test_unknown_object_data_unmetered(parser):
    # DATA for buffers/shaders (not metered textures) passes through
    state = make_state(budget=1, slab=1)
    big = np.zeros((1024, 1024), dtype=np.float32)
    leftover = gm._metered_parse(parser, [('DATA', 99, 0, big)], state)
    assert leftover == []
    assert parser.executed == [('DATA', 99, 0, big)]


def test_install_uninstall_idempotent(monkeypatch):
    monkeypatch.delenv('NAPARI_GLIR_METERING', raising=False)
    original = glir._GlirQueueShare.flush
    try:
        assert gm.install()
        assert gm.is_installed()
        patched = glir._GlirQueueShare.flush
        assert patched is not original
        assert gm.install()  # second install is a no-op
        assert glir._GlirQueueShare.flush is patched
    finally:
        gm.uninstall()
    assert glir._GlirQueueShare.flush is original
    assert not gm.is_installed()
    gm.uninstall()  # second uninstall is a no-op


def test_install_disabled_by_env(monkeypatch):
    monkeypatch.setenv('NAPARI_GLIR_METERING', '0')
    original = glir._GlirQueueShare.flush
    assert not gm.install()
    assert glir._GlirQueueShare.flush is original


def test_metered_flush_carries_and_drains(monkeypatch):
    monkeypatch.delenv('NAPARI_GLIR_METERING', raising=False)
    parser = FakeParser()
    parser.add_texture3d(1)
    try:
        assert gm.install(
            frame_budget_bytes=512 * 2**10,
            slab_bytes=128 * 2**10,
        )
        queue = glir.GlirQueue()
        data = np.zeros((16, 256, 256), dtype=np.uint8)  # 1 MiB
        queue.command('DATA', 1, (0, 0, 0), data)
        queue.flush(parser)
        assert data_bytes(parser.executed) == 512 * 2**10
        state = gm._states[parser]
        assert data_bytes(state.carry) == 512 * 2**10
        # next frame: budget reset (simulating the canvas draw hook),
        # empty queue still drains the carry
        state.reset_budget()
        queue.flush(parser)
        assert data_bytes(parser.executed) == data.nbytes
        assert state.carry == []
    finally:
        gm.uninstall()


def test_metered_flush_new_size_cancels_carry(monkeypatch):
    monkeypatch.delenv('NAPARI_GLIR_METERING', raising=False)
    parser = FakeParser()
    parser.add_texture3d(1)
    try:
        assert gm.install(
            frame_budget_bytes=512 * 2**10,
            slab_bytes=128 * 2**10,
        )
        queue = glir.GlirQueue()
        data = np.zeros((16, 256, 256), dtype=np.uint8)  # 1 MiB
        queue.command('DATA', 1, (0, 0, 0), data)
        queue.flush(parser)
        state = gm._states[parser]
        assert state.carry
        # a SIZE re-specification supersedes the carried uploads
        state.reset_budget()
        small = np.zeros((4, 4, 4), dtype=np.uint8)
        queue.command('SIZE', 1, (4, 4, 4), 'luminance', None)
        queue.command('DATA', 1, (0, 0, 0), small)
        queue.flush(parser)
        assert state.carry == []
        kinds = [c[0] for c in parser.executed]
        assert kinds[-2:] == ['SIZE', 'DATA']
        assert parser.executed[-1][3] is small
        # none of the stale 1 MiB payload was uploaded after the SIZE
        assert data_bytes(parser.executed) == 512 * 2**10 + small.nbytes
    finally:
        gm.uninstall()


def test_redundant_gl_state_calls_skipped(parser):
    state = make_state(budget=2**20, slab=2**20)
    commands = [
        ('FUNC', 'glEnable', 'blend'),
        ('FUNC', 'glBlendFunc', 'src_alpha', 'one_minus_src_alpha'),
        ('FUNC', 'glEnable', 'blend'),  # duplicate: skipped
        ('FUNC', 'glBlendFunc', 'src_alpha', 'one_minus_src_alpha'),  # dup
        ('FUNC', 'glEnable', 'depth_test'),  # different capability: kept
        ('FUNC', 'glDisable', 'blend'),  # state transition: kept
        ('FUNC', 'glEnable', 'blend'),  # transition back: kept
        ('FUNC', 'glBlendFunc', 'one', 'one'),  # changed args: kept
        ('FUNC', 'glClear', 17664),  # never cached
        ('FUNC', 'glClear', 17664),
    ]
    leftover = gm._metered_parse(parser, commands, state)
    assert leftover == []
    executed = [c[1:] for c in parser.executed]
    assert executed == [
        ('glEnable', 'blend'),
        ('glBlendFunc', 'src_alpha', 'one_minus_src_alpha'),
        ('glEnable', 'depth_test'),
        ('glDisable', 'blend'),
        ('glEnable', 'blend'),
        ('glBlendFunc', 'one', 'one'),
        ('glClear', 17664),
        ('glClear', 17664),
    ]
    # the cache persists across flushes (GL state is per-context)
    leftover = gm._metered_parse(
        parser, [('FUNC', 'glBlendFunc', 'one', 'one')], state
    )
    assert leftover == []
    assert len(parser.executed) == 8
