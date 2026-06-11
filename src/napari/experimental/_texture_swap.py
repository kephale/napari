"""Double-buffered texture streaming for vispy Volume and Image nodes.

Writing into a texture that the GPU is concurrently sampling is the
pathological path on macOS GL-over-Metal (and many other drivers): the
driver must either stall until in-flight frames finish or ghost-copy
the whole texture, and that cost lands inside the next ``draw()`` call
— multi-second stalls for a 32 MB volume, regardless of how small or
rare the individual ``glTexSubImage3D`` updates are. Upload pacing
alone cannot fix this; the rendered texture must never be written.

:class:`DoubleBufferedVolumeTexture` gives a vispy ``VolumeVisual`` two
textures with identical shape/format:

- the *front* texture is bound to the shader and is never modified;
- all chunk patches and full-volume rewrites are staged into the *back*
  texture (still rate-limited and GLIR-metered upstream);
- :meth:`present` swaps the sampler binding (one uniform rebind) and
  replays, from a patch log, whatever the new back texture missed while
  it was front.

Net cost: every staged byte is uploaded twice (once per texture, both
times to an unbound texture) and 2x texture GPU memory. Net win: draws
only ever sample stable textures.
"""

from __future__ import annotations

import contextlib
import logging
import os
import time

import numpy as np

LOGGER = logging.getLogger('napari.experimental._texture_swap')

#: Retired textures kept for reuse instead of deleted. GL object
#: deletion synchronizes with the GPU pipeline (profiled ~25ms per
#: DELETE on busy macOS GL-over-Metal), and reallocation costs another
#: sync — so zooming back and forth across two levels would otherwise
#: pay 4 syncs per switch. Bounded GPU memory cost: at most this many
#: spare tiles. Sized for the 16 MB default tile cap, whose quantized
#: shape vocabulary spans ~5 sizes (96..251^3) — up to 4 retired
#: front/back pairs alive at once, <= ~60 MB of spares.
#: NAPARI_PROGRESSIVE_TEXTURE_POOL overrides (0 disables).
DEFAULT_TEXTURE_POOL_SIZE = 8


class DoubleBufferedVolumeTexture:
    """Manage front/back 3D textures for a vispy ``VolumeVisual``.

    Parameters
    ----------
    node : vispy.visuals.VolumeVisual
        The volume node to manage. Its current texture becomes the
        front texture; a sibling back texture of the same class,
        format and interpolation is created for staging.

    """

    def __init__(self, node, pool: list | None = None):
        self._node = node
        self._front = node._texture
        # the shape the pair was built for: a later in-place resize of
        # the front (vispy reuses the texture object) must invalidate
        # the pair, so don't read front.shape live
        self._shape = tuple(self._front.shape[:3])
        # retired-texture pool: list of (key, texture), most recent
        # last; reused by _acquire instead of delete + reallocate.
        # Accepted from a predecessor pair so rebuilds reuse textures.
        self._pool: list[tuple] = pool if pool is not None else []
        env_pool = os.environ.get('NAPARI_PROGRESSIVE_TEXTURE_POOL')
        self._pool_max = (
            int(env_pool) if env_pool else DEFAULT_TEXTURE_POOL_SIZE
        )
        self._back = self._acquire(
            self._front.shape,
            getattr(self._front, '_data_dtype', None) or np.float32,
            lambda: self._make_sibling(node, self._front),
        )
        # patch log: list of (offset, data) staged since creation; each
        # texture tracks how much of the log it has applied. 'full'
        # entries reset the log (older patches are superseded).
        self._log: list[tuple] = []
        self._applied = {id(self._front): 0, id(self._back): 0}
        self._wrapped_set_data = None
        self._suppress_full = False
        # a staged shape change: the back texture already has the new
        # shape/content while the old-shape front keeps rendering until
        # the new texture's uploads have drained (or the deadline hits)
        self._reshape_pending = False
        self._reshape_deadline = 0.0

    # -- construction helpers --

    @staticmethod
    def _make_sibling(node, front):
        """Create a back texture matching the front's class and format."""
        from vispy.visuals._scalable_textures import GPUScaledTextureMixin

        if isinstance(front, GPUScaledTextureMixin):
            # the resolved format (e.g. 'r8'), not 'auto': the pair must
            # stay format-identical for patches to mean the same thing
            texture_format = front.internalformat
        else:  # pragma: no cover - napari uses GPU-scaled textures
            texture_format = None
        dtype = getattr(front, '_data_dtype', None) or np.float32
        rep = np.zeros((1, 1, 1), dtype=dtype)
        back = node._create_texture(texture_format, rep)
        # allocate at full size now (SIZE only, no pixel upload): offset
        # patches require an allocated texture
        back.resize(tuple(front.shape), internalformat=front.internalformat)
        back._data_dtype = dtype
        if front.clim is not None:
            back.set_clim(front.clim)
        if back.interpolation != front.interpolation:
            back.interpolation = front.interpolation
        return back

    def _make_texture_like(self, node, vol):
        """Create a texture formatted for ``vol`` (new shape and dtype)."""
        texture_format = getattr(node, 'texture_format', None)
        if texture_format is None:  # pragma: no cover - napari sets it
            texture_format = self._front.internalformat
        rep = np.zeros((1, 1, 1), dtype=vol.dtype)
        tex = node._create_texture(texture_format, rep)
        tex.resize(tuple(vol.shape[:3]))
        tex._data_dtype = vol.dtype
        return self._sync_aux_state(tex)

    def _sync_aux_state(self, tex):
        front = self._front
        if front.clim is not None:
            with contextlib.suppress(Exception):
                tex.set_clim(front.clim)
        if tex.interpolation != front.interpolation:
            tex.interpolation = front.interpolation
        return tex

    # -- retired-texture pool --

    @staticmethod
    def _pool_key(shape, dtype) -> tuple:
        return (tuple(shape)[:3], np.dtype(dtype).str)

    def _acquire(self, shape, dtype, create):
        """Reuse a retired texture of this shape/dtype, or create one."""
        key = self._pool_key(shape, dtype)
        for i in range(len(self._pool) - 1, -1, -1):
            if self._pool[i][0] == key:
                _, tex = self._pool.pop(i)
                return self._sync_aux_state(tex)
        return create()

    def _release(self, texture) -> None:
        """Retire a texture for reuse (delete only past the pool cap)."""
        dtype = getattr(texture, '_data_dtype', None)
        if self._pool_max <= 0 or dtype is None:
            with contextlib.suppress(Exception):
                texture.delete()
            return
        key = self._pool_key(texture.shape, dtype)
        self._pool.append((key, texture))
        while len(self._pool) > self._pool_max:
            _, old = self._pool.pop(0)
            with contextlib.suppress(Exception):
                old.delete()

    @property
    def shape(self) -> tuple:
        """The shape staged content must have (the pair's tile shape).

        During a pending reshape this is already the NEW shape — the
        old-shape front keeps rendering, but patches target the new
        back texture.
        """
        return self._shape

    def matches(self, node) -> bool:
        """Whether this buffer still belongs to ``node``'s texture pair."""
        if self._node is not node or node._texture not in (
            self._front,
            self._back,
        ):
            return False
        # a pending reshape legitimately renders the old-shape front;
        # otherwise an in-place resize of the bound texture (vispy
        # reuses the object) invalidates the pair
        return (
            self._reshape_pending
            or tuple(node._texture.shape[:3]) == self._shape
        )

    # -- staging --

    def stage(self, offset, data) -> None:
        """Stage a sub-region update; uploaded to the back texture now.

        ``data`` is retained until both textures have applied it (the
        log drains at each :meth:`present`).
        """
        self._log.append((tuple(offset), data, None))
        self._catch_up(self._back)

    def stage_full(self, data, clim=None) -> None:
        """Stage a full-volume rewrite (e.g. a pass-boundary backdrop)."""
        # a full write supersedes everything staged before it
        self._log = [(None, data, clim)]
        for key in self._applied:
            self._applied[key] = 0
        self._catch_up(self._back)

    def stage_reshape(self, vol, clim=None) -> None:
        """Stage a full rewrite at a NEW tile shape.

        A fresh texture is allocated and filled off the rendered path;
        the old-shape front keeps rendering its (valid, previous-level)
        content until :meth:`present` swaps — once the new texture's
        uploads have drained, or after a deadline. This removes the
        last write-to-bound-texture path: previously a shape change
        fell through to vispy's set_data, which re-specifies and
        re-uploads the texture the GPU is sampling.
        """
        node = self._node
        new_back = self._acquire(
            vol.shape,
            vol.dtype,
            lambda: self._make_texture_like(node, vol),
        )
        old_back = self._back
        if old_back is not self._front:
            self._release(old_back)
        self._back = new_back
        self._shape = tuple(vol.shape[:3])
        self._log = [(None, vol, clim)]
        self._applied = {id(self._front): 0, id(new_back): 0}
        self._reshape_pending = True
        self._reshape_deadline = time.monotonic() + 2.0
        self._catch_up(new_back)

    def _catch_up(self, texture) -> None:
        key = id(texture)
        start = self._applied[key]
        for offset, data, clim in self._log[start:]:
            if offset is None:
                # full rewrite, through the scaled-texture path so clim
                # normalization stays correct
                if clim is not None:
                    texture.set_clim(clim)
                texture.check_data_format(data)
                texture.scale_and_set_data(data)
            else:
                texture.set_data(data, offset=offset)
        self._applied[key] = len(self._log)

    # -- presentation --

    @property
    def dirty(self) -> bool:
        """Whether the front texture is behind the staged content."""
        return self._applied[id(self._front)] < len(self._log)

    def present(self) -> bool:
        """Swap the freshly written back texture into the shader.

        Returns True if a swap happened. The new back texture is caught
        up immediately afterwards (still off the rendered path) and the
        drained portion of the log is released.
        """
        if not self.dirty:
            return False
        if self._reshape_pending:
            return self._present_reshape()
        front, back = self._front, self._back
        # propagate authoritative front state (napari writes clims and
        # interpolation to node._texture, i.e. the front); clims staged
        # with a full rewrite override this inside _catch_up
        if front.clim is not None and back.clim != front.clim:
            back.set_clim(front.clim)
        if back.interpolation != front.interpolation:
            back.interpolation = front.interpolation
        self._catch_up(back)
        self._bind(back)
        self._front, self._back = back, front

        # catch the new back up too (off the rendered path), then drop
        # the fully-applied prefix of the log to release chunk memory
        self._catch_up(front)
        self._trim_log()
        return True

    def _present_reshape(self) -> bool:
        """Swap in the new-shape texture once its uploads have drained.

        Swapping earlier would render a partially uploaded (black)
        volume; until then the old-shape front keeps showing the
        previous level. A deadline bounds the wait in case uploads
        never fully settle (e.g. a steady chunk stream).
        """
        from napari.experimental import _glir_metering

        if (
            _glir_metering.is_installed()
            and _glir_metering.pending_upload_bytes() > 0
            and time.monotonic() < self._reshape_deadline
        ):
            return False
        node = self._node
        old_front, back = self._front, self._back
        self._catch_up(back)
        self._bind(back)
        z, y, x = self._shape
        node.shared_program['u_shape'] = (x, y, z)
        node._vol_shape = self._shape
        node._need_vertex_update = True
        self._front = back
        self._release(old_front)
        self._reshape_pending = False
        # rebuild the spare at the new shape and converge it (reusing a
        # retired same-shape texture when available)
        self._back = self._acquire(
            back.shape,
            getattr(back, '_data_dtype', np.float32),
            lambda: self._make_sibling(node, back),
        )
        self._applied = {
            id(self._front): self._applied[id(back)],
            id(self._back): 0,
        }
        self._catch_up(self._back)
        self._trim_log()
        node.update()
        return True

    def _bind(self, texture) -> None:
        """Point the shader at ``texture`` (one sampler rebind)."""
        node = self._node
        node.shared_program['u_volumetex'] = texture
        if getattr(node, '_data_lookup_fn', None) is not None:
            # interpolation lookup samples through this binding; when it
            # is None, vispy's lazy interpolation setup reads
            # node._texture at the next draw instead
            node._data_lookup_fn['texture'] = texture
        node.shared_program['clim'] = texture.clim_normalized
        node._texture = texture

    def _trim_log(self) -> None:
        applied_min = min(self._applied.values())
        if applied_min:
            self._log = self._log[applied_min:]
            for key in self._applied:
                self._applied[key] -= applied_min

    # -- full-refresh interception --

    def attach_set_data(self) -> None:
        """Route ``node.set_data`` calls through the staging texture.

        napari's slicing pipeline rewrites the whole volume through
        ``node.set_data`` at pass boundaries — a multi-second
        write-to-sampled-texture stall on slow drivers. Same-shape
        rewrites are staged into the back texture; shape changes
        (level/tile switches) are staged into a freshly allocated
        texture and swapped in once uploaded (:meth:`stage_reshape`).
        Only non-array payloads or external texture rebinds fall back
        to the original path.
        """
        if self._wrapped_set_data is not None:
            return
        node = self._node
        original = node.set_data

        def set_data_staged(vol, clim=None, copy=True):
            if (
                not isinstance(vol, np.ndarray)
                or vol.ndim != 3
                or node._texture not in (self._front, self._back)
            ):
                # unexpected payload or someone rebound the texture:
                # fall back; the loader rebuilds this buffer on its
                # next patch
                self._suppress_full = False
                self.detach_set_data()
                return original(vol, clim=clim, copy=copy)
            same_shape = tuple(vol.shape[:3]) == self.shape
            if same_shape and self._suppress_full:
                # caller asserts the GPU pair already matches vol
                # (every chunk was patched): skip the redundant
                # full-tile upload entirely
                self._suppress_full = False
                node._last_data = vol
                return None
            self._suppress_full = False
            try:
                if same_shape:
                    self.stage_full(vol, clim=clim)
                else:
                    # a level/tile switch: fill a new-shape texture off
                    # the rendered path; the swap happens once its
                    # uploads drain (the old level renders meanwhile)
                    self.stage_reshape(vol, clim=clim)
                self.present()
            except Exception:  # noqa: BLE001 - dtype/format change
                self.detach_set_data()
                return original(vol, clim=clim, copy=copy)
            node._last_data = vol
            return None

        node.set_data = set_data_staged
        self._wrapped_set_data = original

    def suppress_next_full_upload(self) -> None:
        """Skip the next same-shape full rewrite through ``set_data``.

        For when the caller knows the GPU pair already holds exactly
        the content the rewrite would upload (e.g. the deferred
        end-of-pass reconcile after a fully-patched pass). One-shot;
        cleared by the next ``set_data`` whether suppressed or not.
        """
        self._suppress_full = True

    def detach_set_data(self) -> None:
        if self._wrapped_set_data is not None:
            self._node.set_data = self._wrapped_set_data
            self._wrapped_set_data = None

    def close(self) -> None:
        """Restore the node and release the spare texture.

        The node keeps rendering whatever is currently front.
        """
        self.detach_set_data()
        self._log = []
        if self._back is not self._front:
            with contextlib.suppress(Exception):  # pragma: no cover
                self._back.delete()
        for _key, tex in self._pool:
            with contextlib.suppress(Exception):  # pragma: no cover
                tex.delete()
        self._pool = []


class DoubleBufferedImageTexture:
    """Manage front/back 2D textures for a vispy ``ImageVisual``.

    Same rationale as :class:`DoubleBufferedVolumeTexture`: chunk
    patches and full rewrites go to an unbound *back* texture and
    :meth:`present` swaps the sampler binding, so the texture the GPU is
    sampling is never written. One structural difference: a shape
    change (level/tile switch) binds its freshly filled texture
    *immediately* instead of waiting for uploads to drain — napari
    updates the node transform synchronously with ``set_data``, so
    keeping the old-shape front on screen would render the previous
    tile under the new tile's transform (misplaced content). 2D tiles
    are small enough that the staged-then-bound upload is the cheap
    part; what matters is that it never re-specifies a bound texture.

    Parameters
    ----------
    node : vispy.visuals.ImageVisual
        The image node to manage. Its current texture becomes the
        front texture; a sibling back texture of the same class,
        format and interpolation is created for staging.
    """

    def __init__(self, node, pool: list | None = None):
        self._node = node
        self._front = node._texture
        self._shape = tuple(self._front.shape[:2])
        self._pool: list[tuple] = pool if pool is not None else []
        env_pool = os.environ.get('NAPARI_PROGRESSIVE_TEXTURE_POOL')
        self._pool_max = (
            int(env_pool) if env_pool else DEFAULT_TEXTURE_POOL_SIZE
        )
        self._back = self._acquire(
            self._shape,
            getattr(self._front, '_data_dtype', None) or np.float32,
            lambda: self._make_sibling(node, self._front),
        )
        self._log: list[tuple] = []
        self._applied = {id(self._front): 0, id(self._back): 0}
        self._wrapped_set_data = None
        self._suppress_full = False
        # exempt the pair from upload metering: every write here targets
        # an unbound texture and becomes visible only through an atomic
        # swap — a deferred (carried) upload would present a partially
        # written texture as a black flash. The pair is stable (reshape
        # resizes in place), so the ids never change.
        self._exempt_ids = set()
        from napari.experimental import _glir_metering

        for tex in (self._front, self._back):
            glir_id = getattr(tex, 'id', None)
            if glir_id is not None:
                _glir_metering.add_unmetered_texture(glir_id)
                self._exempt_ids.add(glir_id)

    # -- construction helpers --

    @staticmethod
    def _make_sibling(node, front):
        """Create a back texture matching the front's class and format."""
        from vispy.visuals._scalable_textures import GPUScaledTextureMixin

        if isinstance(front, GPUScaledTextureMixin):
            texture_format = front.internalformat
        else:  # pragma: no cover - napari uses GPU-scaled textures
            texture_format = None
        dtype = getattr(front, '_data_dtype', None) or np.float32
        rep = np.zeros((1, 1), dtype=dtype)
        back = node._init_texture(rep, texture_format)
        back.resize(
            tuple(front.shape),
            internalformat=getattr(front, 'internalformat', None),
        )
        back._data_dtype = dtype
        if front.clim is not None:
            back.set_clim(front.clim)
        if back.interpolation != front.interpolation:
            back.interpolation = front.interpolation
        return back

    def _sync_aux_state(self, tex):
        front = self._front
        if front.clim is not None:
            with contextlib.suppress(Exception):
                tex.set_clim(front.clim)
        if tex.interpolation != front.interpolation:
            tex.interpolation = front.interpolation
        return tex

    # -- retired-texture pool (2-tuple keys: never collide with 3D) --

    @staticmethod
    def _pool_key(shape, dtype) -> tuple:
        return (tuple(shape)[:2], np.dtype(dtype).str)

    def _acquire(self, shape, dtype, create):
        key = self._pool_key(shape, dtype)
        for i in range(len(self._pool) - 1, -1, -1):
            if self._pool[i][0] == key:
                _, tex = self._pool.pop(i)
                return self._sync_aux_state(tex)
        return create()

    def _release(self, texture) -> None:
        dtype = getattr(texture, '_data_dtype', None)
        if self._pool_max <= 0 or dtype is None:
            with contextlib.suppress(Exception):
                texture.delete()
            return
        key = self._pool_key(texture.shape, dtype)
        self._pool.append((key, texture))
        while len(self._pool) > self._pool_max:
            _, old = self._pool.pop(0)
            with contextlib.suppress(Exception):
                old.delete()

    @property
    def shape(self) -> tuple:
        """The (rows, cols) shape staged content must have."""
        return self._shape

    def matches(self, node) -> bool:
        """Whether this buffer still belongs to ``node``'s texture pair."""
        return (
            self._node is node
            and node._texture in (self._front, self._back)
            and tuple(node._texture.shape[:2]) == self._shape
        )

    # -- staging --

    def stage(self, offset, data) -> None:
        """Stage a sub-region update; uploaded to the back texture now."""
        self._log.append((tuple(offset), data))
        self._catch_up(self._back)

    def stage_full(self, data) -> None:
        """Stage a full-image rewrite (e.g. a pass-boundary backdrop)."""
        self._log = [(None, data)]
        for key in self._applied:
            self._applied[key] = 0
        self._catch_up(self._back)

    def _catch_up(self, texture) -> None:
        key = id(texture)
        start = self._applied[key]
        for offset, data in self._log[start:]:
            if offset is None:
                texture.check_data_format(data)
                texture.scale_and_set_data(data)
            else:
                texture.set_data(data, offset=offset)
        self._applied[key] = len(self._log)

    # -- presentation --

    @property
    def dirty(self) -> bool:
        """Whether the front texture is behind the staged content."""
        return self._applied[id(self._front)] < len(self._log)

    def present(self) -> bool:
        """Swap the freshly written back texture into the shader."""
        if not self.dirty:
            return False
        front, back = self._front, self._back
        if front.clim is not None and back.clim != front.clim:
            back.set_clim(front.clim)
        if back.interpolation != front.interpolation:
            back.interpolation = front.interpolation
        self._catch_up(back)
        self._bind(back)
        self._front, self._back = back, front
        self._catch_up(front)
        self._trim_log()
        return True

    def _bind(self, texture) -> None:
        """Point the shader at ``texture`` (one sampler rebind)."""
        node = self._node
        if node._data_lookup_fn is not None:
            node._data_lookup_fn['texture'] = texture
        # keep the clim uniform consistent with the bound texture (the
        # color transform reads it at build time, not per draw); absent
        # until the first _build_color_transform — then the build reads
        # node._texture itself
        with contextlib.suppress(Exception):
            node.shared_program.frag['color_transform'][1]['clim'] = (
                texture.clim_normalized
            )
        node._texture = texture

    def _trim_log(self) -> None:
        applied_min = min(self._applied.values())
        if applied_min:
            self._log = self._log[applied_min:]
            for key in self._applied:
                self._applied[key] -= applied_min

    # -- full-refresh interception --

    def attach_set_data(self) -> None:
        """Route ``node.set_data`` calls through the staging textures.

        Same-shape rewrites are staged into the back texture and
        swapped in; shape changes (level/tile switches) re-spec the
        unbound back texture in place, fill it, and bind it
        immediately. Non-array or multichannel payloads, and dtype or
        format changes, fall back to the original path.
        """
        if self._wrapped_set_data is not None:
            return
        node = self._node
        original = node.set_data

        def set_data_staged(image, copy=False):
            data = np.asarray(image)
            if data.ndim != 2 or node._texture not in (
                self._front,
                self._back,
            ):
                # multichannel or externally rebound texture: fall
                # back; the loader rebuilds this buffer on its next
                # patch
                self._suppress_full = False
                self.detach_set_data()
                return original(image, copy=copy)
            same_shape = tuple(data.shape[:2]) == self._shape
            if same_shape and self._suppress_full:
                # caller asserts the GPU pair already matches the data
                # (every chunk was patched): skip the redundant
                # full-tile upload entirely
                self._suppress_full = False
                node._data = data
                return None
            self._suppress_full = False
            try:
                if same_shape:
                    self.stage_full(data)
                    self.present()
                else:
                    self._reshape_now(data)
            except Exception:  # noqa: BLE001 - dtype/format change
                self.detach_set_data()
                return original(image, copy=copy)
            node._data = data
            # we uploaded the content ourselves; vispy must not re-run
            # scale_and_set_data on the (now bound) front at next draw
            node._need_texture_upload = False
            return None

        node.set_data = set_data_staged
        self._wrapped_set_data = original

    def _reshape_now(self, data) -> None:
        """Re-spec the unbound back texture to the new shape and bind it.

        2D tile shapes follow the corner-pixels crop, which varies
        continuously with the camera — pooling by exact shape never
        hits (the macOS profile showed ~2 creates + 2 deletes per
        re-slice, and GL object deletion syncs the pipeline). Resizing
        the existing pair instead is a SIZE re-spec on textures that
        are unbound at the time, with no object churn at all.
        """
        node = self._node
        old_front, old_back = self._front, self._back
        dtype = getattr(old_front, '_data_dtype', None)
        if (
            old_back is old_front
            or dtype is None
            or np.dtype(data.dtype) != np.dtype(dtype)
        ):
            # dtype/format change (or degenerate pair): a resized
            # texture would no longer match the scaled format
            raise ValueError('texture pair cannot absorb this reshape')
        internalformat = getattr(old_back, 'internalformat', None)
        new_shape = tuple(data.shape[:2])
        # back is unbound: re-spec and fill it off the rendered path
        old_back.resize(new_shape, internalformat=internalformat)
        old_back.check_data_format(data)
        old_back.scale_and_set_data(data)
        self._bind(old_back)
        self._front, self._back = old_back, old_front
        self._shape = new_shape
        # the old front is unbound now: re-spec it for future staging
        # (allocation only, no pixel upload)
        old_front.resize(new_shape, internalformat=internalformat)
        # keep the full rewrite in the log: the new back has undefined
        # content after its re-spec and must replay it before the next
        # present() may swap it in (the front already applied it above)
        self._log = [(None, data)]
        self._applied = {id(self._front): 1, id(self._back): 0}
        # geometry follows the data shape
        node._need_vertex_update = True
        with contextlib.suppress(Exception):
            node.shared_program['image_size'] = data.shape[:2][::-1]
        if node._data_lookup_fn is not None:
            with contextlib.suppress(Exception):
                # kernel-based (e.g. cubic) lookups carry the texture
                # shape as a shader parameter
                if 'shape' in node._data_lookup_fn:
                    node._data_lookup_fn['shape'] = data.shape[:2][::-1]

    def suppress_next_full_upload(self) -> None:
        """Skip the next same-shape full rewrite through ``set_data``.

        One-shot; cleared by the next ``set_data`` whether suppressed
        or not.
        """
        self._suppress_full = True

    def detach_set_data(self) -> None:
        if self._wrapped_set_data is not None:
            self._node.set_data = self._wrapped_set_data
            self._wrapped_set_data = None

    def close(self) -> None:
        """Restore the node and release the spare texture."""
        from napari.experimental import _glir_metering

        for glir_id in self._exempt_ids:
            _glir_metering.discard_unmetered_texture(glir_id)
        self._exempt_ids = set()
        self.detach_set_data()
        self._log = []
        if self._back is not self._front:
            with contextlib.suppress(Exception):  # pragma: no cover
                self._back.delete()
        for _key, tex in self._pool:
            with contextlib.suppress(Exception):  # pragma: no cover
                tex.delete()
        self._pool = []
