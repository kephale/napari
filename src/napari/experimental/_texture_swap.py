"""Double-buffered 3D texture streaming for vispy Volume nodes.

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

import logging

import numpy as np

LOGGER = logging.getLogger('napari.experimental._texture_swap')


class DoubleBufferedVolumeTexture:
    """Manage front/back 3D textures for a vispy ``VolumeVisual``.

    Parameters
    ----------
    node : vispy.visuals.VolumeVisual
        The volume node to manage. Its current texture becomes the
        front texture; a sibling back texture of the same class,
        format and interpolation is created for staging.
    """

    def __init__(self, node):
        self._node = node
        self._front = node._texture
        # the shape the pair was built for: a later in-place resize of
        # the front (vispy reuses the texture object) must invalidate
        # the pair, so don't read front.shape live
        self._shape = tuple(self._front.shape[:3])
        self._back = self._make_sibling(node, self._front)
        # patch log: list of (offset, data) staged since creation; each
        # texture tracks how much of the log it has applied. 'full'
        # entries reset the log (older patches are superseded).
        self._log: list[tuple] = []
        self._applied = {id(self._front): 0, id(self._back): 0}
        self._full_pending: dict[int, bool] = {
            id(self._front): False,
            id(self._back): False,
        }
        self._wrapped_set_data = None
        self._suppress_full = False

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

    @property
    def shape(self) -> tuple:
        return self._shape

    def matches(self, node) -> bool:
        """Whether this buffer still belongs to ``node``'s texture pair."""
        return (
            self._node is node
            and node._texture in (self._front, self._back)
            and tuple(node._texture.shape[:3]) == self._shape
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
        node = self._node
        front, back = self._front, self._back
        # propagate authoritative front state (napari writes clims and
        # interpolation to node._texture, i.e. the front); clims staged
        # with a full rewrite override this inside _catch_up
        if front.clim is not None and back.clim != front.clim:
            back.set_clim(front.clim)
        if back.interpolation != front.interpolation:
            back.interpolation = front.interpolation
        self._catch_up(back)

        # the swap: one sampler rebind plus bookkeeping uniforms
        node.shared_program['u_volumetex'] = back
        if getattr(node, '_data_lookup_fn', None) is not None:
            # interpolation lookup samples through this binding; when it
            # is None, vispy's lazy interpolation setup reads
            # node._texture at the next draw instead
            node._data_lookup_fn['texture'] = back
        node.shared_program['clim'] = back.clim_normalized
        node._texture = back
        self._front, self._back = back, front

        # catch the new back up too (off the rendered path), then drop
        # the fully-applied prefix of the log to release chunk memory
        self._catch_up(front)
        applied_min = min(self._applied.values())
        if applied_min:
            self._log = self._log[applied_min:]
            for key in self._applied:
                self._applied[key] -= applied_min
        return True

    # -- full-refresh interception --

    def attach_set_data(self) -> None:
        """Route same-shape ``node.set_data`` calls through the staging
        texture.

        napari's slicing pipeline rewrites the whole volume through
        ``node.set_data`` at pass boundaries — a multi-second
        write-to-sampled-texture stall on slow drivers. Same-shape
        rewrites are staged and presented instead; shape changes fall
        back to the original path (textures must be re-specified) and
        the loader rebuilds this buffer afterwards.
        """
        if self._wrapped_set_data is not None:
            return
        node = self._node
        original = node.set_data

        def set_data_staged(vol, clim=None, copy=True):
            same_shape = (
                isinstance(vol, np.ndarray)
                and vol.ndim == 3
                and tuple(vol.shape[:3]) == self.shape
            )
            if not same_shape or node._texture not in (
                self._front,
                self._back,
            ):
                # shape/format change: textures must be re-specified
                # through the original path; the loader rebuilds this
                # buffer on its next patch
                self._suppress_full = False
                self.detach_set_data()
                return original(vol, clim=clim, copy=copy)
            if self._suppress_full:
                # caller asserts the GPU pair already matches vol
                # (every chunk was patched): skip the redundant
                # full-tile upload entirely
                self._suppress_full = False
                node._last_data = vol
                return None
            try:
                self.stage_full(vol, clim=clim)
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
        import contextlib

        self.detach_set_data()
        self._log = []
        with contextlib.suppress(Exception):  # pragma: no cover - teardown
            self._back.delete()
