"""Shared fixtures for the progressive loading tests."""

import pytest

try:
    from napari.experimental import _progressive_loading
except ModuleNotFoundError as error:
    if error.name != 'lodstone':
        raise
    _progressive_loading = None


@pytest.fixture(autouse=True)
def _no_progress_bars(monkeypatch):
    """Suppress napari's Qt progress bars for the whole directory.

    The activity-dock progress bar runs ``processEvents()`` on every
    update. That nested event processing wedges Qt timer dispatch in
    headless pytest runs on macOS: the suite stops making progress and
    the CI job is killed at its step timeout. Progress-bar cosmetics are
    not what these tests cover, and the deferred-update behavior that
    does matter is tested with an injected fake bar (see
    ``test_progress_updates_deferred``).
    """
    if _progressive_loading is not None:
        monkeypatch.setattr(
            _progressive_loading.ProgressiveLoader,
            '_make_progress',
            lambda self, total, description: None,
        )
