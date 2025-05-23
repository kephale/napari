import os
import runpy
import sys
from pathlib import Path

import numpy as np
import pytest
import skimage.data
from qtpy import API_NAME

import napari
from napari._qt.qt_main_window import Window
from napari.utils.notifications import notification_manager

# check if this module has been explicitly requested or `--test-examples` is included
fpath = os.path.join(*__file__.split(os.path.sep)[-4:])
if '--test-examples' not in sys.argv and fpath not in sys.argv:
    pytest.skip(
        'Use `--test-examples` to test examples.', allow_module_level=True
    )

# not testing these examples
skip = [
    '3d_kymograph_.py',  # needs tqdm, omero-py and can take some time downloading data
    'live_tiffs_.py',  # requires files
    'tiled-rendering-2d_.py',  # too slow
    'live_tiffs_generator_.py',  # to generate files for live_tiffs_.py
    'points-over-time.py',  # too resource hungry
    'embed_ipython_.py',  # fails without monkeypatch
    'new_theme.py',  # testing theme is extremely slow on CI
    'dynamic-projections-dask.py',  # extremely slow / does not finish
]
# To skip examples during docs build end name with `_.py`

# these are more interactive tools than proper examples, so skip them
# cause they are hard to adapt for testing
skip_dev = ['leaking_check.py', 'demo_shape_creation.py']

EXAMPLE_DIR = Path(__file__).parent.parent.parent.parent / 'examples/'
DEV_EXAMPLE_DIR = Path(__file__).parent.parent.parent.parent / 'examples' / 'dev'
# using f.name here and re-joining at `run_path()` for test key presentation
# (works even if the examples list is empty, as opposed to using an ids lambda)
examples = [f.name for f in EXAMPLE_DIR.glob('*.py') if f.name not in skip]
dev_examples = [f.name for f in DEV_EXAMPLE_DIR.glob('*.py') if f.name not in skip_dev]


# still some CI segfaults, but only on windows with pyqt5
if os.getenv('CI') and os.name == 'nt' and API_NAME == 'PyQt5':
    examples = []

if os.getenv('CI') and os.name == 'nt' and 'to_screenshot.py' in examples:
    examples.remove('to_screenshot.py')

@pytest.fixture
def _example_monkeypatch(monkeypatch):
    # hide viewer window
    monkeypatch.setattr(Window, 'show', lambda *a: None)
    # prevent running the event loop
    monkeypatch.setattr(napari, 'run', lambda *a, **k: None)
    # Prevent downloading example data because this sometimes fails.
    monkeypatch.setattr(
        skimage.data,
        'cells3d',
        lambda: np.zeros((60, 2, 256, 256), dtype=np.uint16),
    )

    # make sure our sys.excepthook override doesn't hide errors
    def raise_errors(etype, value, tb):
        raise value

    monkeypatch.setattr(notification_manager, 'receive_error', raise_errors)


def _run_example(example_path):
    try:
        runpy.run_path(example_path)
    except SystemExit as e:
        # we use sys.exit(0) to gracefully exit from examples
        if e.code != 0:
            raise
    finally:
        napari.Viewer.close_all()


@pytest.mark.usefixtures('_example_monkeypatch')
@pytest.mark.filterwarnings('ignore')
@pytest.mark.skipif(not examples, reason='No examples were found.')
@pytest.mark.parametrize('fname', examples)
def test_examples(builtins, fname, monkeypatch):
    """Test that all of our examples are still working without warnings."""
    example_path = str(EXAMPLE_DIR / fname)
    monkeypatch.setattr(sys, 'argv', [fname])
    _run_example(example_path)


@pytest.mark.usefixtures('_example_monkeypatch')
@pytest.mark.filterwarnings('ignore')
@pytest.mark.skipif(not dev_examples, reason='No dev examples were found.')
@pytest.mark.parametrize('fname', dev_examples)
def test_dev_examples(fname, monkeypatch):
    """Test that all of our dev examples are still working without warnings."""
    example_path = str(DEV_EXAMPLE_DIR / fname)
    monkeypatch.setattr(sys, 'argv', [fname])
    _run_example(example_path)
