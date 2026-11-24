"""Dive into a generated Mandelbulb with N pyramid levels and record what arrives.

Usage:
    python mandelbulb_depth_probe.py LEVELS OUTDIR [--zoom 8] [--wait 120]
        [--overview 30] [--maxiter 64] [--no-content]

Opens the Mandelbulb with LEVELS levels in 3-D, waits ``--overview`` seconds,
then centers the camera on a point on the bulb surface at ``--zoom`` and waits
``--wait`` seconds. Writes LEVELS.json (samples, focus-box diagnostics), a log
and screenshots to OUTDIR. ``--no-content`` restores box-face anchoring of the
3-D focus segment for an A/B comparison. Needs numba and a real GPU: software
GL renders a frame in seconds and starves delivery.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("levels", type=int)
parser.add_argument("outdir", type=Path)
parser.add_argument("--maxiter", type=int, default=64)
parser.add_argument("--wait", type=float, default=45)
parser.add_argument("--zoom", type=float, default=8.0)
parser.add_argument("--no-content", action="store_true")
parser.add_argument("--overview", type=float, default=30)
parser.add_argument(
    "--target-level",
    type=int,
    default=0,
    help="pyramid level whose voxels should be about one screen pixel",
)
args = parser.parse_args()
args.outdir.mkdir(parents=True, exist_ok=True)

log_path = args.outdir / f"L{args.levels:02d}.log"
logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

from lodstone.datasets import mandelbulb_dataset  # noqa: E402
from lodstone.datasets.generative import mandelbulb  # noqa: E402
from qtpy.QtCore import QTimer  # noqa: E402

import napari  # noqa: E402
from napari.experimental._lodstone_loading import (  # noqa: E402
    add_lodstone_loading_image,
)

TILE = 32
LO, HI = -1.25, 1.25


def surface_point() -> np.ndarray:
    """A point just outside the bulb along a diagonal, in fractal coords."""
    direction = np.array([0.35, 0.55, 0.76])
    direction /= np.linalg.norm(direction)
    out = np.zeros(1, dtype=np.uint16)
    inside_r = 0.0
    outside_r = 1.25
    for _ in range(60):
        r = 0.5 * (inside_r + outside_r)
        x, y, z = direction * r
        mandelbulb(out, x, y, z, x + 1e-12, y + 1e-12, z + 1e-12, 1, args.maxiter, 8)
        if out[0] >= args.maxiter:
            inside_r = r
        else:
            outside_r = r
    return direction * outside_r


_warm = np.zeros(TILE**3, dtype=np.uint8)
_tw = time.perf_counter()
mandelbulb(_warm, -1.25, -1.25, -1.25, 1.25, 1.25, 1.25, TILE, args.maxiter, 8)
_tw2 = time.perf_counter()
mandelbulb(_warm, -1.25, -1.25, -1.25, 1.25, 1.25, 1.25, TILE, args.maxiter, 8)
chunk_s = time.perf_counter() - _tw2
compile_s = _tw2 - _tw - chunk_s

t0 = time.perf_counter()
dataset = mandelbulb_dataset(
    max_levels=args.levels,
    tilesize=TILE,
    maxiter=args.maxiter,
    cache_bytes=2_000_000_000,
)
arrays = dataset["arrays"]
t_open = time.perf_counter() - t0
width0 = TILE * 2**args.levels

point_xyz = surface_point()
# data coords (z, y, x) at level 0
point_zyx = ((point_xyz[::-1] - LO) / (HI - LO)) * width0

viewer = napari.Viewer(show=True)
viewer.window._qt_window.resize(1000, 800)
t1 = time.perf_counter()
layer = add_lodstone_loading_image(
    arrays,
    viewer=viewer,
    contrast_limits=(0, args.maxiter),
    colormap="magma",
    rendering="attenuated_mip",
)
t_add = time.perf_counter() - t1
viewer.dims.ndisplay = 3
viewer.reset_view()
loader = layer.metadata.get("progressive_loader")
if args.no_content:
    loader._first_content_along_ray = lambda *a, **k: None

result = {
    "levels": args.levels,
    "level0_width": width0,
    "open_s": round(t_open, 3),
    "add_layer_s": round(t_add, 3),
    "surface_point_xyz": point_xyz.tolist(),
    "target_level": args.target_level,
    "content_anchoring": not args.no_content,
    "numba_compile_s": round(compile_s, 2),
    "chunk_compute_s": round(chunk_s, 4),
    "float32_spacing_at_point": float(np.spacing(np.float32(point_zyx.max()))),
}
samples = []
t_dive = None


def sample(tag):
    entry = {
        "tag": tag,
        "t": round(time.perf_counter() - t1, 2),
        "data_level": int(layer.data_level),
        "zoom": float(viewer.scene.camera.zoom),
    }
    try:
        records = loader.performance_records if loader is not None else ()
        if records:
            last = records[-1]
            entry["last_record"] = {
                k: last[k]
                for k in list(last)[:40]
                if isinstance(last[k], (int, float, str, bool, type(None)))
            }
            entry["records"] = len(records)
    except Exception as exc:  # noqa: BLE001
        entry["records_error"] = repr(exc)
    samples.append(entry)


def dive():
    global t_dive
    sample("before_dive")
    viewer.screenshot(
        str(args.outdir / f"L{args.levels:02d}-overview.png"), canvas_only=True
    )
    # napari may rescale very large layers to stay within float32
    # precision, so convert the data-space point to world space
    world = np.asarray(layer.data_to_world(point_zyx), dtype=float)
    viewer.scene.camera.center = tuple(world)
    # napari 3-D zoom is screen pixels per world unit
    viewer.scene.camera.zoom = args.zoom
    viewer.scene.camera.angles = (0, -30, 60)
    t_dive = time.perf_counter()
    QTimer.singleShot(5000, lambda: diagnose("dive+5s"))


def tick():
    sample("tick")


def diagnose(tag):
    displayed = list(layer._slice_input.displayed)
    bbox = loader._camera_bbox_level0(displayed)
    cam = np.asarray(viewer.scene.camera.center, float)
    entry = {
        "tag": tag,
        "camera_center": cam.tolist(),
        "zoom": float(viewer.scene.camera.zoom),
    }
    if bbox is not None:
        mid = bbox.mean(axis=0)
        entry["bbox_center"] = mid.tolist()
        entry["bbox_size"] = (bbox[1] - bbox[0]).tolist()
        entry["bbox_center_to_surface_point"] = float(np.linalg.norm(mid - point_zyx))
    entry["surface_point_data"] = point_zyx.tolist()
    entry["layer_scale"] = np.asarray(layer.scale).tolist()
    try:
        entry["zoom_target_level"] = int(loader._zoom_target_level_3d())
    except Exception as exc:  # noqa: BLE001
        entry["zoom_target_error"] = repr(exc)
    entry["data_level"] = int(layer.data_level)
    entry["corner_pixels"] = np.asarray(layer.corner_pixels).tolist()
    result.setdefault("diagnostics", []).append(entry)


def finish():
    diagnose("final")
    sample("final")
    shot = args.outdir / f"L{args.levels:02d}.png"
    viewer.screenshot(str(shot), canvas_only=True)
    result["samples"] = samples
    levels_seen = sorted({s["data_level"] for s in samples})
    result["levels_seen"] = levels_seen
    result["final_level"] = samples[-1]["data_level"]
    (args.outdir / f"L{args.levels:02d}.json").write_text(
        json.dumps(result, indent=1, default=str)
    )
    if loader is not None:
        loader.close()
    QTimer.singleShot(200, viewer.close)


timer = QTimer()
timer.timeout.connect(tick)
timer.start(3000)
QTimer.singleShot(int(args.overview * 1000), dive)
QTimer.singleShot(int((args.overview + args.wait) * 1000), finish)
napari.run()
print(
    json.dumps(
        {
            k: result.get(k)
            for k in (
                "levels",
                "open_s",
                "add_layer_s",
                "levels_seen",
                "final_level",
                "float32_spacing_at_point",
            )
        }
    )
)
sys.exit(0)
