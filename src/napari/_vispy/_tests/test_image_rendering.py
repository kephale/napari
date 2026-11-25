import numpy as np
import pytest

from napari._tests.utils import skip_on_win_ci
from napari._vispy.layers.image import VispyImageLayer
from napari._vispy.utils.qt_font import FontInfo
from napari.components.dims import Dims
from napari.layers.image import Image


def test_image_rendering(make_napari_viewer):
    """Test 3D image with different rendering."""
    viewer = make_napari_viewer()

    viewer.dims.ndisplay = 3

    data = np.random.random((20, 20, 20))
    layer = viewer.add_image(data)
    vispy_layer = viewer.window._qt_viewer.layer_to_visual[layer]

    assert layer.rendering == 'mip'

    assert layer.interpolation2d == 'nearest'
    assert layer.interpolation3d == 'linear'

    # Change rendering property
    layer.rendering = 'translucent'
    assert layer.rendering == 'translucent'

    # Change rendering property
    layer.rendering = 'attenuated_mip'
    assert layer.rendering == 'attenuated_mip'
    layer.attenuation = 0.15
    assert layer.attenuation == 0.15

    # Change rendering property
    layer.rendering = 'iso'
    assert layer.rendering == 'iso'
    layer.iso_threshold = 0.3
    assert layer.iso_threshold == 0.3

    # Change rendering property
    layer.rendering = 'additive'
    assert layer.rendering == 'additive'

    # check custom interpolation works on the 2D node
    with pytest.raises(NotImplementedError):
        layer.interpolation3d = 'custom'
    viewer.dims.ndisplay = 2
    layer.interpolation2d = 'custom'
    assert vispy_layer.node.interpolation == 'custom'


@pytest.mark.parametrize(
    ('dtype', 'expected'),
    [(np.uint8, 32 / 255), (np.float32, 32.0)],
)
def test_iso_threshold_matches_the_texture_values(
    make_napari_viewer, dtype, expected
):
    """The iso shader compares the threshold with raw texture values. A
    uint8 texture stores value / 255, so a threshold normalized by a
    narrower contrast range (here 0-64) lies above every voxel and the
    surface never draws."""
    viewer = make_napari_viewer()
    viewer.dims.ndisplay = 3
    data = np.zeros((16, 16, 16), dtype=dtype)
    data[4:12, 4:12, 4:12] = 64
    layer = viewer.add_image(
        data, rendering='iso', contrast_limits=(0, 64), iso_threshold=32
    )
    node = viewer.window._qt_viewer.layer_to_visual[layer].node

    assert node.threshold == pytest.approx(expected)
    assert node.threshold < node._texture.clim_normalized[1]


def test_volume_clipmap_renders_overview_outside_detail(
    qapp,
    make_napari_viewer,
):
    """A blank high-resolution crop must not hide coarse context."""
    viewer = make_napari_viewer(show=True)
    viewer.dims.ndisplay = 3
    layer = viewer.add_image(
        np.zeros((16, 16, 16), dtype=np.uint8),
        contrast_limits=(0, 1),
        rendering='mip',
    )
    visual = viewer.window._qt_viewer.layer_to_visual[layer]
    node = visual.node
    node.enable_clipmap(
        np.ones((16, 16, 16), dtype=np.uint8),
        full_shape=(64, 64, 64),
    )
    node.set_clipmap_detail_bounds((24, 24, 24), (40, 40, 40))
    # Vispy rebuilds this lookup lazily when interpolation changes. The
    # rebuild must retain the movable detail box instead of resetting it to
    # the full volume and hiding the overview.
    node._build_interpolation()
    visual._on_matrix_change()
    viewer.reset_view()
    qapp.processEvents()

    screenshot = viewer.screenshot(canvas_only=True, flash=False)

    assert np.count_nonzero(screenshot[..., :3]) > 100


def test_clipmap_steps_follow_each_texture_resolution(make_napari_viewer):
    """A clipmap ray box is the full-resolution level, so one vispy step
    sized for it is far finer than the overview needs and, for deep
    pyramids, far coarser than the detail texture. Steps are instead the
    requested step in overview voxels outside the detail box and in detail
    voxels inside it."""
    viewer = make_napari_viewer()
    viewer.dims.ndisplay = 3
    layer = viewer.add_image(np.zeros((16, 16, 16), dtype=np.uint8))
    node = viewer.window._qt_viewer.layer_to_visual[layer].node
    node.relative_step_size = 0.5
    node.enable_clipmap(
        np.zeros((16, 16, 16), dtype=np.uint8), full_shape=(64, 64, 64)
    )
    # a 16^3 detail texture spanning 16 voxels: one full-resolution voxel
    node.set_clipmap_detail_bounds((24, 24, 24), (40, 40, 40))

    program = node.shared_program
    assert program['u_clipmap_adaptive'] == 1.0
    # overview voxels are 4 full-resolution voxels wide
    assert program['u_relative_step_size'] == pytest.approx(2.0)
    assert program['u_fine_fraction'] == pytest.approx(0.25)
    # each sample spans the requested step in its own texture's voxels
    assert program['u_coarse_weight'] == pytest.approx(0.5)
    # surface gradients sample one texel of the texture being read
    np.testing.assert_allclose(program['u_overview_voxel'], [1 / 16] * 3)
    np.testing.assert_allclose(program['u_detail_voxel'], [0.25 / 16] * 3)
    # relative to the floating origin at the detail box's low corner
    np.testing.assert_allclose(node._origin, [24, 24, 24])
    np.testing.assert_allclose(program['u_detail_min'], [0.0] * 3)
    np.testing.assert_allclose(program['u_detail_max'], [0.25] * 3)
    assert node.relative_step_size == 0.5

    # interaction coarsens both steps together
    node.relative_step_size = 2.0
    assert program['u_relative_step_size'] == pytest.approx(8.0)
    assert program['u_fine_fraction'] == pytest.approx(0.25)

    node.disable_clipmap()
    assert program['u_clipmap_adaptive'] == 0.0
    assert program['u_relative_step_size'] == pytest.approx(2.0)


def test_clipmap_floating_origin_composes_the_same_transforms(
    qapp, make_napari_viewer
):
    """Clipmap volumes draw relative to an origin at the detail box, with
    transforms composed in float64; they must map points exactly as the
    vispy chains do from absolute coordinates."""
    from napari._vispy.visuals.volume import _chain_matrix

    viewer = make_napari_viewer(show=True)
    viewer.dims.ndisplay = 3
    layer = viewer.add_image(
        np.zeros((16, 16, 16), dtype=np.uint8), scale=(2, 3, 4)
    )
    viewer.scene.camera.angles = (10, 20, 30)
    visual = viewer.window._qt_viewer.layer_to_visual[layer]
    node = visual.node
    node.enable_clipmap(
        np.zeros((16, 16, 16), dtype=np.uint8), full_shape=(64, 64, 64)
    )
    node.set_clipmap_detail_bounds((24, 20, 8), (40, 36, 24))
    visual._on_matrix_change()
    viewer.reset_view()
    qapp.processEvents()
    viewer.screenshot(canvas_only=True, flash=False)

    assert node._origin_active
    origin = node._origin
    np.testing.assert_array_equal(origin, [8, 20, 24])  # xyz
    points = np.array([[0.0, 0.0, 0.0], [3.5, -2.0, 7.0], [-30.0, 12.0, 1.0]])
    for name, target in (('render', 'render'), ('document', 'document')):
        chain = node.transforms.get_transform('visual', target)
        expected = chain.map(points + origin)
        composed = node._origin_transforms[name].map(points)
        np.testing.assert_allclose(
            composed[:, :3] / composed[:, 3:],
            expected[:, :3] / expected[:, 3:],
            rtol=1e-5,
            atol=1e-5,
        )
    inverse = node._origin_transforms['document_inv'].map(
        node._origin_transforms['document'].map(points)
    )
    np.testing.assert_allclose(inverse[:, :3] / inverse[:, 3:], points, atol=1e-4)
    assert _chain_matrix(node.transforms.get_transform('visual', 'render')) is not None

    # the ray box is drawn relative to the origin
    assert node._compute_bounds(0, None) == (-8, 56)
    node.disable_clipmap()
    assert not node._origin_active
    assert node._compute_bounds(0, None)[0] == 0  # no origin offset


@skip_on_win_ci
def test_visibility_consistency(qapp, make_napari_viewer):
    """Make sure toggling visibility maintains image contrast.

    see #1622 for details.
    """
    viewer = make_napari_viewer(show=True)

    layer = viewer.add_image(
        np.random.random((200, 200)), contrast_limits=[0, 10]
    )
    qapp.processEvents()
    layer.contrast_limits = (0, 2)
    screen1 = viewer.screenshot(flash=False).astype('float')
    layer.visible = True
    screen2 = viewer.screenshot(flash=False).astype('float')
    assert np.max(np.abs(screen2 - screen1)) < 5


def test_clipping_planes_dims():
    """
    Ensure that dims are correctly set on clipping planes
    (vispy uses xyz, napary zyx)
    """
    clipping_planes = {
        'position': (1, 2, 3),
        'normal': (1, 2, 3),
    }
    image_layer = Image(
        np.zeros((2, 2, 2)), experimental_clipping_planes=clipping_planes
    )
    vispy_layer = VispyImageLayer(image_layer, font_info=FontInfo())
    napari_clip = image_layer.experimental_clipping_planes.as_array()
    # needed to get volume node
    image_layer._slice_dims(Dims(ndim=3, ndisplay=3))
    vispy_clip = vispy_layer.node.clipping_planes
    np.testing.assert_array_equal(napari_clip, vispy_clip[..., ::-1])
