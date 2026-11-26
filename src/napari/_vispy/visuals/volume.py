from collections.abc import Sequence
from typing import Any

import numpy as np
from vispy.scene.visuals import Volume as BaseVolume
from vispy.visuals.shaders import Function
from vispy.visuals.transforms import (
    ChainTransform,
    MatrixTransform,
    NullTransform,
    STTransform,
)
from vispy.visuals.transforms.base_transform import InverseTransform

from napari._vispy.visuals.util import TextureMixin
from napari.layers.labels._labels_constants import IsoCategoricalGradientMode

FUNCTION_DEFINITIONS = """
// switch for clamping values at volume limits
uniform bool u_clamp_at_border;

// the tolerance for testing equality of floats with floatEqual and floatNotEqual
const float equality_tolerance = 1e-8;

bool floatNotEqual(float val1, float val2)
{
    // check if val1 and val2 are not equal
    bool not_equal = abs(val1 - val2) > equality_tolerance;

    return not_equal;
}

bool floatEqual(float val1, float val2)
{
    // check if val1 and val2 are equal
    bool equal = abs(val1 - val2) < equality_tolerance;

    return equal;
}

// the background value for the iso_categorical shader
const float categorical_bg_value = 0;

int detectAdjacentBackground(float val_neg, float val_pos)
{
    // determine if the adjacent voxels along an axis are both background
    int adjacent_bg = int( floatEqual(val_neg, categorical_bg_value) );
    adjacent_bg = adjacent_bg * int( floatEqual(val_pos, categorical_bg_value) );
    return adjacent_bg;
}
"""

CALCULATE_COLOR_DEFINITION = """
vec4 calculateShadedCategoricalColor(vec4 betterColor, vec3 loc, vec3 step)
{
    // Calculate color by incorporating ambient and diffuse lighting
    vec4 color0 = $get_data(loc);
    vec4 color1;
    vec4 color2;
    float val0 = colorToVal(color0);
    float val1 = 0;
    float val2 = 0;

    // View direction
    vec3 V = normalize(view_ray);

    // Calculate normal vector from gradient
    vec3 N;
    N = calculateGradient(loc, step, val0);

    // Normalize and flip normal so it points towards viewer
    N = normalize(N);
    float Nselect = float(dot(N,V) > 0.0);
    N = (2.0*Nselect - 1.0) * N;  // ==  Nselect * N - (1.0-Nselect)*N;

    // Init colors
    vec4 ambient_color = vec4(0.0, 0.0, 0.0, 0.0);
    vec4 diffuse_color = vec4(0.0, 0.0, 0.0, 0.0);
    vec4 final_color;

    // todo: allow multiple light, define lights on viewvox or subscene
    int nlights = 1;
    for (int i=0; i<nlights; i++)
    {
        // Get light direction (make sure to prevent zero devision)
        vec3 L = normalize(view_ray);  //lightDirs[i];
        float lightEnabled = float( length(L) > 0.0 );
        L = normalize(L+(1.0-lightEnabled));

        // Calculate lighting properties
        float lambertTerm = clamp( dot(N,L), 0.0, 1.0 );

        // Calculate mask
        float mask1 = lightEnabled;

        // Calculate colors
        ambient_color +=  mask1 * u_ambient;  // * gl_LightSource[i].ambient;
        diffuse_color +=  mask1 * lambertTerm;
    }

    // Calculate final color by componing different components
    final_color = betterColor * ( ambient_color + diffuse_color);
    final_color.a = betterColor.a;

    // Done
    return final_color;
}
"""

FAST_GRADIENT_DEFINITION = """
vec3 calculateGradient(vec3 loc, vec3 step, float current_val) {
    // calculate gradient within the volume by finite differences

    vec3 G = vec3(0.0);

    float prev;
    float next;
    int in_bounds;

    for (int i=0; i<3; i++) {
        vec3 ax_step = vec3(0.0);
        ax_step[i] = step[i];

        vec3 prev_loc = loc - ax_step;
        if (u_clamp_at_border || (prev_loc[i] >= 0.0 && prev_loc[i] <= 1.0)) {
            prev = colorToVal($get_data(prev_loc));
        } else {
            prev = categorical_bg_value;
        }

        vec3 next_loc = loc + ax_step;
        if (u_clamp_at_border || (next_loc[i] >= 0.0 && next_loc[i] <= 1.0)) {
            next = colorToVal($get_data(next_loc));
        } else {
            next = categorical_bg_value;
        }

        // add to the gradient where the adjacent voxels are both background
        // to fix dim pixels due to poor normal estimation
        G[i] = next - prev + (next - current_val) * 2.0 * detectAdjacentBackground(prev, next);
    }

    return G;
}
"""

SMOOTH_GRADIENT_DEFINITION = """
vec3 calculateGradient(vec3 loc, vec3 step, float current_val) {
    // calculate gradient within the volume by finite differences
    // using a 3D sobel-feldman convolution kernel

    // the kernel here is a 3x3 cube, centered on the sample at `loc`
    // the kernel for G.z looks like this:

    // [ +1 +2 +1 ]
    // [ +2 +4 +2 ]    <-- "loc - step.z" is in the center
    // [ +1 +2 +1 ]

    // [  0  0  0 ]
    // [  0  0  0 ]    <-- "loc" is in the center
    // [  0  0  0 ]

    // [ -1 -2 -1 ]
    // [ -2 -4 -2 ]    <-- "loc + step.z" is in the center
    // [ -1 -2 -1 ]

    // kernels for G.x and G.y similar, but transposed
    // see https://en.wikipedia.org/wiki/Sobel_operator#Extension_to_other_dimensions

    vec3 G = vec3(0.0);
    // next and prev are the directly adjacent values along x, y, and z
    vec3 next = vec3(0.0);
    vec3 prev = vec3(0.0);

    float val;
    bool is_on_border = false;
    for (int i=-1; i <= 1; i++) {
        for (int j=-1; j <= 1; j++) {
            for (int k=-1; k <= 1; k++) {
                if (is_on_border && (i != 0 && j != 0 && k != 0)) {
                    // we only care about on-axis values if we are on a border
                    continue;
                }
                vec3 sample_loc = loc + vec3(i, j, k) * step;
                bool is_in_bounds = all(greaterThanEqual(sample_loc, vec3(0.0)))
                    && all(lessThanEqual(sample_loc, vec3(1.0)));

                if (is_in_bounds || u_clamp_at_border) {
                    val = colorToVal($get_data(sample_loc));
                } else {
                    val = categorical_bg_value;
                }

                G.x += val * -float(i) *
                    (1 + float(j == 0 || k == 0) + 2 * float(j == 0 && k == 0));
                G.y += val * -float(j) *
                    (1 + float(i == 0 || k == 0) + 2 * float(i == 0 && k == 0));
                G.z += val * -float(k) *
                    (1 + float(i == 0 || j == 0) + 2 * float(i == 0 && j == 0));

                next.x += int(i == 1 && j == 0 && k == 0) * val;
                next.y += int(i == 0 && j == 1 && k == 0) * val;
                next.z += int(i == 0 && j == 0 && k == 1) * val;
                prev.x += int(i == -1 && j == 0 && k == 0) * val;
                prev.y += int(i == 0 && j == -1 && k == 0) * val;
                prev.z += int(i == 0 && j == 0 && k == -1) * val;

                is_on_border = is_on_border || (!is_in_bounds && (i == 0 || j == 0 || k == 0));
            }
        }
    }

    if (is_on_border && u_clamp_at_border) {
        // fallback to simple gradient calculation if we are on the border
        // and clamping is enabled (old behavior with dark/hollow faces at the border)
        // this makes the faces in `fast` and `smooth` look the same in both clamping modes
        G = next - prev;
    } else {
        // add to the gradient where the adjacent voxels are both background
        // to fix dim pixels due to poor normal estimation
        G.x = G.x + (next.x - current_val) * 2.0 * detectAdjacentBackground(prev.x, next.x);
        G.y = G.y + (next.y - current_val) * 2.0 * detectAdjacentBackground(prev.y, next.y);
        G.z = G.z + (next.z - current_val) * 2.0 * detectAdjacentBackground(prev.z, next.z);
    }

    return G;
}
"""

ISO_CATEGORICAL_SNIPPETS = {
    'before_loop': """
        vec4 color3 = vec4(0.0);  // final color
        vec3 dstep = 1.5 / u_shape;  // step to sample derivative, set to match iso shader
        gl_FragColor = vec4(0.0);
        bool discard_fragment = true;
        vec4 label_id = vec4(0.0);
        """,
    'in_loop': """
        // check if value is different from the background value
        if ( floatNotEqual(val, categorical_bg_value) ) {
            // Take the last interval in smaller steps
            vec3 iloc = loc - step;
            for (int i=0; i<10; i++) {
                label_id = $get_data(iloc);
                color = sample_label_color(label_id.r);
                if (floatNotEqual(color.a, 0) ) {
                    // fully transparent color is considered as background, see napari/napari#5227
                    // when the value mapped to non-transparent color is reached
                    // calculate the shaded color (apply lighting effects)
                    color = calculateShadedCategoricalColor(color, iloc, dstep);
                    gl_FragColor = color;

                    // set the variables for the depth buffer
                    frag_depth_point = iloc * u_shape;
                    discard_fragment = false;

                    iter = nsteps;
                    break;
                }
                iloc += step * 0.1;
            }
        }
        """,
    'after_loop': """
        if (discard_fragment)
            discard;
        """,
}

TRANSLUCENT_CATEGORICAL_SNIPPETS = {
    'before_loop': """
        vec4 color3 = vec4(0.0);  // final color
        gl_FragColor = vec4(0.0);
        bool discard_fragment = true;
        vec4 label_id = vec4(0.0);
        """,
    'in_loop': """
        // check if value is different from the background value
        if ( floatNotEqual(val, categorical_bg_value) ) {
            // Take the last interval in smaller steps
            vec3 iloc = loc - step;
            for (int i=0; i<10; i++) {
                label_id = $get_data(iloc);
                color = sample_label_color(label_id.r);
                if (floatNotEqual(color.a, 0) ) {
                    // fully transparent color is considered as background, see napari/napari#5227
                    // when the value mapped to non-transparent color is reached
                    // calculate the color (apply lighting effects)
                    gl_FragColor = color;

                    // set the variables for the depth buffer
                    frag_depth_point = iloc * u_shape;
                    discard_fragment = false;

                    iter = nsteps;
                    break;
                }
                iloc += step * 0.1;
            }
        }
        """,
    'after_loop': """
        if (discard_fragment)
            discard;
        """,
}


def _patch(source: str, old: str, new: str) -> str:
    """Replace ``old``, which must occur exactly once, in shader text."""
    if source.count(old) != 1:
        raise RuntimeError(
            f'vispy volume shader changed: expected one {old!r}; the '
            'clipmap ray stepping in napari/_vispy/visuals/volume.py needs '
            'updating'
        )
    return source.replace(old, new)


# Clipmap ray stepping.
#
# A clipmap volume ray-marches a box the size of the full-resolution level
# (``u_shape``), sampling a coarse overview texture everywhere except inside
# a movable detail box. vispy marches with one fixed step, and bounds it
# below by ``side / 15360`` voxels of that box to keep the iteration count
# sane. For a deep pyramid that step is tens or hundreds of full-resolution
# voxels: far coarser than the detail texture, so thin structures fall
# between samples (striped or empty frames), while still finer than the
# overview needs.
#
# With ``u_clipmap_adaptive`` set the march instead takes overview-sized
# steps (``u_relative_step_size`` is set to the requested step in overview
# voxels), lands exactly on the detail box's entry so no step jumps over it,
# and takes ``u_fine_fraction`` of a step inside it (the requested step in
# detail voxels). Extra iterations for the detail box are capped per ray.
# Each sample therefore spans the requested step in voxels of the texture
# it reads, as when that level is rendered alone, so snippets that
# integrate along the ray (attenuation, compositing) keep their per-voxel
# meaning in both regions: ``step_weight`` is the requested step. Surface
# gradients likewise sample one voxel of that texture
# (``clipmap_gradient_step``): a full-resolution voxel lies inside a single
# overview voxel, where the gradient is zero and surfaces render unlit.
CLIPMAP_STEPPING_UNIFORMS = """
uniform float u_clipmap_adaptive;
uniform vec3 u_detail_min;
uniform vec3 u_detail_max;
uniform float u_fine_fraction;
uniform float u_max_fine_steps;
uniform float u_coarse_weight;
uniform vec3 u_detail_voxel;
uniform vec3 u_overview_voxel;
$ring_uniforms

// Floating origin (see CLIPMAP_ORIGIN_SETUP): rays from the pixel position
// and per-frame values composed in float64, relative to an origin at the
// detail box.
uniform float u_origin_mode;
uniform vec2 u_frag_origin;
uniform vec3 u_frag_dx;
uniform vec3 u_frag_dy;
uniform vec3 u_ray_dir;
uniform float u_perspective;
uniform vec3 u_eye;
uniform float u_near_t;
uniform vec3 u_box_min;
uniform vec3 u_box_max;

vec3 clipmap_gradient_step(vec3 loc, vec3 dstep) {
    if (u_clipmap_adaptive < 0.5) {
        return dstep;
    }
    if (all(greaterThanEqual(loc, u_detail_min))
            && all(lessThan(loc, u_detail_max))) {
        return 1.5 * u_detail_voxel;
    }
$ring_gradient
    return 1.5 * u_overview_voxel;
}
"""

# Rings: up to CLIPMAP_MAX_RINGS more textures between the detail box and
# the overview, each a coarser level over a larger box around the detail
# (a clipmap's nested levels). Rays step and take gradients at each ring's
# own resolution, finest box first. Unused rings have empty bounds.
CLIPMAP_MAX_RINGS = 3


def _ring_source(template: str) -> str:
    return ''.join(
        template.replace('K', str(index)) for index in range(CLIPMAP_MAX_RINGS)
    )


CLIPMAP_STEPPING_UNIFORMS = CLIPMAP_STEPPING_UNIFORMS.replace(
    '$ring_uniforms\n',
    _ring_source(
        """uniform float u_ringK_active;
uniform vec3 u_ringK_min;
uniform vec3 u_ringK_max;
uniform vec3 u_ringK_voxel;
uniform float u_ringK_fraction;
"""
    ),
).replace(
    '$ring_gradient\n',
    _ring_source(
        """    if (u_ringK_active > 0.5 && all(greaterThanEqual(loc, u_ringK_min))
            && all(lessThan(loc, u_ringK_max))) {
        return 1.5 * u_ringK_voxel;
    }
"""
    ),
)

CLIPMAP_STEPPING_SETUP = """$raycasting_setup

    vec3 coarse_step = step;
    // loc(ray_t) = ray_anchor + ray_t * coarse_step; the ray ends at
    // ray_t == ray_steps. With a floating origin the anchor is the ray's
    // point near the origin, so positions near the detail box are small,
    // exactly representable numbers rather than a far box face plus steps.
    vec3 ray_anchor = start_loc;
    float ray_t = 0.0;
    float ray_steps = float(nsteps);
    if (origin_ray) {
        ray_anchor = (origin_anchor + 0.5) / u_shape;
        ray_t = origin_t_start;
        ray_steps = origin_t_start + float(nsteps);
    }
    float step_weight = u_relative_step_size;
    float fine_fraction = 1.0;
    float detail_t0 = 1e30;
    float detail_t1 = -1e30;
    if (u_clipmap_adaptive > 0.5) {
        step_weight = u_coarse_weight;
        // ray/box slab test with loc(t) = ray_anchor + t * coarse_step
        vec3 safe_step = coarse_step + vec3(equal(coarse_step, vec3(0.0))) * 1e-20;
        vec3 ta = (u_detail_min - ray_anchor) / safe_step;
        vec3 tb = (u_detail_max - ray_anchor) / safe_step;
        vec3 tlo = min(ta, tb);
        vec3 thi = max(ta, tb);
        detail_t0 = max(max(tlo.x, tlo.y), max(tlo.z, ray_t));
        detail_t1 = min(min(thi.x, thi.y), min(thi.z, ray_steps));
        if (detail_t1 > detail_t0) {
            fine_fraction = clamp(
                max(u_fine_fraction, (detail_t1 - detail_t0) / u_max_fine_steps),
                1e-12, 1.0
            );
            nsteps += int((detail_t1 - detail_t0) / fine_fraction) + 2;
        }
    }
$ring_setup"""

CLIPMAP_STEPPING_ADVANCE = """
            if (u_clipmap_adaptive > 0.5) {
                // the finest box holding the ray sets the step; tolerance
                // in that box's steps: a fixed one is wider than the
                // whole detail box at deep levels (1e-4 of an overview
                // voxel is 340 level-0 voxels at 23 levels)
                float fraction = 1.0;
$ring_inside                if (ray_t >= detail_t0 - 0.5 * fine_fraction && ray_t < detail_t1) {
                    fraction = min(fraction, fine_fraction);
                }
                float next_t = ray_t + fraction;
                // land exactly on the entry of a finer box rather than
                // step over it (or carry far-away rounding into it)
$ring_landing                if (fine_fraction < fraction && detail_t0 > ray_t && detail_t0 <= next_t) {
                    next_t = min(next_t, detail_t0);
                }
                step = coarse_step * (next_t - ray_t);
                ray_t = next_t;
                loc = ray_anchor + coarse_step * ray_t;
                if (ray_t >= ray_steps) {
                    iter = nsteps;
                }
            } else {
                loc += step;
            }
"""

CLIPMAP_STEPPING_SETUP = CLIPMAP_STEPPING_SETUP.replace(
    '$ring_setup',
    _ring_source(
        """    float ringK_t0 = 1e30;
    float ringK_t1 = -1e30;
    float ringK_fraction = 1.0;
    if (u_clipmap_adaptive > 0.5 && u_ringK_active > 0.5) {
        vec3 safe_step = coarse_step + vec3(equal(coarse_step, vec3(0.0))) * 1e-20;
        vec3 ta = (u_ringK_min - ray_anchor) / safe_step;
        vec3 tb = (u_ringK_max - ray_anchor) / safe_step;
        vec3 tlo = min(ta, tb);
        vec3 thi = max(ta, tb);
        ringK_t0 = max(max(tlo.x, tlo.y), max(tlo.z, ray_t));
        ringK_t1 = min(min(thi.x, thi.y), min(thi.z, ray_steps));
        if (ringK_t1 > ringK_t0) {
            ringK_fraction = clamp(
                max(u_ringK_fraction, (ringK_t1 - ringK_t0) / u_max_fine_steps),
                1e-12, 1.0
            );
            nsteps += int((ringK_t1 - ringK_t0) / ringK_fraction) + 2;
        } else {
            ringK_t0 = 1e30;
            ringK_t1 = -1e30;
        }
    }
"""
    ),
)

CLIPMAP_STEPPING_ADVANCE = CLIPMAP_STEPPING_ADVANCE.replace(
    '$ring_inside',
    _ring_source(
        """                if (ray_t >= ringK_t0 - 0.5 * ringK_fraction && ray_t < ringK_t1) {
                    fraction = min(fraction, ringK_fraction);
                }
"""
    ),
).replace(
    '$ring_landing',
    _ring_source(
        """                if (ringK_fraction < fraction && ringK_t0 > ray_t && ringK_t0 <= next_t) {
                    next_t = min(next_t, ringK_t0);
                }
"""
    ),
)

_VOLUME_SETUP = BaseVolume._raycasting_modes['volume']
for _declared in (
    'float distance =',
    'vec3 front =',
    'int nsteps =',
    'float f_nsteps =',
    'vec3 step =',
    'vec3 start_loc =',
):
    _VOLUME_SETUP = _patch(
        _VOLUME_SETUP, _declared, _declared.split(' ', 1)[1]
    )

CLIPMAP_ORIGIN_SETUP = (
    """
    float distance;
    vec3 front;
    int nsteps;
    float f_nsteps;
    vec3 step;
    vec3 start_loc;
    bool origin_ray = u_origin_mode > 0.5;
    vec3 origin_anchor = vec3(0.0);
    float origin_t_start = 0.0;
    if (origin_ray) {
        // The interpolated box face (v_position) and near/far points are up
        // to the whole volume away, so in float32 they place a deep
        // pyramid's ray several voxels off. Build the ray from the pixel
        // instead: its point on the plane through the origin facing the
        // camera, and its direction, from values composed in float64.
        origin_anchor = (gl_FragCoord.x - u_frag_origin.x) * u_frag_dx
            + (gl_FragCoord.y - u_frag_origin.y) * u_frag_dy;
        vec3 dir = u_ray_dir;
        float t_near = u_near_t;
        if (u_perspective > 0.5) {
            dir = normalize(origin_anchor - u_eye);
            t_near = dot(u_eye - origin_anchor, dir);
        }
        view_ray = dir;
        vec3 safe_dir = dir + vec3(equal(dir, vec3(0.0))) * 1e-20;
        vec3 ta = (u_box_min - origin_anchor) / safe_dir;
        vec3 tb = (u_box_max - origin_anchor) / safe_dir;
        vec3 tlo = min(ta, tb);
        vec3 thi = max(ta, tb);
        float t_enter = max(max(max(tlo.x, tlo.y), tlo.z), t_near);
        float t_exit = min(min(thi.x, thi.y), thi.z);
        if (t_exit <= t_enter)
            discard;
        nsteps = int((t_exit - t_enter) / u_relative_step_size + 0.5);
        f_nsteps = float(nsteps);
        if (nsteps < 1)
            discard;
        front = origin_anchor + dir * t_enter;
        step = dir * u_relative_step_size / u_shape;
        start_loc = (front + 0.5) / u_shape;
        origin_t_start = t_enter / u_relative_step_size;
        frag_depth_point = front;
    } else {
"""
    + _VOLUME_SETUP
    + """
    }
"""
)

_ORIGIN_OFF = """
    bool origin_ray = false;
    vec3 origin_anchor = vec3(0.0);
    float origin_t_start = 0.0;
"""
raycasting_modes = {
    'volume': CLIPMAP_ORIGIN_SETUP,
    'plane': _ORIGIN_OFF + BaseVolume._raycasting_modes['plane'],
}


def _chain_matrix(transform: Any) -> np.ndarray | None:
    """A vispy transform chain as one float64 matrix (row vectors), or None
    if it holds a transform that is not a matrix."""
    if isinstance(transform, ChainTransform):
        matrix = np.eye(4)
        # the last transform in a chain is applied first
        for inner in reversed(transform.transforms):
            inner_matrix = _chain_matrix(inner)
            if inner_matrix is None:
                return None
            matrix = matrix @ inner_matrix
        return matrix
    if isinstance(transform, NullTransform):
        return np.eye(4)
    if isinstance(transform, STTransform):
        return np.asarray(transform.as_matrix().matrix, dtype=np.float64)
    if isinstance(transform, MatrixTransform):
        return np.asarray(transform.matrix, dtype=np.float64)
    if isinstance(transform, InverseTransform):
        # e.g. the scene transform: the inverse of the camera transform
        forward = _chain_matrix(transform._inverse)
        return None if forward is None else np.linalg.inv(forward)
    return None


def _eye_point(unproject: Any, fx: float, fy: float) -> np.ndarray:
    """Least-squares meeting point of the rays through three pixels."""
    lines = []
    for x, y in ((fx, fy), (fx + 64.0, fy), (fx, fy + 64.0)):
        near = unproject(x, y, -1.0)
        far = unproject(x, y, 1.0)
        direction = (far - near) / np.linalg.norm(far - near)
        lines.append((near, direction))
    system = np.zeros((3, 3))
    target = np.zeros(3)
    for point, direction in lines:
        projector = np.eye(3) - np.outer(direction, direction)
        system += projector
        target += projector @ point
    return np.linalg.solve(system, target)


shaders = BaseVolume._shaders.copy()
before, after = shaders['fragment'].split('void main()')
after = _patch(after, '$raycasting_setup', CLIPMAP_STEPPING_SETUP)
after = _patch(after, 'loc += step;', CLIPMAP_STEPPING_ADVANCE)
FUNCTION_DEFINITIONS = FUNCTION_DEFINITIONS + CLIPMAP_STEPPING_UNIFORMS
FAST_GRADIENT_SHADER = (
    before
    + FUNCTION_DEFINITIONS
    + FAST_GRADIENT_DEFINITION
    + CALCULATE_COLOR_DEFINITION
    + 'void main()'
    + after
)
SMOOTH_GRADIENT_SHADER = (
    before
    + FUNCTION_DEFINITIONS
    + SMOOTH_GRADIENT_DEFINITION
    + CALCULATE_COLOR_DEFINITION
    + 'void main()'
    + after
)

shaders['fragment'] = FAST_GRADIENT_SHADER


def _position_tracking(snippets: dict, index: str) -> dict:
    """Make a MIP/MinIP snippet record where its extremum was sampled.

    vispy rebuilds that position as ``start_loc + step * index``, which only
    holds for a uniform step; clipmap rays vary their step.
    """
    snippets = dict(snippets)
    snippets['before_loop'] += (
        '\n        vec3 extreme_loc = vec3(0.0);'
        '\n        vec3 extreme_step = vec3(0.0);\n'
    )
    snippets['in_loop'] = _patch(
        snippets['in_loop'],
        f'{index} = iter;',
        f'{index} = iter;\n'
        '            extreme_loc = loc;\n'
        '            extreme_step = step;',
    )
    after = snippets['after_loop']
    after = _patch(
        after,
        f'start_loc + step * (float({index}) - 0.5)',
        'extreme_loc - extreme_step * 0.5',
    )
    after = _patch(after, 'step * 0.1', 'extreme_step * 0.1')
    after = _patch(after, f'start_loc + step * float({index})', 'extreme_loc')
    snippets['after_loop'] = after
    return snippets


def _patch_in_loop(snippets: dict, old: str, new: str) -> dict:
    snippets = dict(snippets)
    snippets['in_loop'] = _patch(snippets['in_loop'], old, new)
    return snippets


rendering_methods = BaseVolume._rendering_methods.copy()
rendering_methods['mip'] = _position_tracking(rendering_methods['mip'], 'maxi')
rendering_methods['minip'] = _position_tracking(
    rendering_methods['minip'], 'mini'
)
rendering_methods['attenuated_mip'] = _patch_in_loop(
    rendering_methods['attenuated_mip'],
    'u_relative_step_size * clamp(',
    'step_weight * clamp(',
)
iso_snippets = dict(rendering_methods['iso'])
iso_snippets['in_loop'] = _patch(
    iso_snippets['in_loop'],
    'calculateColor(color, iloc, dstep)',
    'calculateColor(color, iloc, clipmap_gradient_step(iloc, dstep))',
)
rendering_methods['iso'] = iso_snippets
rendering_methods['iso_categorical'] = ISO_CATEGORICAL_SNIPPETS
rendering_methods['translucent_categorical'] = TRANSLUCENT_CATEGORICAL_SNIPPETS


CLIPMAP_LOOKUP = """
vec4 clipmap_lookup(vec3 texcoord) {
    // texcoord and the box bounds are relative to the floating origin
    // (zero when it is off); the overview spans the whole volume. The
    // finest box holding texcoord answers.
    bool in_detail = all(greaterThanEqual(texcoord, $detail_min))
        && all(lessThan(texcoord, $detail_max));
    if (in_detail) {
        vec3 span = max($detail_max - $detail_min, vec3(1e-12));
        return texture3D(
            $detail_texture, (texcoord - $detail_min) / span
        );
    }
$rings
    return texture3D($overview_texture, texcoord + $origin_tex);
}
"""

# Rings sit in front of the detail box as well as around it. Where a ray
# crosses the detail box, its samples before the box keep reading the
# overview, as without rings: a ring level's surfaces need not match the
# detail's (a generated fractal's levels each sample the function
# independently), and a coarser one in front would hide the detail. A
# sample is in front when the view ray onward from it enters the detail
# box; $view_dir is that direction in texture coordinates (from $view_eye
# under perspective), zero to never skip rings. Only surface methods
# ($ring_gate): the others blend everything along the ray, and skipping
# rings would leave overview resolution wherever the detail box's sides
# show around its front.
CLIPMAP_RING_GATE = """
    bool use_rings = true;
    vec3 gate_dir = $view_perspective > 0.5 ? texcoord - $view_eye : $view_dir;
    if ($ring_gate > 0.5 && any(notEqual(gate_dir, vec3(0.0)))) {
        vec3 safe_dir = gate_dir + vec3(equal(gate_dir, vec3(0.0))) * 1e-30;
        vec3 ta = ($detail_min - texcoord) / safe_dir;
        vec3 tb = ($detail_max - texcoord) / safe_dir;
        vec3 tlo = min(ta, tb);
        vec3 thi = max(ta, tb);
        float t_in = max(max(tlo.x, tlo.y), tlo.z);
        float t_out = min(min(thi.x, thi.y), thi.z);
        use_rings = !(t_out > max(t_in, 0.0));
    }
"""

CLIPMAP_RING_LOOKUP = """
    if (use_rings && all(greaterThanEqual(texcoord, $ringK_min))
            && all(lessThan(texcoord, $ringK_max))) {
        return texture3D(
            $ringK_texture,
            (texcoord - $ringK_min) / max($ringK_max - $ringK_min, vec3(1e-12))
        );
    }
"""


def _clipmap_lookup_source(rings: Sequence[int]) -> str:
    """The lookup, checking the given rings (finest first)."""
    if not rings:
        return CLIPMAP_LOOKUP.replace('$rings\n', '')
    return CLIPMAP_LOOKUP.replace(
        '$rings\n',
        CLIPMAP_RING_GATE
        + ''.join(
            CLIPMAP_RING_LOOKUP.replace('K', str(index)) for index in rings
        ),
    )


class Volume(TextureMixin, BaseVolume):
    """This class extends the vispy Volume visual to add categorical isosurface rendering."""

    # add the new rendering method to the snippets dict
    _shaders = shaders
    _rendering_methods = rendering_methods
    _raycasting_modes = raycasting_modes

    # Floating origin, xyz full-resolution voxels: the clipmap ray box is
    # drawn relative to it, so float32 stays precise near the detail box
    # whatever the volume's size. Zero when off.
    _origin = np.zeros(3)
    _origin_active = False
    _origin_supported = True
    _origin_transforms: dict | None = None

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.unfreeze()
        self._clipmap_enabled = False
        self._clipmap_shape: tuple[int, ...] | None = None
        self._overview_texture: Any | None = None
        self._clipmap_lookup = None
        self._clipmap_detail_bounds = None
        # per ring index: None, or {'texture', 'bounds'} with bounds in
        # normalized full-volume xyz coordinates
        self._clipmap_rings: list[dict | None] = [None] * CLIPMAP_MAX_RINGS
        self._staged_rings: dict[int, dict] = {}
        self.clamp_at_border = False
        self.iso_gradient_mode = IsoCategoricalGradientMode.FAST.value
        self.freeze()
        self._update_clipmap_stepping()

    @property
    def clipmap_enabled(self) -> bool:
        return self._clipmap_enabled

    def enable_clipmap(
        self, overview: np.ndarray, full_shape: Sequence[int]
    ) -> None:
        """Use ``overview`` outside the movable detail texture bounds."""
        overview = np.ascontiguousarray(overview)
        self.unfreeze()
        self._overview_texture = self._create_texture(
            self.texture_format, overview
        )
        self._overview_texture.set_clim(self._texture.clim)
        self._overview_texture.scale_and_set_data(overview, copy=False)
        self._clipmap_shape = tuple(full_shape)
        self._clipmap_enabled = True
        self.freeze()
        self._install_clipmap_lookup()
        self.set_clipmap_geometry()

    def disable_clipmap(self) -> None:
        self.unfreeze()
        self._clipmap_enabled = False
        self._clipmap_shape = None
        self._overview_texture = None
        self._clipmap_lookup = None
        self._clipmap_detail_bounds = None
        self._clipmap_rings = [None] * CLIPMAP_MAX_RINGS
        self._staged_rings = {}
        self.freeze()
        self._refresh_origin()
        if self._requested_step_size is not None:
            # back to one uniform step, with vispy's bound
            BaseVolume.relative_step_size.fset(self, self._requested_step_size)  # type: ignore[attr-defined]
        self._need_interpolation_update = True
        self.update()

    def set_clipmap_detail_bounds(
        self,
        low: Sequence[float],
        high: Sequence[float],
        scale: float | Sequence[float] | None = None,
    ) -> None:
        """Set detail bounds in normalized full-volume xyz coordinates."""
        if not self._clipmap_enabled or self._clipmap_lookup is None:
            return
        if scale is None:
            scale = 1
        shape = np.asarray(self._clipmap_shape, dtype=float)[::-1]
        detail_min = np.asarray(low, dtype=float) * np.asarray(scale)
        detail_max = np.asarray(high, dtype=float) * np.asarray(scale)
        detail_min = tuple(detail_min[::-1] / shape)
        detail_max = tuple(detail_max[::-1] / shape)
        self._set_clipmap_detail_bounds_normalized(detail_min, detail_max)
        self.set_clipmap_geometry()

    def _set_clipmap_detail_bounds_normalized(
        self,
        detail_min: Sequence[float],
        detail_max: Sequence[float],
    ) -> None:
        """Apply already-normalized bounds without changing geometry."""
        if not self._clipmap_enabled or self._clipmap_lookup is None:
            return
        detail_min = tuple(float(value) for value in detail_min)
        detail_max = tuple(float(value) for value in detail_max)
        self._clipmap_detail_bounds = (detail_min, detail_max)
        self._refresh_origin()

    def set_overview_data(
        self, data: np.ndarray, offset: Sequence[int] = (0, 0, 0)
    ) -> None:
        """Patch the persistent coarse texture."""
        if not self._clipmap_enabled or self._overview_texture is None:
            return
        data = np.ascontiguousarray(data)
        texture_dtype = getattr(self._overview_texture, '_data_dtype', None)
        if texture_dtype is not None and data.dtype != texture_dtype:
            data = data.astype(texture_dtype)
        self._overview_texture.set_data(data, offset=tuple(offset))
        self.update()

    @property
    def clipmap_rings(self) -> tuple[int, ...]:
        """Indices of the rings in use."""
        return tuple(
            index
            for index, ring in enumerate(self._clipmap_rings)
            if ring is not None
        )

    def set_clipmap_ring(
        self,
        index: int,
        data: np.ndarray,
        low: Sequence[float],
        high: Sequence[float],
        scale: float | Sequence[float] | None = None,
    ) -> None:
        """Show ``data`` over ``[low, high)`` as clipmap ring ``index``.

        A ring is a box of a level coarser than the detail texture and
        finer than the overview, drawn wherever no finer box covers the
        ray. Ring 0 is the finest. ``low`` and ``high`` are zyx voxels of
        the ring's level, which ``scale`` maps to full-resolution voxels,
        as for :meth:`set_clipmap_detail_bounds`.

        Shows it right away: when texture uploads are deferred (napari's
        GLIR metering), use :meth:`stage_clipmap_ring` and commit once
        they are done instead.
        """
        self.stage_clipmap_ring(index, data, low, high, scale)
        self.commit_clipmap_rings()

    def stage_clipmap_ring(
        self,
        index: int,
        data: np.ndarray,
        low: Sequence[float],
        high: Sequence[float],
        scale: float | Sequence[float] | None = None,
    ) -> None:
        """Upload ring ``index``'s next contents and bounds off-screen.

        The ring keeps drawing its current texture at its current bounds
        until :meth:`commit_clipmap_rings`: moving the bounds before a
        deferred upload lands would draw the old contents displaced by
        the move. Patches (:meth:`set_clipmap_ring_data`) go to the
        staged texture meanwhile.
        """
        if not self._clipmap_enabled or self._clipmap_shape is None:
            return
        data = np.ascontiguousarray(data)
        staged = self._staged_rings.get(index)
        ring = self._clipmap_rings[index]
        texture = None
        for candidate in (
            None if staged is None else staged['texture'],
            None if ring is None else ring.get('spare'),
        ):
            if (
                candidate is not None
                and tuple(candidate.shape[:3]) == data.shape[:3]
            ):
                texture = candidate
                break
        if texture is None:
            texture = self._create_texture(self.texture_format, data)
            texture.set_clim(self._texture.clim)
            texture.interpolation = self._texture.interpolation
            texture.scale_and_set_data(data, copy=False)
        else:
            self._set_texture_data(texture, data, (0, 0, 0))
        if scale is None:
            scale = 1
        shape = np.asarray(self._clipmap_shape, dtype=float)[::-1]
        ring_min = np.asarray(low, dtype=float) * np.asarray(scale)
        ring_max = np.asarray(high, dtype=float) * np.asarray(scale)
        bounds = (
            tuple(float(v) for v in ring_min[::-1] / shape),
            tuple(float(v) for v in ring_max[::-1] / shape),
        )
        self._staged_rings[index] = {'texture': texture, 'bounds': bounds}

    @property
    def clipmap_rings_staged(self) -> tuple[int, ...]:
        """Indices of rings with staged contents not yet shown."""
        return tuple(sorted(self._staged_rings))

    def commit_clipmap_rings(self) -> None:
        """Show every staged ring: its texture and bounds together."""
        if not self._staged_rings:
            return
        before = self.clipmap_rings
        for index, staged in self._staged_rings.items():
            old = self._clipmap_rings[index]
            # the texture it replaces takes the next staged upload
            staged['spare'] = None if old is None else old['texture']
            self._clipmap_rings[index] = staged
        self._staged_rings.clear()
        if self.clipmap_rings != before or self._clipmap_lookup is None:
            self._install_clipmap_lookup()
            return
        for index in self.clipmap_rings:
            self._clipmap_lookup[f'ring{index}_texture'] = self._clipmap_rings[
                index
            ]['texture']  # type: ignore[index]
        self._refresh_origin()

    def set_clipmap_ring_data(
        self, index: int, data: np.ndarray, offset: Sequence[int]
    ) -> None:
        """Patch ring ``index``'s texture at ``offset`` (zyx texels): the
        staged one if the ring is moving, else the one shown."""
        if not self._clipmap_enabled:
            return
        ring = self._staged_rings.get(index) or self._clipmap_rings[index]
        if ring is None:
            return
        self._set_texture_data(ring['texture'], data, offset)
        self.update()

    def clear_clipmap_rings(self, start: int = 0) -> None:
        """Stop drawing rings ``start`` and up."""
        for index in list(self._staged_rings):
            if index >= start:
                del self._staged_rings[index]
        if not any(ring is not None for ring in self._clipmap_rings[start:]):
            return
        for index in range(start, CLIPMAP_MAX_RINGS):
            self._clipmap_rings[index] = None
        self._install_clipmap_lookup()
        self._refresh_origin()

    @staticmethod
    def _set_texture_data(
        texture: Any, data: np.ndarray, offset: Sequence[int]
    ) -> None:
        data = np.ascontiguousarray(data)
        texture_dtype = getattr(texture, '_data_dtype', None)
        if texture_dtype is not None and data.dtype != texture_dtype:
            data = data.astype(texture_dtype)
        texture.set_data(data, offset=tuple(int(v) for v in offset))

    def rebind_clipmap_detail_texture(self) -> None:
        """Point the clipmap lookup at the currently presented detail."""
        if self._clipmap_lookup is not None:
            self._clipmap_lookup['detail_texture'] = self._texture
        self._update_clipmap_stepping()

    def set_clipmap_geometry(self) -> None:
        """Keep the ray box in full-resolution level-zero coordinates."""
        if not self._clipmap_enabled or self._clipmap_shape is None:
            return
        shape = self._clipmap_shape
        self.shared_program['u_shape'] = (shape[2], shape[1], shape[0])
        if getattr(self, '_vol_shape', None) != shape:
            self._vol_shape = shape
            self._need_vertex_update = True
        self._refresh_origin()
        self.update()

    @property
    def raycasting_mode(self) -> str:
        return BaseVolume.raycasting_mode.fget(self)  # type: ignore[attr-defined]

    @raycasting_mode.setter
    def raycasting_mode(self, value: str) -> None:
        BaseVolume.raycasting_mode.fset(self, value)  # type: ignore[attr-defined]
        if hasattr(self, '_clipmap_enabled'):
            # the floating origin applies to volume rays only
            self._refresh_origin()

    # Requested relative step, before vispy's lower bound; see
    # CLIPMAP_STEPPING_SETUP. In clipmap mode it is measured in overview
    # voxels outside the detail box and detail voxels inside it.
    _requested_step_size: float | None = None
    _max_fine_steps_per_ray = 1024

    @property
    def relative_step_size(self) -> float:
        if self._requested_step_size is not None:
            return self._requested_step_size
        return BaseVolume.relative_step_size.fget(self)  # type: ignore[attr-defined]

    @relative_step_size.setter
    def relative_step_size(self, value: float) -> None:
        value = float(value)
        if value <= 0:
            raise ValueError('relative_step_size cannot be 0 or negative.')
        self._requested_step_size = value
        if getattr(self, '_clipmap_enabled', False):
            # vispy's bound applies to a single full-resolution step; the
            # clipmap sizes its steps from the textures instead
            self._update_clipmap_stepping()
            self.update()
        else:
            BaseVolume.relative_step_size.fset(self, value)  # type: ignore[attr-defined]

    def _clipmap_step_sizes(self) -> tuple[float, float, float] | None:
        """(coarse step, fine fraction, coarse weight), or None.

        Steps are in full-resolution voxels, the ray box's units. The coarse
        step is the requested step in overview voxels; the fine fraction
        scales it to the requested step in detail voxels. Every sample thus
        spans the requested step in its own texture's voxels, which is the
        weight integrating snippets (attenuated MIP) give it.
        """
        if (
            not getattr(self, '_clipmap_enabled', False)
            or self._clipmap_shape is None
            or self._overview_texture is None
        ):
            return None
        full = np.asarray(self._clipmap_shape[-3:], dtype=float)[::-1]
        overview = np.asarray(self._overview_texture.shape[:3], dtype=float)[
            ::-1
        ]
        detail = np.asarray(self._texture.shape[:3], dtype=float)[::-1]
        if np.any(full <= 0) or np.any(overview <= 0) or np.any(detail <= 0):
            return None
        requested = self._requested_step_size
        if requested is None:
            requested = float(BaseVolume.relative_step_size.fget(self))  # type: ignore[attr-defined]
        overview_voxel = float(np.min(full / overview))
        coarse = max(requested * overview_voxel, float(np.min(full)) / 15360)
        fine = 1.0
        if self._clipmap_detail_bounds is not None:
            low, high = (
                np.asarray(b, dtype=float) for b in self._clipmap_detail_bounds
            )
            span = np.maximum(high - low, 0) * full
            if np.all(span > 0):
                detail_voxel = float(np.min(span / detail))
                # deep pyramids: a detail voxel is ~1e-7 of an overview
                # voxel at 23 levels; with the floating origin such steps
                # stay exact near the detail box
                fine = float(
                    np.clip(requested * detail_voxel / coarse, 1e-12, 1.0)
                )
        return coarse, fine, coarse / overview_voxel

    def _update_clipmap_stepping(self) -> None:
        program = self.shared_program
        sizes = self._clipmap_step_sizes()
        if sizes is None:
            program['u_clipmap_adaptive'] = 0.0
            program['u_detail_min'] = (0.0, 0.0, 0.0)
            program['u_detail_max'] = (0.0, 0.0, 0.0)
            program['u_fine_fraction'] = 1.0
            program['u_max_fine_steps'] = float(self._max_fine_steps_per_ray)
            program['u_coarse_weight'] = 1.0
            program['u_detail_voxel'] = (0.0, 0.0, 0.0)
            program['u_overview_voxel'] = (0.0, 0.0, 0.0)
            self._update_ring_stepping(None, None)
            return
        coarse, fine, weight = sizes
        low, high = self._clipmap_detail_bounds or ((0.0,) * 3, (0.0,) * 3)
        overview = np.asarray(self._overview_texture.shape[:3], dtype=float)
        detail = np.asarray(self._texture.shape[:3], dtype=float)
        span = np.maximum(np.asarray(high, float) - np.asarray(low, float), 0)
        # one texel of each texture, in normalized xyz box coordinates
        program['u_overview_voxel'] = tuple(1.0 / overview[::-1])
        program['u_detail_voxel'] = tuple(span / np.maximum(detail[::-1], 1))
        program['u_clipmap_adaptive'] = 1.0
        program['u_relative_step_size'] = coarse
        origin_tex = self._origin_tex()
        program['u_detail_min'] = tuple(np.asarray(low, float) - origin_tex)
        program['u_detail_max'] = tuple(np.asarray(high, float) - origin_tex)
        program['u_fine_fraction'] = fine
        program['u_max_fine_steps'] = float(self._max_fine_steps_per_ray)
        program['u_coarse_weight'] = weight
        self._update_ring_stepping(coarse, origin_tex)

    def _update_ring_stepping(
        self, coarse: float | None, origin_tex: np.ndarray | None
    ) -> None:
        """Each ring's box, texel and step, like the detail box's."""
        program = self.shared_program
        requested = self._requested_step_size
        if requested is None:
            requested = float(BaseVolume.relative_step_size.fget(self))  # type: ignore[attr-defined]
        full = (
            None
            if self._clipmap_shape is None
            else np.asarray(self._clipmap_shape[-3:], dtype=float)[::-1]
        )
        for index in range(CLIPMAP_MAX_RINGS):
            name = f'u_ring{index}'
            ring = self._clipmap_rings[index]
            if ring is None or coarse is None or full is None:
                program[f'{name}_active'] = 0.0
                program[f'{name}_min'] = (0.0, 0.0, 0.0)
                program[f'{name}_max'] = (0.0, 0.0, 0.0)
                program[f'{name}_voxel'] = (0.0, 0.0, 0.0)
                program[f'{name}_fraction'] = 1.0
                continue
            low, high = (np.asarray(b, dtype=float) for b in ring['bounds'])
            span = np.maximum(high - low, 0)
            texels = np.asarray(ring['texture'].shape[:3], dtype=float)[::-1]
            texels = np.maximum(texels, 1)
            voxel = float(np.min(span * full / texels))
            program[f'{name}_active'] = 1.0 if np.all(span > 0) else 0.0
            program[f'{name}_min'] = tuple(low - origin_tex)
            program[f'{name}_max'] = tuple(high - origin_tex)
            program[f'{name}_voxel'] = tuple(span / texels)
            program[f'{name}_fraction'] = float(
                np.clip(requested * voxel / coarse, 1e-12, 1.0)
            )

    # -- floating origin --

    def _desired_origin(self) -> np.ndarray | None:
        """The detail box's low corner in xyz full-resolution voxels, or
        None when the floating origin does not apply."""
        if (
            not self._origin_supported
            or not getattr(self, '_clipmap_enabled', False)
            or self._clipmap_shape is None
            or self._clipmap_detail_bounds is None
            or self.raycasting_mode != 'volume'
        ):
            return None
        full = np.asarray(self._clipmap_shape[-3:], dtype=float)[::-1]
        low = np.asarray(self._clipmap_detail_bounds[0], dtype=float) * full
        if not np.all(np.isfinite(low)):
            return None
        return np.floor(low)

    def _origin_tex(self) -> np.ndarray:
        """The origin in normalized xyz box coordinates."""
        if not self._origin_active or self._clipmap_shape is None:
            return np.zeros(3)
        full = np.asarray(self._clipmap_shape[-3:], dtype=float)[::-1]
        return self._origin / full

    def _refresh_origin(self, force: bool = False) -> None:
        """Move the origin to the detail box and re-express everything
        drawn relative to it."""
        origin = self._desired_origin()
        active = origin is not None
        if origin is None:
            origin = np.zeros(3)
        toggled = active != self._origin_active
        moved = not np.array_equal(origin, self._origin)
        self.unfreeze()
        self._origin = origin
        self._origin_active = active
        self.freeze()
        if toggled or moved or force:
            self._need_vertex_update = True
        program = self.shared_program
        program['u_origin_mode'] = 1.0 if active else 0.0
        if not active:
            # the shader ignores the per-frame ray uniforms without an
            # origin, but vispy warns about any that were never set
            program['u_frag_origin'] = (0.0, 0.0)
            program['u_frag_dx'] = (0.0, 0.0, 0.0)
            program['u_frag_dy'] = (0.0, 0.0, 0.0)
            program['u_ray_dir'] = (0.0, 0.0, 1.0)
            program['u_perspective'] = 0.0
            program['u_eye'] = (0.0, 0.0, 0.0)
            program['u_near_t'] = 0.0
        if self._clipmap_shape is not None:
            full = np.asarray(self._clipmap_shape[-3:], dtype=float)[::-1]
        else:
            full = np.asarray(
                getattr(self, '_vol_shape', (1, 1, 1))[-3:], float
            )[::-1]
        program['u_box_min'] = tuple(-0.5 - origin)
        program['u_box_max'] = tuple(full - 0.5 - origin)
        if self._clipmap_lookup is not None:
            low, high = self._clipmap_detail_bounds or ((0.0,) * 3, (1.0,) * 3)
            origin_tex = self._origin_tex()
            self._clipmap_lookup['detail_min'] = tuple(
                np.asarray(low, float) - origin_tex
            )
            self._clipmap_lookup['detail_max'] = tuple(
                np.asarray(high, float) - origin_tex
            )
            self._clipmap_lookup['origin_tex'] = tuple(origin_tex)
            for index, ring in enumerate(self._clipmap_rings):
                if ring is None:
                    continue
                ring_low, ring_high = ring['bounds']
                self._clipmap_lookup[f'ring{index}_min'] = tuple(
                    np.asarray(ring_low, float) - origin_tex
                )
                self._clipmap_lookup[f'ring{index}_max'] = tuple(
                    np.asarray(ring_high, float) - origin_tex
                )
        self._update_clipmap_stepping()
        if toggled:
            # (un)install the composed transforms
            self._prepare_transforms(self)
        self.update()

    def _create_vertex_data(self) -> None:
        super()._create_vertex_data()
        if not self._origin_active:
            return
        shape = self._vol_shape
        ox, oy, oz = self._origin
        x0, x1 = -0.5 - ox, shape[2] - 0.5 - ox
        y0, y1 = -0.5 - oy, shape[1] - 0.5 - oy
        z0, z1 = -0.5 - oz, shape[0] - 0.5 - oz
        pos = np.array(
            [
                [x0, y0, z0],
                [x1, y0, z0],
                [x0, y1, z0],
                [x1, y1, z0],
                [x0, y0, z1],
                [x1, y0, z1],
                [x0, y1, z1],
                [x1, y1, z1],
            ],
            dtype=np.float32,
        )
        self._vertices.set_data(pos)

    def _compute_bounds(self, axis: int, view: Any) -> tuple[float, float]:
        low, high = super()._compute_bounds(axis, view)
        if not self._origin_active:
            return low, high
        return low - self._origin[axis], high - self._origin[axis]

    def _prepare_transforms(self, view: Any) -> None:
        super()._prepare_transforms(view)
        if not getattr(self, '_origin_active', False):
            return
        if self._origin_transforms is None:
            self.unfreeze()
            self._origin_transforms = {
                name: MatrixTransform()
                for name in ('render', 'document', 'document_inv', 'clip')
            }
            self.freeze()
        transforms = self._origin_transforms
        view.view_program.vert['transform'] = transforms['render']
        view.view_program.vert['viewtransformf'] = transforms['document']
        view.view_program.vert['viewtransformi'] = transforms['document_inv']
        view.view_program.frag['viewtransformf'] = transforms['document']
        self._clip_transform = transforms['clip']
        clip_func = getattr(self, '_clip_func', None)
        if clip_func is not None:
            clip_func['clip_transform'] = transforms['clip']
        self._update_origin_frame(view)

    def _prepare_draw(self, view: Any) -> Any:
        result = super()._prepare_draw(view)
        if self._origin_active:
            self._update_origin_frame(view)
        return result

    def _update_origin_frame(self, view: Any) -> None:
        """Compose this frame's transforms with the origin in float64 and
        derive each pixel's ray from them (see CLIPMAP_ORIGIN_SETUP)."""
        transforms = self._origin_transforms
        if transforms is None:
            return
        system = view.transforms
        to_origin = np.eye(4)
        to_origin[3, :3] = self._origin
        matrices = {}
        for name, target in (
            ('render', 'render'),
            ('document', 'document'),
            ('framebuffer', 'framebuffer'),
            ('clip', self._clipping_planes_coord_system),
        ):
            matrix = _chain_matrix(system.get_transform('visual', target))
            if matrix is None:
                # not a matrix chain: draw without the floating origin
                self.unfreeze()
                self._origin_supported = False
                self.freeze()
                self._refresh_origin()
                return
            matrices[name] = to_origin @ matrix
        transforms['render'].matrix = matrices['render']
        transforms['document'].matrix = matrices['document']
        transforms['document_inv'].matrix = np.linalg.inv(matrices['document'])
        transforms['clip'].matrix = matrices['clip']

        to_pixels = matrices['framebuffer']
        from_pixels = np.linalg.inv(to_pixels)

        def unproject(x: float, y: float, z: float) -> np.ndarray:
            point = np.array([x, y, z, 1.0]) @ from_pixels
            return point[:3] / point[3]

        origin_px = np.array([0.0, 0.0, 0.0, 1.0]) @ to_pixels
        fx, fy, fz = origin_px[:3] / origin_px[3]
        anchor = unproject(fx, fy, fz)
        near = unproject(fx, fy, -1.0)
        far = unproject(fx, fy, 1.0)
        direction = far - near
        direction /= np.linalg.norm(direction)
        program = self.shared_program
        program['u_frag_origin'] = (float(fx), float(fy))
        program['u_frag_dx'] = tuple(unproject(fx + 1.0, fy, fz) - anchor)
        program['u_frag_dy'] = tuple(unproject(fx, fy + 1.0, fz) - anchor)
        program['u_ray_dir'] = tuple(direction)
        program['u_near_t'] = float(np.dot(near - anchor, direction))
        perspective = not np.allclose(to_pixels[:, 3], (0.0, 0.0, 0.0, 1.0))
        program['u_perspective'] = 1.0 if perspective else 0.0
        eye = _eye_point(unproject, fx, fy) if perspective else None
        if eye is not None:
            program['u_eye'] = tuple(eye)
        lookup = self._clipmap_lookup
        if lookup is not None and self.clipmap_rings:
            # the ring gate (CLIPMAP_RING_GATE), in texture coordinates
            # relative to the origin
            full = np.asarray(self._clipmap_shape[-3:], dtype=float)[::-1]
            lookup['view_dir'] = tuple(direction / full)
            lookup['view_eye'] = tuple(
                (eye + 0.5) / full if eye is not None else np.zeros(3)
            )
            lookup['view_perspective'] = 1.0 if perspective else 0.0
            lookup['ring_gate'] = self._ring_gate()

    def _ring_gate(self) -> float:
        """1 when rings must not hide the detail box (CLIPMAP_RING_GATE)."""
        return 1.0 if str(self.method).startswith('iso') else 0.0

    def _install_clipmap_lookup(self) -> None:
        if not self._clipmap_enabled or self._overview_texture is None:
            return
        lookup = Function(_clipmap_lookup_source(self.clipmap_rings))
        lookup['detail_texture'] = self._texture
        lookup['overview_texture'] = self._overview_texture
        for index in self.clipmap_rings:
            ring = self._clipmap_rings[index]
            lookup[f'ring{index}_texture'] = ring['texture']  # type: ignore[index]
            ring_low, ring_high = ring['bounds']  # type: ignore[index]
            lookup[f'ring{index}_min'] = ring_low
            lookup[f'ring{index}_max'] = ring_high
        if self.clipmap_rings:
            lookup['view_dir'] = (0.0, 0.0, 0.0)
            lookup['view_eye'] = (0.0, 0.0, 0.0)
            lookup['view_perspective'] = 0.0
            lookup['ring_gate'] = self._ring_gate()
        detail_min, detail_max = self._clipmap_detail_bounds or (
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
        )
        origin_tex = self._origin_tex()
        lookup['detail_min'] = tuple(
            np.asarray(detail_min, float) - origin_tex
        )
        lookup['detail_max'] = tuple(
            np.asarray(detail_max, float) - origin_tex
        )
        lookup['origin_tex'] = tuple(origin_tex)
        self._clipmap_lookup = lookup
        self.shared_program.frag['get_data'] = lookup
        self._refresh_origin(force=True)

    def _build_interpolation(self) -> None:
        super()._build_interpolation()
        if self._clipmap_enabled and self._overview_texture is not None:
            self._overview_texture.interpolation = self._texture.interpolation
            for ring in self._clipmap_rings:
                if ring is not None:
                    ring['texture'].interpolation = self._texture.interpolation
            self._install_clipmap_lookup()

    @property
    def iso_gradient_mode(self) -> str:
        return str(self._iso_gradient_mode)

    @iso_gradient_mode.setter
    def iso_gradient_mode(self, value: str) -> None:
        self._iso_gradient_mode = IsoCategoricalGradientMode(value)
        self.shared_program.frag = (
            SMOOTH_GRADIENT_SHADER
            if value == IsoCategoricalGradientMode.SMOOTH
            else FAST_GRADIENT_SHADER
        )
        self.shared_program['u_clamp_at_border'] = self._clamp_at_border
        if hasattr(self, '_clipmap_enabled'):
            self._update_clipmap_stepping()
        self.update()

    @property
    def clamp_at_border(self) -> bool:
        """Clamp values beyond volume limits when computing isosurface gradients.

        This has an effect on the appearance of labels at the border of the volume.

            True: labels will appear darker at the border. [DEFAULT]

            False: labels will appear brighter at the border, as if the volume extends beyond its
            actual limits but the labels do not.
        """
        return self._clamp_at_border

    @clamp_at_border.setter
    def clamp_at_border(self, value: bool) -> None:
        self._clamp_at_border = value
        self.shared_program['u_clamp_at_border'] = self._clamp_at_border
        self.update()
