import numpy as np
from vispy.scene.visuals import Volume as BaseVolume
from vispy.visuals.shaders import Function

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

shaders = BaseVolume._shaders.copy()
before, after = shaders['fragment'].split('void main()')
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

rendering_methods = BaseVolume._rendering_methods.copy()
rendering_methods['iso_categorical'] = ISO_CATEGORICAL_SNIPPETS
rendering_methods['translucent_categorical'] = TRANSLUCENT_CATEGORICAL_SNIPPETS


CLIPMAP_LOOKUP = """
vec4 clipmap_lookup(vec3 texcoord) {
    bool in_detail = all(greaterThanEqual(texcoord, $detail_min))
        && all(lessThan(texcoord, $detail_max));
    if (in_detail) {
        vec3 span = max($detail_max - $detail_min, vec3(1e-12));
        return texture3D(
            $detail_texture, (texcoord - $detail_min) / span
        );
    }
    return texture3D($overview_texture, texcoord);
}
"""


class Volume(TextureMixin, BaseVolume):
    """This class extends the vispy Volume visual to add categorical isosurface rendering."""

    # add the new rendering method to the snippets dict
    _shaders = shaders
    _rendering_methods = rendering_methods

    def __init__(self, *args, **kwargs) -> None:  # type: ignore [no-untyped-def]
        super().__init__(*args, **kwargs)
        self.unfreeze()
        self._clipmap_enabled = False
        self._clipmap_shape = None
        self._overview_texture = None
        self._clipmap_lookup = None
        self._clipmap_detail_bounds = None
        self.clamp_at_border = False
        self.iso_gradient_mode = IsoCategoricalGradientMode.FAST.value
        self.freeze()

    @property
    def clipmap_enabled(self) -> bool:
        return self._clipmap_enabled

    def enable_clipmap(self, overview: np.ndarray, full_shape) -> None:
        """Use ``overview`` outside the movable detail texture bounds."""
        overview = np.ascontiguousarray(overview)
        self.unfreeze()
        self._overview_texture = self._create_texture(
            self.texture_format, overview
        )
        self._overview_texture.set_clim(self._texture.clim)
        self._overview_texture.scale_and_set_data(overview, copy=False)
        self._clipmap_shape = tuple(int(value) for value in full_shape)
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
        self.freeze()
        self._need_interpolation_update = True
        self.update()

    def set_clipmap_detail_bounds(self, low, high, scale=None) -> None:
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
        self, detail_min, detail_max
    ) -> None:
        """Apply already-normalized bounds without changing geometry."""
        if not self._clipmap_enabled or self._clipmap_lookup is None:
            return
        detail_min = tuple(float(value) for value in detail_min)
        detail_max = tuple(float(value) for value in detail_max)
        self._clipmap_lookup['detail_min'] = detail_min
        self._clipmap_lookup['detail_max'] = detail_max
        self._clipmap_detail_bounds = (detail_min, detail_max)

    def set_overview_data(self, data: np.ndarray, offset=(0, 0, 0)) -> None:
        """Patch the persistent coarse texture."""
        if not self._clipmap_enabled or self._overview_texture is None:
            return
        data = np.ascontiguousarray(data)
        texture_dtype = getattr(self._overview_texture, '_data_dtype', None)
        if texture_dtype is not None and data.dtype != texture_dtype:
            data = data.astype(texture_dtype)
        self._overview_texture.set_data(data, offset=tuple(offset))
        self.update()

    def rebind_clipmap_detail_texture(self) -> None:
        """Point the clipmap lookup at the currently presented detail."""
        if self._clipmap_lookup is not None:
            self._clipmap_lookup['detail_texture'] = self._texture

    def set_clipmap_geometry(self) -> None:
        """Keep the ray box in full-resolution level-zero coordinates."""
        if not self._clipmap_enabled or self._clipmap_shape is None:
            return
        shape = self._clipmap_shape
        self.shared_program['u_shape'] = (shape[2], shape[1], shape[0])
        if self._vol_shape != shape:
            self._vol_shape = shape
            self._need_vertex_update = True
        self.update()

    def _install_clipmap_lookup(self) -> None:
        if not self._clipmap_enabled or self._overview_texture is None:
            return
        lookup = Function(CLIPMAP_LOOKUP)
        lookup['detail_texture'] = self._texture
        lookup['overview_texture'] = self._overview_texture
        detail_min, detail_max = self._clipmap_detail_bounds or (
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
        )
        lookup['detail_min'] = detail_min
        lookup['detail_max'] = detail_max
        self._clipmap_lookup = lookup
        self.shared_program.frag['get_data'] = lookup

    def _build_interpolation(self) -> None:
        super()._build_interpolation()
        if self._clipmap_enabled:
            self._overview_texture.interpolation = self._texture.interpolation
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
