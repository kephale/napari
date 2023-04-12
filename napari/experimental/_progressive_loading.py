from typing import Tuple, Union
import numpy as np

from ome_zarr.io import parse_url
from ome_zarr.reader import Reader
from scipy.spatial.transform import Rotation as R
from fibsem_tools import read_xarray
from cachey import Cache
import itertools
from napari.layers._data_protocols import Index, LayerDataProtocol
#from napari.qt.threading import thread_worker
import toolz as tz

from skimage.transform import resize
import dask.array as da

# A ChunkCacheManager manages multiple chunk caches
class ChunkCacheManager:
    def __init__(self, cache_size=1e9, cost_cutoff=0):
        """
        cache_size, size of cache in bytes
        cost_cutoff, cutoff anything with cost_cutoff or less
        """
        self.c = Cache(cache_size, cost_cutoff)

    def put(self, container, dataset, chunk_slice, value, cost=1):
        """Associate value with key in the given container.
        Container might be a zarr/dataset, key is a chunk_slice, and
        value is the chunk itself.
        """
        k = self.get_container_key(container, dataset, chunk_slice)
        self.c.put(k, value, cost=cost)

    def get_container_key(self, container, dataset, slice_key):
        """Create a key from container, dataset, and chunk_slice

        Parameters
        ----------
        container : str
            A string representing a zarr container
        dataset : str
            A string representing a dataset inside a zarr
        chunk_slice : slice
            A ND slice for the chunk to grab

        """
        if type(slice_key) is tuple:
            slice_key = ",".join(
                [f"{st.start}:{st.stop}:{st.step}" for st in slice_key]
            )

        return f"{container}/{dataset}@({slice_key})"

    def get(self, container, dataset, chunk_slice):
        """Get a chunk associated with the container, dataset, and chunk_size

        Parameters
        ----------
        container : str
            A string represening a zarr container
        dataset : str
            A string representing a dataset inside the container
        chunk_slice : slice
            A ND slice for the chunk to grab

        """
        return self.c.get(
            self.get_container_key(container, dataset, chunk_slice)
        )


def get_chunk(
    chunk_slice,
    array=None,
    container=None,
    dataset=None,
    cache_manager=None,
    dtype=np.uint8,
    num_retry=3,
):
    """Get a specified slice from an array (uses a cache).

    Parameters
    ----------
    chunk_slice : tuple
        a slice in array space
    array : ndarray
        one of the scales from the multiscale image
    container: str
        the zarr container name (this is used to disambiguate the cache)
    dataset: str
        the group in the zarr (this is used to disambiguate the cache)
    chunk_size: tuple
        the size of chunk that you want to fetch

    Returns
    -------
    real_array : ndarray
        an ndarray of data sliced with chunk_slice
    """

    real_array = cache_manager.get(container, dataset, chunk_slice)
    retry = 0
    while real_array is None and retry < num_retry:
        try:
            real_array = np.asarray(array[chunk_slice].compute(), dtype=dtype)
            # TODO check for a race condition that is causing this exception
            #      some dask backends are not thread-safe
        except Exception:
            print(
                f"Can't find key: {chunk_slice}, {container}, {dataset}, {array.shape}"
            )
        cache_manager.put(container, dataset, chunk_slice, real_array)
        retry += 1
    return real_array


# Example data loaders
# TODO capture some sort of metadata about scale factors
def openorganelle_mouse_kidney_labels():
    large_image = {
        "container": "s3://janelia-cosem-datasets/jrc_mus-kidney/jrc_mus-kidney.n5",
        "dataset": "labels/empanada-mito_seg",
        "scale_levels": 4,
        "scale_factors": [(1, 1, 1), (2, 2, 2), (4, 4, 4), (8, 8, 8)],
    }
    large_image["arrays"] = [
        read_xarray(
            f"{large_image['container']}/{large_image['dataset']}/s{scale}/",
            storage_options={"anon": True},
        ).data
        for scale in range(large_image["scale_levels"])
    ]
    return large_image


def openorganelle_mouse_kidney_em():
    large_image = {
        "container": "s3://janelia-cosem-datasets/jrc_mus-kidney/jrc_mus-kidney.n5",
        "dataset": "em/fibsem-uint8",
        "scale_levels": 5,
        "scale_factors": [
            (1, 1, 1),
            (2, 2, 2),
            (4, 4, 4),
            (8, 8, 8),
            (16, 16, 16),
        ],
    }
    large_image["arrays"] = [
        read_xarray(
            f"{large_image['container']}/{large_image['dataset']}/s{scale}/",
            storage_options={"anon": True},
        ).data
        for scale in range(large_image["scale_levels"])
    ]
    return large_image


# TODO this one needs testing, it is chunked over 5D
def idr0044A():
    large_image = {
        "container": "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.4/idr0044A/4007801.zarr",
        "dataset": "",
        "scale_levels": 5,
        "scale_factors": [
            (1, 1, 1),
            (1, 2, 2),
            (1, 4, 4),
            (1, 8, 8),
            (1, 16, 16),
        ],
    }
    large_image["arrays"] = [
        read_xarray(
            f"{large_image['container']}/{scale}/",
            #            storage_options={"anon": True},
        ).data.rechunk((1, 1, 128, 128, 128))
        # .data[362, 0, :, :, :].rechunk((512, 512, 512))
        for scale in range(large_image["scale_levels"])
    ]
    return large_image


def idr0075A():
    large_image = {
        "container": "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.3/idr0075A/9528933.zarr",
        "dataset": "",
        "scale_levels": 4,
        "scale_factors": [(1, 1, 1), (1, 2, 2), (1, 4, 4), (1, 8, 8)],
    }
    large_image["arrays"] = [
        read_xarray(
            f"{large_image['container']}/{scale}/",
            #            storage_options={"anon": True},
        ).data
        # .data[362, 0, :, :, :].rechunk((512, 512, 512))
        for scale in range(large_image["scale_levels"])
    ]
    # .rechunk((1, 1, 128, 128, 128))
    return large_image


def idr0051A():
    large_image = {
        "container": "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.3/idr0051A/4007817.zarr",
        "dataset": "",
        "scale_levels": 3,
        "scale_factors": [(1, 1, 1), (1, 2, 2), (1, 4, 4)],
    }
    large_image["arrays"] = [
        read_xarray(
            f"{large_image['container']}/{scale}/",
            #            storage_options={"anon": True},
        ).data
        # .data[362, 0, :, :, :].rechunk((512, 512, 512))
        for scale in range(large_image["scale_levels"])
    ]
    # .rechunk((1, 1, 128, 128, 128))
    return large_image


def luethi_zenodo_7144919():
    import os
    import pooch

    # Downloaded from https://zenodo.org/record/7144919#.Y-OvqhPMI0R
    # TODO use pooch to fetch from zenodo
    # zip_path = pooch.retrieve(
    #     url="https://zenodo.org/record/7144919#.Y-OvqhPMI0R",
    #     known_hash=None,# Update hash
    # )
    dest_dir = pooch.retrieve(
        url="https://zenodo.org/record/7144919/files/20200812-CardiomyocyteDifferentiation14-Cycle1.zarr.zip?download=1",
        known_hash="e6773fc97dcf3689e2f42e6504e0d4f4d0845c329dfbdfe92f61c2f3f1a4d55d",
        processor=pooch.Unzip(),
    )
    local_container = os.path.split(dest_dir[0])[0]
    print(local_container)
    store = parse_url(local_container, mode="r").store
    reader = Reader(parse_url(local_container))
    nodes = list(reader())
    image_node = nodes[0]
    dask_data = image_node.data

    large_image = {
        "container": local_container,
        "dataset": "B/03/0",
        "scale_levels": 5,
        "scale_factors": [
            (1, 0.1625, 0.1625),
            (1, 0.325, 0.325),
            (1, 0.65, 0.65),
            (1, 1.3, 1.3),
            (1, 2.6, 2.6),
        ],
        "chunk_size": (1, 10, 256, 256),
    }
    large_image["arrays"] = []
    for scale in range(large_image["scale_levels"]):
        array = dask_data[scale]

        # TODO extract scale_factors now

        # large_image["arrays"].append(result.data.rechunk((3, 10, 256, 256)))
        large_image["arrays"].append(
            array.rechunk((1, 10, 256, 256)).squeeze()
            # result.data[2, :, :, :].rechunk((10, 256, 256)).squeeze()
        )
    return large_image


# Code from an earlier stage to support visual debugging
# @tz.curry
# def update_point_colors(event, viewer, alpha=1.0):
#     """Update the points based on their distance to current camera.

#     Parameters:
#     -----------
#     viewer : napari.Viewer
#         Current viewer
#     event : camera.events.angles event
#         The event triggered by changing the camera angles
#     """
#     # TODO we need a grid for each scale, or the grid needs to include all scales
#     points_layer = viewer.layers['grid']
#     points = points_layer.data
#     distances = distance_from_camera_centre_line(points, viewer.camera)
#     depth = visual_depth(points, viewer.camera)
#     priorities = prioritised_chunk_loading(
#         depth, distances, viewer.camera.zoom, alpha=alpha
#     )
#     points_layer.features = pd.DataFrame(
#         {'distance': distances, 'depth': depth, 'priority': priorities}
#     )
#     # TODO want widget to change color
#     points_layer.face_color = 'priority'
#     points_layer.refresh()


# @tz.curry
# def update_shown_chunk(event, viewer, chunk_map, array, alpha=1.0):
#     """
#     chunk map is a dictionary mapping chunk centers to chunk slices
#     array is the array containing the chunks
#     """
#     # TODO hack here to insert the recursive drawing
#     points = np.array(list(chunk_map.keys()))
#     distances = distance_from_camera_centre_line(points, viewer.camera)
#     depth = visual_depth(points, viewer.camera)
#     priorities = prioritised_chunk_loading(
#         depth, distances, viewer.camera.zoom, alpha=alpha
#     )
#     first_priority_idx = np.argmin(priorities)
#     first_priority_coord = tuple(points[first_priority_idx])
#     chunk_slice = chunk_map[first_priority_coord]
#     offset = [sl.start for sl in chunk_slice]
#     # TODO note that this only updates the highest resolution
#     hi_res_layer = viewer.layers['high-res']
#     hi_res_layer.data = array[chunk_slice]
#     hi_res_layer.translate = offset
#     hi_res_layer.refresh()


def chunk_centers_3D(array: da.Array):
    """Make a dictionary mapping chunk centers to chunk slices.
    Note: if array is >3D, then the last 3 dimensions are assumed as ZYX
    and will be used for calculating centers


    Parameters
    ----------
    array: dask Array
        The input array.

    Returns
    -------
    chunk_map : dict {tuple of float: tuple of slices}
        A dictionary mapping chunk centers to chunk slices.
    """
    start_pos = [np.cumsum(sizes) - sizes for sizes in array.chunks]
    middle_pos = [
        np.cumsum(sizes) - (np.array(sizes) / 2) for sizes in array.chunks
    ]
    end_pos = [np.cumsum(sizes) for sizes in array.chunks]
    all_start_pos = list(itertools.product(*start_pos))
    # We impose 3D here
    all_middle_pos = [el[-3:] for el in list(itertools.product(*middle_pos))]
    all_end_pos = list(itertools.product(*end_pos))
    chunk_slices = []
    for start, end in zip(all_start_pos, all_end_pos):
        chunk_slice = [
            slice(start_i, end_i) for start_i, end_i in zip(start, end)
        ]
        # We impose 3D here
        chunk_slices.append(tuple(chunk_slice[-3:]))

    mapping = dict(zip(all_middle_pos, chunk_slices))
    return mapping


def rotation_matrix_from_camera(
    angles,
) -> np.ndarray:
    return R.from_euler(seq='yzx', angles=angles, degrees=True)


def visual_depth(points, camera):
    """Compute visual depth from camera position to a(n array of) point(s).

    Parameters
    ----------
    points: (N, D) array of float
        An array of N points. This can be one point or many thanks to NumPy
        broadcasting.
    camera: napari.components.Camera
        A camera model specifying a view direction and a center or focus point.

    Returns
    -------
    projected_length : (N,) array of float
        Position of the points along the view vector of the camera. These can
        be negative (in front of the center) or positive (behind the center).
    """
    view_direction = camera.view_direction
    points_relative_to_camera = points - camera.center
    projected_length = points_relative_to_camera @ view_direction
    return projected_length


def distance_from_camera_centre_line(points, camera):
    """Compute distance from a point or array of points to camera center line.

    This is the line aligned to the camera view direction and passing through
    the camera's center point, aka camera.position.

    Parameters
    ----------
    points: (N, D) array of float
        An array of N points. This can be one point or many thanks to NumPy
        broadcasting.
    camera: napari.components.Camera
        A camera model specifying a view direction and a center or focus point.

    Returns
    -------
    distances : (N,) array of float
        Distances from points to the center line of the camera.
    """
    view_direction = camera.view_direction
    projected_length = visual_depth(points, camera)
    projected = view_direction * np.reshape(projected_length, (-1, 1))
    points_relative_to_camera = (
        points - camera.center
    )  # for performance, don't compute this twice in both functions
    distances = np.linalg.norm(projected - points_relative_to_camera, axis=-1)
    return distances


def prioritised_chunk_loading(depth, distance, zoom, alpha=1.0, visible=None):
    """Compute a chunk priority based on chunk location relative to camera.
    Lower priority is preferred.

    Parameters
    ----------
    depth : (N,) array of float
        The visual depth of the points.
    distance : (N,) array of float
        The distance from the camera centerline of each point.
    zoom : float
        The camera zoom level. The higher the zoom (magnification), the
        higher the relative importance of the distance from the centerline.
    alpha : float
        Parameter weighing distance from centerline and depth. Higher alpha
        means centerline distance is weighted more heavily.
    visible : (N,) array of bool
        An array that indicates the visibility of each chunk

    Returns
    -------
    priority : (N,) array of float
        The loading priority of each chunk.
    """
    chunk_load_priority = depth + alpha * zoom * distance
    if visible is not None:
        chunk_load_priority[np.logical_not(visible)] = np.inf
    return chunk_load_priority


#@thread_worker
def render_sequence_3D_caller(
    view_slice,
    scale=0,
    camera=None,
    cache_manager=None,
    arrays=None,
    chunk_maps=None,
    container="",
    dataset="",
    alpha=0.8,
    scale_factors=[],
    dtype=np.uint16,
    dims=None,
):
    """
    Entry point for recursive function render_sequence.

    See render_sequence for docs.
    """
    yield from render_sequence_3D(
        view_slice,
        scale=scale,
        camera=camera,
        cache_manager=cache_manager,
        arrays=arrays,
        chunk_maps=chunk_maps,
        container=container,
        dataset=dataset,
        alpha=alpha,
        scale_factors=scale_factors,
        dtype=dtype,
        dims=dims,
    )


def render_sequence_3D(
    view_slice,
    scale=0,
    camera=None,
    cache_manager=None,
    arrays=None,
    chunk_maps=None,
    container="",
    dataset="",
    alpha=0.8,
    scale_factors=[],
    dtype=np.uint16,
    dims=None,
):
    """Recursively add multiscale chunks to a napari viewer for some multiscale arrays

    Note: scale levels are assumed to be 2x factors of each other

    Parameters
    ----------
    view_slice : tuple or list of slices
        A tuple/list of slices defining the region to display
    scale : float
        The scale level to display. 0 is highest resolution
    camera : Camera
        a napari Camera used for prioritizing data loading
        Note: the camera instance should be immutable.
    cache_manager : ChunkCacheManager
        An instance of a ChunkCacheManager for data fetching
    arrays : list
        multiscale arrays to display
    chunk_maps : list
        a list of dictionaries mapping chunk coordinates to chunk
        slices
    container : str
        the name of a zarr container, used for making unique keys in
        cache
    dataset : str
        the name of a zarr dataset, used for making unique keys in
        cache
    alpha : float
        a parameter that tunes the behavior of chunk prioritization
        see prioritised_chunk_loading for more info
    scale_factors : list of tuples
        a list of tuples of scale factors for each array
    dtype : dtype
        dtype of data
    """


    # Get some variables specific to this scale level
    min_coord = [st.start for st in view_slice]
    max_coord = [st.stop for st in view_slice]
    array = arrays[scale]
    chunk_map = chunk_maps[scale]
    scale_factor = scale_factors[scale]

    # Points for each chunk, for example, centers
    points = np.array(list(chunk_map.keys()))

    # Mask of whether points are within our interval, this is in array coordinates
    point_mask = np.array(
        [
            np.all(point >= min_coord) and np.all(point <= max_coord)
            for point in points
        ]
    )

    # Rescale points to world for priority calculations
    points_world = points * np.array(scale_factor)

    # Prioritize chunks using world coordinates
    distances = distance_from_camera_centre_line(points_world, camera)
    depth = visual_depth(points_world, camera)
    priorities = prioritised_chunk_loading(
        depth, distances, camera.zoom, alpha=alpha, visible=point_mask
    )

    # Select the number of chunks
    # TODO consider using threshold on priorities
    """
    Note: 
    switching from recursing on 1 top chunk to N-best introduces extra
    complexity, because the shape of texture allocation needs to
    accommodate projections from all viewpoints around the volume.
    """
    n = 1
    best_priorities = np.argsort(priorities)[:n]

    # This node's offset in world space
    world_offset = np.array(min_coord) * np.array(scale_factor)

    # Iterate over points/chunks and add corresponding nodes when appropriate
    for idx, point in enumerate(points):
        # TODO: There are 2 strategies here:
        # 1. Render *visible* chunks, or all if we're on the last scale level
        #    Skip the chunk at this resolution because it will be shown in higher res
        #    This fetches less data.
        # if point_mask[idx] and (idx not in best_priorities or scale == 0):
        # 2. Render all chunks because we know we will zero out this data when
        #    it is loaded at the next resolution level.
        if point_mask[idx]:
            coord = tuple(point)
            chunk_slice = chunk_map[coord]
            offset = [sl.start for sl in chunk_slice]
            min_interval = offset

            # find position and scale
            node_offset = (
                min_interval[0] * scale_factor[0],
                min_interval[1] * scale_factor[1],
                min_interval[2] * scale_factor[2],
            )

            # LOGGER.debug(
            #     f"Fetching: {(scale, chunk_slice)} World offset: {node_offset}"
            # )

            scale_dataset = f"{dataset}/s{scale}"

            # When we get_chunk chunk_slice needs to be in data space, but chunk slices are 3D
            data_slice = tuple(
                [slice(el, el + 1) for el in dims.current_step[:-3]]
                + [slice(sl.start, sl.stop) for sl in chunk_slice]
            )

            data = get_chunk(
                data_slice,
                array=array,
                container=container,
                dataset=scale_dataset,
                cache_manager=cache_manager,
                dtype=dtype,
            )

            # Texture slice (needs to be in layer.data dimensions)
            # TODO there is a 3D ordering assumption here
            texture_slice = tuple(
                [slice(el, el + 1) for el in dims.current_step[:-3]]
                + [
                    slice(sl.start - offset, sl.stop - offset)
                    for sl, offset in zip(chunk_slice, min_coord)
                ]
            )
            if texture_slice[1].start < 0:
                import pdb

                pdb.set_trace()

            # TODO consider a data class instead of a tuple
            yield (
                np.asarray(data),
                scale,
                offset,
                # world_offset,
                None,
                chunk_slice,
                texture_slice,
            )

    # TODO make sure that all of low res loads first
    # TODO take this 1 step further and fill all high resolutions with low res

    # recurse on best priorities
    if scale > 0:
        # The next priorities for loading in higher resolution are the best ones
        for priority_idx in best_priorities:
            # Get the coordinates of the chunk for next scale
            priority_coord = tuple(points[priority_idx])
            chunk_slice = chunk_map[priority_coord]

            # Blank out the region that will be filled in by other scales
            zeros_size = list(array.shape[:-3]) + [
                sl.stop - sl.start for sl in chunk_slice
            ]

            zdata = np.zeros(np.array(zeros_size, dtype=dtype), dtype=dtype)

            # TODO there is a 3D ordering assumption here
            texture_slice = tuple(
                [slice(el, el + 1) for el in dims.current_step[:-3]]
                + [
                    slice(sl.start - offset, sl.stop - offset)
                    for sl, offset in zip(chunk_slice, min_coord)
                ]
            )

            # Compute the relative scale factor between these layers
            relative_scale_factor = [
                this_scale / next_scale
                for this_scale, next_scale in zip(
                    scale_factors[scale], scale_factors[scale - 1]
                )
            ]

            # now convert the chunk slice to the next scale
            next_chunk_slice = [
                slice(st.start * dim_scale, st.stop * dim_scale)
                for st, dim_scale in zip(chunk_slice, relative_scale_factor)
            ]

            next_min_coord = [st.start for st in next_chunk_slice]
            # TODO this offset is incorrect
            next_world_offset = np.array(next_min_coord) * np.array(
                scale_factors[scale - 1]
            )

            # TODO Note that we need to be blanking out lower res data at the same time
            # TODO this is when we should move the node from the next resolution.
            yield (
                np.asarray(zdata),
                scale,
                tuple([sl.start for sl in chunk_slice]),
                next_world_offset,
                chunk_slice,
                texture_slice,
            )

            # LOGGER.info(
            #     f"Recursive add on\t{str(next_chunk_slice)} idx {priority_idx}",
            #     f"visible {point_mask[priority_idx]} for scale {scale} to {scale-1}\n",
            #     f"Relative scale factor {relative_scale_factor}",
            # )
            yield from render_sequence_3D(
                next_chunk_slice,
                scale=scale - 1,
                camera=camera,
                cache_manager=cache_manager,
                arrays=arrays,
                chunk_maps=chunk_maps,
                container=container,
                dataset=dataset,
                scale_factors=scale_factors,
                dtype=dtype,
                dims=dims,
            )

    
@tz.curry
def _on_yield_2D(coord, layer=None):
    chunk_slice, scale, chunk = coord
    layer.data[chunk_slice] = chunk
    layer.refresh()
    
# # TODO consider filling these chunks into a queue and processing them in batches
@tz.curry
def _on_yield_3D(
        chunk_tuple,
        layer=None,
        viewer=None,
):
    """Update the display with a chunk

    Update a display when a chunk is recieved. This has been developed
    as a function that is associated with on_yielded events from a
    generator that yields chunks of image data (e.g. chunks from a
    zarr image).

    Parameters
    ----------
    chunk_tuple : tuple
        tuple that contains data and metadata necessary to update the
        display
    viewer : napari.viewer.Viewer
        a napari viewer with layers the given container, dataset that
        will be updated with the chunk's data
    container : str
        a zarr container
    dataset : str
        group in container
    dtype : dtype
        a numpy-like dtype

    """
    (
        data,
        scale,
        array_offset,
        node_offset,
        chunk_slice,
        texture_slice,
    ) = chunk_tuple

    dtype = np.uint16
    
    # TODO 3D assumed here as last 3 dimensions
    texture_offset = tuple([sl.start for sl in texture_slice[-3:]])    

    volume = viewer.window.qt_viewer.layer_to_visual[
        layer
    ]._layer_node.get_node(3)
    
    texture = volume._texture
    
    new_texture_data = np.asarray(
        data,
        dtype=dtype,
    )
    
    # Note: due to odd dimensions in scale pyramids sometimes we have off by 1
    ntd_slice = (
        slice(0, layer.data[texture_slice].shape[1]),
        slice(0, layer.data[texture_slice].shape[2]),
        slice(0, layer.data[texture_slice].shape[3]),
    )
    if len(new_texture_data.shape) == 3:
        import pdb
        
        pdb.set_trace()

    layer.data[texture_slice] = new_texture_data[
        : layer.data[texture_slice].shape[0],
        : layer.data[texture_slice].shape[1],
        : layer.data[texture_slice].shape[2],
        : layer.data[texture_slice].shape[3],
    ]

    # TODO explore efficiency of either approach, or maybe even an alternative
    # Writing a texture with an offset is slower
    # texture.set_data(new_texture_data, offset=texture_offset)
    texture.set_data(np.asarray(layer.data[texture_slice]).squeeze())
    # TODO we might need to zero out parts of the texture when there is a ragged boundary size

    volume.update()

    # Translate the layer we're rendering to the right place
    # TODO: this is called too many times. we only need to call it once for each time it changes
    # Note: this can trigger a refresh, we should do it after setting data to not trigger an
    #       extra materialization with
    # TODO this might be a race condition with multiple on_yielded events
    if node_offset is not None:
        try:
            # Trigger a refresh of our layer
            layer.refresh()

            container = layer.metadata["zarr_container"]
            dataset = layer.metadata["zarr_dataset"]
            
            next_layer_name = f"{container}/{dataset}/s{scale - 1}"
            next_layer = viewer.layers[next_layer_name]
            next_layer.translate = node_offset
        except Exception as e:
            import pdb

            pdb.traceback.print_exception(e)
            pdb.traceback.print_stack()

            pdb.set_trace()

    # The node translate calls made when zeroing will update everyone except 0
    if scale == 0:
        layer.refresh()


# from interpolated_tiled_rendering

def interpolated_get_chunk_2D(
    chunk_slice, array=None, container=None, dataset=None, cache_manager=None
):
    """Get a specified slice from an array, with interpolation when necessary.
    Interpolation is linear.
    Out of bounds behavior is zeros outside the shape.

    Parameters
    ----------
    coord : tuple
        a float 3D coordinate into the array like (0.5, 0, 0)
    array : ndarray
        one of the scales from the multiscale image
    container: str
        the zarr container name (this is used to disambiguate the cache)
    dataset: str
        the group in the zarr (this is used to disambiguate the cache)
    chunk_size: tuple
        the size of chunk that you want to fetch

    Returns
    -------
    real_array : ndarray
        an ndarray of data sliced with chunk_slice
    """
    real_array = cache_manager.get(container, dataset, chunk_slice)
    if real_array is None:
        # If we do not need to interpolate
        # TODO this isn't safe enough
        if all([(sl.start % 1 == 0) for sl in chunk_slice]):
            real_array = get_chunk(
                chunk_slice,
                array=array,
                container=container,
                dataset=dataset,
                cache_manager=cache_manager,
            )
        else:
            # Get left and right keys
            # TODO int casting may be dangerous
            lchunk_slice = (
                slice(
                    int(np.floor(chunk_slice[0].start - 1)),
                    int(np.floor(chunk_slice[0].stop - 1)),
                ),
                chunk_slice[1],
                chunk_slice[2],
            )
            rchunk_slice = (
                slice(
                    int(np.ceil(chunk_slice[0].start + 1)),
                    int(np.ceil(chunk_slice[0].stop + 1)),
                ),
                chunk_slice[1],
                chunk_slice[2],
            )

            # Handle out of bounds with zeros
            try:
                lvalue = get_chunk(
                    lchunk_slice,
                    array=array,
                    container=container,
                    dataset=dataset,
                    cache_manager=cache_manager,
                )
            except:
                lvalue = np.zeros([1] + list(array.chunksize[-2:]))
            try:
                rvalue = get_chunk(
                    rchunk_slice,
                    array=array,
                    container=container,
                    dataset=dataset,
                    cache_manager=cache_manager,
                )
            except:
                rvalue = np.zeros([1] + list(array.chunksize[-2:]))

            # Linear weight between left/right, assumes parallel
            w = chunk_slice[0].start - lchunk_slice[0].start

            # TODO hardcoded dtype
            # TODO squeeze is a bad sign
            real_array = (
                ((1 - w) * lvalue + w * rvalue).astype(np.uint16).squeeze()
            )

        # Save in cache
        cache_manager.put(container, dataset, chunk_slice, real_array)
    return real_array


def chunks_for_scale_2D(corner_pixels, array, scale):
    """Return the keys for all chunks at this scale within the corner_pixels

    Parameters
    ----------
    corner_pixels : tuple
        ND top left and bottom right coordinates for the current view
    array : arraylike
        a ND numpy array with the data for this scale
    scale : int
        the scale level, assuming powers of 2

    """

    # TODO all of this needs to be generalized to ND or replaced/merged with volume rendering code

    mins = corner_pixels[0, :] / (2**scale)
    maxs = corner_pixels[1, :] / (2**scale)

    chunk_size = array.chunksize

    # TODO kludge for 3D z-only interpolation
    zval = mins[-3]
    
    # Find the extent from the current corner pixels, limit by data shape
    # TODO risky int cast
    mins = (np.floor(mins / chunk_size) * chunk_size).astype(int)
    maxs = np.min(
        (np.ceil(maxs / chunk_size) * chunk_size, np.array(array.shape)),
        axis=0,
    ).astype(int)

    mins[-3] = maxs[-3] = zval

    xs = range(mins[-1], maxs[-1], chunk_size[-1])
    ys = range(mins[-2], maxs[-2], chunk_size[-2])
    zs = [zval]
    # TODO kludge

    for x in xs:
        for y in ys:
            for z in zs:
                yield (
                    slice(z, (z + 1)),
                    slice(y, (y + chunk_size[-2])),
                    slice(x, (x + chunk_size[-1])),
                )


class VirtualData:
    """VirtualData is used to use a 2D array to represent
    a larger shape. The purpose of this function is to provide
    a 2D slice that acts like a canvas for rendering a large ND image.

    When you try to fetch a ND coordinate, only the last 2 dimensions
    will be used as the (y, x) values.
    """

    def __init__(self, dtype, shape):
        self.dtype = dtype
        self.shape = shape
        self.ndim = len(shape)

        self.d = 2

        # TODO: I don't like that this is making a choice of slicing axis
        self.data_plane = np.zeros(self.shape[-1 * self.d :], dtype=self.dtype)

    def __getitem__(
        self, key: Union[Index, Tuple[Index, ...], LayerDataProtocol]
    ) -> LayerDataProtocol:
        """Returns self[key]."""
        if type(key) is tuple:
            return self.data_plane.__getitem__(tuple(key[-1 * self.d :]))
        else:
            return self.data_plane.__getitem__(key)

    def __setitem__(
        self, key: Union[Index, Tuple[Index, ...], LayerDataProtocol], value
    ) -> LayerDataProtocol:
        """Returns self[key]."""
        if type(key) is tuple:
            return self.data_plane.__setitem__(
                tuple(key[-1 * self.d :]), value
            )
        else:
            return self.data_plane.__setitem__(key, value)



def get_and_process_chunk_2D(chunk_slice, scale, array, full_shape, cache_manager=None, dataset="", container=""):
    """Fetch and upscale a chunk

    Parameters
    ----------
    chunk_slice : tuple of slices
        a key corresponding to the chunk to fetch
    scale : int
        scale level, assumes power of 2
    array : arraylike
        the ND array to fetch a chunk from
    full_shape : tuple
        a tuple storing the shape of the highest resolution level

    """
    # z, y, x = coord

    # Trigger a fetch of the data
    dataset = f"{dataset}/s{scale}"
    real_array = interpolated_get_chunk_2D(
        chunk_slice,
        array=array,
        container=container,
        dataset=dataset,
        cache_manager=cache_manager,
    )
    
    upscale_factor = [el * 2**scale for el in real_array.shape]
    
    # Upscale the data to highest resolution
    upscaled = resize(
        real_array,
        upscale_factor,
        preserve_range=True,
    )

    # TODO imposes 3D
    z, y, x = [sl.start for sl in chunk_slice]

    # Use this to overwrite data and then use a colormap to debug where resolution levels go
    # upscaled = np.ones_like(upscaled) * scale

    # Return upscaled coordinates, the scale, and chunk
    chunk_size = upscaled.shape

    upscaled_chunk_size = [0, 0]
    upscaled_chunk_size[0] = min(
        full_shape[-2] - y * 2**scale,
        chunk_size[-2],
    )
    upscaled_chunk_size[1] = min(
        full_shape[-1] - x * 2**scale,
        chunk_size[-1],
    )

    upscaled = upscaled[: upscaled_chunk_size[-2], : upscaled_chunk_size[-1]]

    # TODO This is unclean!
    upscaled_chunk_slice = [None] * 3
    for idx, sl in enumerate(chunk_slice):
        start_coord = sl.start * 2**scale
        stop_coord = sl.stop * 2**scale
        # Check for ragged edges
        if idx > 0:
            stop_coord = start_coord + min(
                stop_coord - start_coord, upscaled_chunk_size[idx - 1]
            )

        upscaled_chunk_slice[idx] = slice(start_coord, stop_coord)

    return (
        tuple(upscaled_chunk_slice),
        scale,
        upscaled,
    )


#@thread_worker
def render_sequence_2D(
        corner_pixels, full_shape, num_threads=1, cache_manager=None, arrays=None, dataset="", container="",
):
    """A generator that yields multiscale chunk tuples from low to high resolution.

    Parameters
    ----------
    corner_pixels : tuple
        ND coordinates of the topleft bottomright coordinates of the
        current view
    full_shape : tuple
        shape of highest resolution array
    num_threads : int
        number of threads for multithreaded fetching
    """
    # NOTE this corner_pixels means something else and should be renamed
    # it is further limited to the visible data on the vispy canvas

    for scale in reversed(range(len(arrays))):
        array = arrays[scale]

        chunks_to_fetch = list(chunks_for_scale_2D(corner_pixels, array, scale))

        if num_threads == 1:
            # Single threaded:
            for chunk_slice in chunks_to_fetch:
                yield get_and_process_chunk_2D(
                    chunk_slice,
                    scale,
                    array,
                    full_shape,
                    cache_manager=cache_manager,
                    dataset=dataset,
                    container=container,
                )

        else:
            raise Exception("Multithreading disabled")


        
