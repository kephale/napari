import numpy as np
import pandas as pd
import napari
from napari.layers import Graph
from napari_graph import UndirectedGraph, DirectedGraph

# Data from zebrahub
# See the paper for more details: https://www.biorxiv.org/content/10.1101/2023.03.06.531398v1.full.pdf

def build_graph(n_nodes: int, n_neighbors: int) -> UndirectedGraph:
    neighbors = np.random.randint(n_nodes, size=(n_nodes * n_neighbors))
    edges = np.stack([np.repeat(np.arange(n_nodes), n_neighbors), neighbors], axis=1)

    nodes_df = pd.DataFrame(
        400 * np.random.uniform(size=(n_nodes, 4)),
        columns=["t", "z", "y", "x"],
    )
    graph = UndirectedGraph(edges=edges, coords=nodes_df[["t", "z", "y", "x"]])

    import pdb; pdb.set_trace()
    
    return graph


def load_graph(filename, scale_factors=(1, 1.24, 0.439, 0.439)):
    # TrackID,NodeID,ParentTrackID,t,t_hier_id,z,y,x,area

    df = pd.read_csv(filename)

    if "NodeID" not in df:
        df["NodeID"] = list(range(len(df)))
    df["x"] = df["x"] * scale_factors[-1]
    df["y"] = df["y"] * scale_factors[-2]
    df["z"] = df["z"] * scale_factors[-3]
    df["t"] = df["t"] * scale_factors[-4]

    # Get all unique track IDs
    all_track_ids = df["TrackID"].unique()

    # List of pairs of NodeIDs
    neighbors = []

    print(f"Number of tracks: {len(all_track_ids)}")
    
    # Collect tracks
    # TODO this is a hardcoded subset because otherwise this gets slow, number chosen to be >1M edges
    for idx, track_id in enumerate(all_track_ids[:16000]):
        if idx % 100 == 0:
            print(f"Track idx {idx}")
        nodes = df.loc[df["TrackID"] == track_id]        
        pairs = zip(nodes["NodeID"], nodes["NodeID"][1:])
        neighbors += tuple(pairs)

    # Map from NodeIDs to indices
    nodeid_to_idx = dict(zip(df["NodeID"], range(len(df))))

    edges = tuple(map(lambda tup: (nodeid_to_idx[tup[0]], nodeid_to_idx[tup[1]]), neighbors))

    nodes_df = df[["t", "z", "y", "x"]]

    print(f"Making graph with {len(edges)} edges and {len(nodes_df)} nodes.")
    
    
    graph = DirectedGraph(edges=edges, coords=nodes_df[["t", "z", "y", "x"]])
    
    return (graph, df)


if __name__ == "__main__":
    import pooch

    dest_path = pooch.retrieve(
        url="https://public.czbiohub.org/royerlab/zebrahub/imaging/single-objective/ZSNS001_tracks.csv",  # noqa: E501
        known_hash="b674b385cb98f7a892ecc6d44258b9fbc6108f4e94b3343ac93143539b780922",  # noqa: E501
    )    
    
    viewer = napari.Viewer()
    n_nodes = 1000000
    graph, df = load_graph(dest_path)

    # graph = build_graph(n_nodes, 5)
    layer = Graph(graph, out_of_slice_display=True, features=df[["t", "z", "y", "x"]], face_color="z", size=4)
    
    viewer.add_layer(layer)

    napari.run()
