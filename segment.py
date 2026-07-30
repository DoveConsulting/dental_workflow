"""
Dental STL Connector Segmentation — Curvature + Morphological Approach
----------------------------------------------------------------------
Segments fused connectors from teeth in single-body dental STL meshes.

Strategy (two complementary methods, automatically selected):

1. **Curvature-based** (primary):
   - Compute discrete mean curvature at every vertex via the cotangent
     Laplacian (angle-deficit method).
   - Threshold vertices with highly negative mean curvature — these form
     the concave crease boundaries where connectors meet teeth.
   - Remove boundary vertices from the face adjacency graph, yielding
     disconnected sub-graphs (one per tooth or connector).
   - Classify sub-graphs by volume: large = tooth, small = connector.

2. **Voxel morphological** (fallback for noisy/low-res scans):
   - Voxelize the mesh into a binary occupancy grid.
   - Apply Euclidean Distance Transform (EDT); thin connectors have small
     EDT values everywhere while tooth cores have large EDT peaks.
   - Threshold EDT to isolate tooth seeds, label them, then watershed-
     expand back to reclaim the full mesh.
   - Vertices in collision zones between expanding labels → connectors.

Usage:
    python segment.py input.stl [--output-dir segmented]
    python segment.py input.stl --method voxel --voxel-size 0.3
    python segment.py stl_test/  # batch process a directory
"""

import argparse
import os
from pathlib import Path

import numpy as np
import trimesh
import networkx as nx
from scipy import ndimage
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# Mesh loading
# ---------------------------------------------------------------------------

def load_mesh(filepath: str) -> trimesh.Trimesh:
    mesh = trimesh.load(filepath, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh.dump())
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Could not load a triangle mesh from {filepath}")
    return mesh


# ---------------------------------------------------------------------------
# Discrete curvature estimation
# ---------------------------------------------------------------------------

def compute_mean_curvature(mesh: trimesh.Trimesh) -> np.ndarray:
    """
    Approximate discrete mean curvature per vertex using trimesh's
    discrete_mean_curvature_measure with a small radius neighbourhood.
    Falls back to Laplacian dot normal if the above is unavailable.
    """
    try:
        curvature = trimesh.curvature.discrete_mean_curvature_measure(
            mesh, mesh.vertices, radius=mesh.scale * 0.02
        )
    except Exception:
        # Fallback: Laplacian-based approximation
        laplacian = trimesh.smoothing.laplacian_calculation(mesh)
        delta = laplacian.dot(mesh.vertices)
        curvature = np.sum(delta * mesh.vertex_normals, axis=1)
    return curvature


# ---------------------------------------------------------------------------
# Curvature-based segmentation
# ---------------------------------------------------------------------------

def segment_by_curvature(
    mesh: trimesh.Trimesh,
    curvature_percentile: float = 8.0,
    min_component_faces: int = 50,
) -> list[trimesh.Trimesh]:
    """
    Identify concave crease vertices (highly negative curvature), remove them
    from the face adjacency graph, then extract connected components.
    """
    curvature = compute_mean_curvature(mesh)

    # Negative curvature = concave. Find the most concave vertices.
    threshold = np.percentile(curvature, curvature_percentile)
    boundary_verts = set(np.where(curvature < threshold)[0])

    # Build face adjacency graph, excluding faces that touch boundary vertices
    faces = mesh.faces
    num_faces = len(faces)

    boundary_faces = set()
    for fi in range(num_faces):
        if any(int(v) in boundary_verts for v in faces[fi]):
            boundary_faces.add(fi)

    # Use trimesh's face adjacency for efficient neighbour lookup
    face_graph = nx.Graph()
    for fi in range(num_faces):
        if fi not in boundary_faces:
            face_graph.add_node(fi)

    for pair in mesh.face_adjacency:
        f0, f1 = int(pair[0]), int(pair[1])
        if f0 not in boundary_faces and f1 not in boundary_faces:
            face_graph.add_edge(f0, f1)

    # Extract connected components
    components = []
    for cc in nx.connected_components(face_graph):
        if len(cc) < min_component_faces:
            continue
        face_indices = np.array(sorted(cc))
        submesh = mesh.submesh([face_indices], append=True)
        if submesh is not None and len(submesh.faces) > 0:
            components.append(submesh)

    return components


# ---------------------------------------------------------------------------
# Voxel morphological segmentation (fallback)
# ---------------------------------------------------------------------------

def segment_by_voxel_morphology(
    mesh: trimesh.Trimesh,
    voxel_size: float | None = None,
    erosion_iterations: int = 3,
    min_component_faces: int = 50,
) -> list[trimesh.Trimesh]:
    """
    Voxelize → EDT → threshold to find tooth cores → label → watershed
    expand → map labels back to mesh vertices.
    """
    if voxel_size is None:
        voxel_size = mesh.scale * 0.005  # ~200 voxels along longest axis

    # Voxelize
    voxel_grid = mesh.voxelized(voxel_size)
    matrix = voxel_grid.matrix.astype(bool)

    # Euclidean distance transform (distance to nearest background voxel)
    edt = ndimage.distance_transform_edt(matrix)

    # Erode: only keep voxels far from the surface (tooth cores survive)
    erode_threshold = np.percentile(edt[matrix], 60)
    seeds = edt > erode_threshold

    # Label the surviving seed regions
    labelled_seeds, num_seeds = ndimage.label(seeds)

    if num_seeds < 2:
        # Try more aggressive threshold
        erode_threshold = np.percentile(edt[matrix], 75)
        seeds = edt > erode_threshold
        labelled_seeds, num_seeds = ndimage.label(seeds)

    if num_seeds < 2:
        return []

    # Watershed-like expansion: iteratively dilate seeds into occupied space
    labels = labelled_seeds.copy()
    max_iter = int(np.max(edt)) + 10
    for _ in range(max_iter):
        dilated = ndimage.grey_dilation(labels, size=3)
        expand_mask = (labels == 0) & matrix
        labels[expand_mask] = dilated[expand_mask]
        if not np.any((labels == 0) & matrix):
            break

    # Map voxel labels back to mesh vertices via nearest filled voxel
    filled_coords = np.argwhere(matrix)
    label_values = labels[matrix]

    if len(filled_coords) == 0:
        return []

    # Convert voxel indices to world coordinates
    voxel_origin = voxel_grid.transform[:3, 3]
    filled_world = filled_coords * voxel_size + voxel_origin
    tree = cKDTree(filled_world)
    _, indices = tree.query(mesh.vertices)

    vertex_labels = label_values[indices]

    # Extract submeshes per label
    components = []
    for label_id in range(1, num_seeds + 1):
        verts_in_label = set(np.where(vertex_labels == label_id)[0])
        face_mask = np.array([
            all(int(v) in verts_in_label for v in face)
            for face in mesh.faces
        ])
        if face_mask.sum() < min_component_faces:
            continue
        face_indices = np.where(face_mask)[0]
        submesh = mesh.submesh([face_indices], append=True)
        if submesh is not None and len(submesh.faces) > 0:
            components.append(submesh)

    return components


# ---------------------------------------------------------------------------
# Classification: teeth vs connectors
# ---------------------------------------------------------------------------

def classify_components(
    components: list[trimesh.Trimesh],
) -> tuple[list[trimesh.Trimesh], list[trimesh.Trimesh]]:
    """
    Classify segmented components as teeth or connectors using volume and
    aspect ratio. Connectors are thin/elongated and small in volume.
    """
    if not components:
        return [], []

    if len(components) == 1:
        return components, []

    metrics = []
    for comp in components:
        try:
            vol = abs(comp.volume)
        except Exception:
            vol = comp.convex_hull.volume if comp.is_watertight else comp.area * 0.01
        obb = comp.bounding_box_oriented
        extents = sorted(obb.extents)
        # Aspect ratio: ratio of smallest to largest extent
        aspect = extents[0] / (extents[-1] + 1e-9)
        metrics.append((vol, aspect))

    volumes = np.array([m[0] for m in metrics])
    aspects = np.array([m[1] for m in metrics])

    # Use a combined score: high = likely tooth, low = likely connector
    # Normalize both metrics to [0, 1]
    vol_norm = (volumes - volumes.min()) / (volumes.max() - volumes.min() + 1e-9)
    asp_norm = (aspects - aspects.min()) / (aspects.max() - aspects.min() + 1e-9)

    # Combined score weighted toward volume
    score = 0.6 * vol_norm + 0.4 * asp_norm

    # Use Otsu-like threshold on scores
    threshold = np.percentile(score, 30)

    teeth, connectors = [], []
    for comp, s in zip(components, score):
        if s <= threshold:
            connectors.append(comp)
        else:
            teeth.append(comp)

    # Sanity: if everything is "connector", treat all as teeth
    if not teeth:
        teeth = components
        connectors = []

    return teeth, connectors


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def save_components(
    components: list[trimesh.Trimesh],
    output_dir: str,
    prefix: str,
) -> list[str]:
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    for i, comp in enumerate(components):
        out_path = os.path.join(output_dir, f"{prefix}_{i:03d}.stl")
        comp.export(out_path)
        paths.append(out_path)
        print(f"  Saved: {out_path}")
    return paths


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def segment(
    input_path: str,
    output_dir: str = "segmented",
    method: str = "auto",
    curvature_percentile: float = 8.0,
    voxel_size: float | None = None,
    min_component_faces: int = 50,
) -> None:
    print(f"Loading {input_path} …")
    mesh = load_mesh(input_path)
    print(f"  {len(mesh.vertices):,} vertices, {len(mesh.faces):,} faces")

    base = Path(input_path).stem

    # Choose method
    if method == "auto":
        # Try curvature first; fall back to voxel if it yields too few segments
        components = segment_by_curvature(mesh, curvature_percentile, min_component_faces)
        if len(components) < 3:
            print("  Curvature method yielded few segments, trying voxel morphology…")
            components_voxel = segment_by_voxel_morphology(
                mesh, voxel_size, min_component_faces=min_component_faces
            )
            if len(components_voxel) > len(components):
                components = components_voxel
    elif method == "curvature":
        components = segment_by_curvature(mesh, curvature_percentile, min_component_faces)
    elif method == "voxel":
        components = segment_by_voxel_morphology(
            mesh, voxel_size, min_component_faces=min_component_faces
        )
    else:
        raise ValueError(f"Unknown method: {method}. Use 'auto', 'curvature', or 'voxel'.")

    print(f"  Segments found: {len(components)}")

    if not components:
        print("  No segments detected. Try adjusting parameters:")
        print("    --curvature-percentile (lower = more aggressive boundary)")
        print("    --voxel-size (smaller = finer resolution)")
        return

    teeth, connectors = classify_components(components)
    print(f"  → {len(teeth)} tooth/teeth, {len(connectors)} connector(s)")

    teeth_dir = os.path.join(output_dir, base, "teeth")
    conn_dir = os.path.join(output_dir, base, "connectors")

    if teeth:
        save_components(teeth, teeth_dir, f"{base}_tooth")
    if connectors:
        save_components(connectors, conn_dir, f"{base}_connector")

    print("Done.\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Segment connectors from dental STL files using curvature "
                    "analysis and/or voxel morphology."
    )
    parser.add_argument(
        "input",
        help="Input STL file or directory of STL files"
    )
    parser.add_argument(
        "--output-dir", default="segmented",
        help="Root directory for output STL files (default: segmented/)"
    )
    parser.add_argument(
        "--method", choices=["auto", "curvature", "voxel"], default="auto",
        help="Segmentation method (default: auto — tries curvature, then voxel)"
    )
    parser.add_argument(
        "--curvature-percentile", type=float, default=8.0,
        help="Percentile threshold for concave boundary detection (default: 8.0)"
    )
    parser.add_argument(
        "--voxel-size", type=float, default=None,
        help="Voxel edge length for morphological method (default: auto)"
    )
    parser.add_argument(
        "--min-faces", type=int, default=50,
        help="Minimum faces for a valid segment (default: 50)"
    )
    args = parser.parse_args()

    input_path = Path(args.input)

    if input_path.is_dir():
        stl_files = sorted(input_path.glob("*.stl"))
        print(f"Batch processing {len(stl_files)} STL files from {input_path}\n")
        for stl in stl_files:
            segment(
                str(stl),
                output_dir=args.output_dir,
                method=args.method,
                curvature_percentile=args.curvature_percentile,
                voxel_size=args.voxel_size,
                min_component_faces=args.min_faces,
            )
    elif input_path.is_file():
        segment(
            str(input_path),
            output_dir=args.output_dir,
            method=args.method,
            curvature_percentile=args.curvature_percentile,
            voxel_size=args.voxel_size,
            min_component_faces=args.min_faces,
        )
    else:
        raise FileNotFoundError(f"Input not found: {args.input}")


if __name__ == "__main__":
    main()
