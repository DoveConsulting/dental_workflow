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
import colorsys
import os
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh
import networkx as nx


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
# Ray-cast thickness segmentation (fallback / alternative to curvature)
# ---------------------------------------------------------------------------

def _compute_local_thickness(mesh: trimesh.Trimesh) -> np.ndarray:
    """
    For each vertex, cast a ray inward (along -normal) and measure the
    distance to the first intersection with the opposite wall.  This gives
    the local structural diameter of the mesh at that point.

    Teeth (bulky) → large thickness.
    Connectors (thin bars) → small thickness.
    """
    origins = mesh.vertices.copy()
    directions = -mesh.vertex_normals.copy()

    # Offset origins slightly along the normal to avoid self-intersection
    origins += directions * (mesh.scale * 1e-4)

    locations, index_ray, _ = mesh.ray.intersects_location(
        ray_origins=origins,
        ray_directions=directions,
        multiple_hits=False,
    )

    thickness = np.zeros(len(mesh.vertices), dtype=float)
    for loc, ri in zip(locations, index_ray):
        d = np.linalg.norm(loc - mesh.vertices[ri])
        if d > mesh.scale * 0.001:  # skip near-self hits
            thickness[ri] = d

    return thickness


def segment_by_voxel_morphology(
    mesh: trimesh.Trimesh,
    voxel_size: float | None = None,
    erosion_iterations: int = 3,
    min_component_faces: int = 50,
) -> list[trimesh.Trimesh]:
    """
    Ray-cast thickness segmentation.

    For each vertex, cast a ray inward and measure the local structural
    diameter.  Connector vertices have small thickness (thin bars); tooth
    vertices have large thickness.  Vertices below a thickness percentile
    are treated as boundary vertices and removed from the face-adjacency
    graph, yielding connected components (same logic as the curvature
    method but using a different geometric criterion).
    """
    thickness = _compute_local_thickness(mesh)

    # Vertices with zero thickness = ray missed; assign median
    valid = thickness[thickness > 0]
    if len(valid) == 0:
        return []
    thickness[thickness == 0] = np.median(valid)

    # Smooth thickness over vertex neighbours to reduce noise
    smoothed = thickness.copy()
    from collections import defaultdict
    vert_adj = defaultdict(set)
    for f in mesh.faces:
        vert_adj[int(f[0])].update([int(f[1]), int(f[2])])
        vert_adj[int(f[1])].update([int(f[0]), int(f[2])])
        vert_adj[int(f[2])].update([int(f[0]), int(f[1])])
    for vi in range(len(mesh.vertices)):
        nbrs = vert_adj[vi]
        if nbrs:
            smoothed[vi] = 0.5 * thickness[vi] + 0.5 * np.mean(thickness[list(nbrs)])

    # Connector boundary = vertices with low thickness (bottom percentile)
    threshold = np.percentile(smoothed, 8.0)
    boundary_verts = set(np.where(smoothed < threshold)[0])

    # Build face adjacency graph excluding boundary faces
    faces = mesh.faces
    num_faces = len(faces)

    boundary_faces = set()
    for fi in range(num_faces):
        if any(int(v) in boundary_verts for v in faces[fi]):
            boundary_faces.add(fi)

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
# Visualization: colored point cloud snapshots
# ---------------------------------------------------------------------------

def _generate_distinct_colors(n: int) -> list[tuple[float, float, float]]:
    """Generate *n* visually distinguishable colours (maximally spaced hues)."""
    colors = []
    for i in range(n):
        hue = i / n
        # High saturation + value for vivid colours on black background
        r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 0.95)
        colors.append((r, g, b))
    return colors


def _trimesh_to_o3d(mesh: trimesh.Trimesh) -> o3d.geometry.TriangleMesh:
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
    o3d_mesh.compute_vertex_normals()
    return o3d_mesh


def snapshot_components(
    original_mesh: trimesh.Trimesh,
    components: list[trimesh.Trimesh],
    output_dir: str,
    base_name: str,
    image_width: int = 1920,
    image_height: int = 1080,
    point_density: int = 5000,
) -> list[str]:
    """
    Build a coloured point cloud (one colour per component) overlaid on the
    semi-transparent original mesh, then render snapshots from multiple
    viewpoints over a black background.

    Returns the list of saved image paths.
    """
    snap_dir = os.path.join(output_dir, base_name, "snapshots")
    os.makedirs(snap_dir, exist_ok=True)

    # --- build original mesh (dark grey, semi-transparent wireframe look) ---
    o3d_original = _trimesh_to_o3d(original_mesh)
    o3d_original.paint_uniform_color([1.0, 1.0, 1.0])

    # --- build coloured point cloud from components ---
    colors = _generate_distinct_colors(len(components))
    all_points = []
    all_colors = []

    for comp, color in zip(components, colors):
        n_samples = max(200, int(point_density * (comp.area / original_mesh.area)))
        pts, _ = trimesh.sample.sample_surface(comp, n_samples)
        all_points.append(pts)
        all_colors.append(np.tile(color, (len(pts), 1)))

    combined_pts = np.vstack(all_points)
    combined_colors = np.vstack(all_colors)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(combined_pts)
    pcd.colors = o3d.utility.Vector3dVector(combined_colors)

    # --- camera setup ---
    center = original_mesh.centroid
    scale = original_mesh.scale
    distance = scale * 1.6

    # Multiple viewpoints: front, back, left, right, top, bottom, 3/4 views
    views = {
        "front":       (0, 0, distance),
        "back":        (0, 0, -distance),
        "left":        (-distance, 0, 0),
        "right":       (distance, 0, 0),
        "top":         (0, distance, 0),
        "bottom":      (0, -distance, 0),
        "front_left":  (-distance * 0.7, distance * 0.5, distance * 0.7),
        "front_right": (distance * 0.7, distance * 0.5, distance * 0.7),
    }

    saved_paths = []

    renderer = o3d.visualization.rendering.OffscreenRenderer(
        image_width, image_height
    )
    renderer.scene.set_background([0.0, 0.0, 0.0, 1.0])  # black

    # Material for mesh
    mesh_mat = o3d.visualization.rendering.MaterialRecord()
    mesh_mat.shader = "defaultLitTransparency"
    mesh_mat.base_color = [1.0, 1.0, 1.0, 0.35]

    # Material for point cloud
    pcd_mat = o3d.visualization.rendering.MaterialRecord()
    pcd_mat.shader = "defaultUnlit"
    pcd_mat.point_size = 3.0

    renderer.scene.add_geometry("original_mesh", o3d_original, mesh_mat)
    renderer.scene.add_geometry("point_cloud", pcd, pcd_mat)

    # Lighting
    renderer.scene.scene.enable_sun_light(True)
    renderer.scene.scene.set_sun_light(
        [0.5, -1.0, -0.5], [1.0, 1.0, 1.0], 60000
    )

    for view_name, eye_offset in views.items():
        eye = center + np.array(eye_offset)
        up = np.array([0.0, 1.0, 0.0])
        # Avoid degenerate up vector for top/bottom views
        if view_name == "top":
            up = np.array([0.0, 0.0, -1.0])
        elif view_name == "bottom":
            up = np.array([0.0, 0.0, 1.0])

        renderer.setup_camera(60.0, center, eye, up)

        img = renderer.render_to_image()
        out_path = os.path.join(snap_dir, f"{base_name}_{view_name}.png")
        o3d.io.write_image(out_path, img)
        saved_paths.append(out_path)
        print(f"  Snapshot: {out_path}")

    return saved_paths


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

    # if teeth:
    #     save_components(teeth, teeth_dir, f"{base}_tooth")
    # if connectors:
    #     save_components(connectors, conn_dir, f"{base}_connector")

    # Render coloured point-cloud snapshots
    print("  Rendering snapshots …")
    # snapshot_components(mesh, components, output_dir, base)
    snapshot_components(mesh, connectors, output_dir, base)

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
