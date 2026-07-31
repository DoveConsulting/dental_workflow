
"""
Dental STL — Occlusal-plane slicing
------------------------------------
1. Load STL
2. Align mesh (translate to origin, PCA rotation: widest spread XY, least Z)
3. Slice at Z=0 to get the arch intersection curve
4. Fit a 3D spline to those intersection points (occlusal curve)
5. Create equally spaced slicing planes tangent to the curve
6. Render mesh + slicing planes with Open3D

Usage:
    python segment.py stl/0325.stl
    python segment.py stl/0325.stl --spacing 2.0
    python segment.py stl/0325.stl --spacing 1.5 --plane-size 8 --wireframe
    python segment.py stl/0325.stl --threshold-ratio 0.6 --skip-extremes
    python segment.py stl/0325.stl --show mesh curve

Arguments:
    stl                 Path to the input STL file
    --spacing           Distance between slicing planes (default: 0.02)
    --plane-size        Half-size of rendered slice planes (default: 5.0)
    --show              Items to render: mesh, curve, planes (default: mesh curve planes)
    --wireframe         Render mesh as wireframe instead of solid surface
    --threshold-ratio   Z-extent ratio below which a plane is classified as a
                        connector region (default: 0.75)
    --skip-extremes     Discard connector groups that touch the first or last plane
"""

import argparse
import sys

import numpy as np
import open3d as o3d
import trimesh
from scipy.interpolate import splprep, splev


# ── 1. Load STL ──────────────────────────────────────────────────────────

def load_mesh(filepath: str) -> trimesh.Trimesh:
    mesh = trimesh.load(filepath, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh.dump())
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Could not load a triangle mesh from {filepath}")
    return mesh


# ── 2. Align mesh ────────────────────────────────────────────────────────

def align_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Translate centroid to origin, then PCA-rotate so the widest spread
    falls on X, second-widest on Y, and the thinnest on Z."""
    centroid = mesh.vertices.mean(axis=0)
    mesh.vertices -= centroid

    cov = np.cov(mesh.vertices, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # eigh returns eigenvalues in ascending order → flip so largest first
    order = np.argsort(-eigenvalues)
    rotation = eigenvectors[:, order].T          # rows = new axes

    # Ensure a right-handed coordinate system
    if np.linalg.det(rotation) < 0:
        rotation[2] *= -1

    mesh.vertices = mesh.vertices @ rotation.T
    return mesh


# ── 3. Slice at Z=0 ──────────────────────────────────────────────────────

def slice_at_z0(mesh: trimesh.Trimesh) -> np.ndarray:
    """Slice the mesh at Z=0 and return the centroid of each discrete
    cross-section loop as an Nx3 array (one point per tooth outline)."""
    section = mesh.section(plane_origin=[0, 0, 0], plane_normal=[0, 0, 1])
    if section is None:
        raise RuntimeError("Mesh does not intersect the Z=0 plane")

    points_2d, _ = section.to_planar()

    centroids = []
    for entity in points_2d.entities:
        pts = points_2d.vertices[entity.points]
        centroids.append(pts.mean(axis=0))

    centroids = np.array(centroids)
    pts_3d = np.column_stack([
        centroids[:, 0],
        centroids[:, 1],
        np.zeros(len(centroids)),
    ])
    return pts_3d


# ── 4. Fit 3D spline (occlusal curve) ────────────────────────────────────

def fit_occlusal_curve(
    pts: np.ndarray,
    mesh_vertices: np.ndarray | None = None,
    num_eval: int = 500,
) -> np.ndarray:
    """Fit a parabolic arch through the cross-section centroids *pts*
    and return *num_eval* evenly sampled points (Nx3).

    If *mesh_vertices* is provided the curve is extended to span the full
    mesh extent along the arch axis, otherwise it only covers the centroids.
    """
    xy = pts[:, :2]
    centroid_2d = xy.mean(axis=0)
    xy_c = xy - centroid_2d

    # PCA to find arch's main axis
    cov = np.cov(xy_c, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Principal axis = largest eigenvalue (along the arch span)
    order = np.argsort(-eigvals)
    axis_u = eigvecs[:, order[0]]  # along the arch
    axis_v = eigvecs[:, order[1]]  # perpendicular

    # Project centroids into PCA frame
    u_coords = xy_c @ axis_u
    v_coords = xy_c @ axis_v

    # Fit parabola: v = a*u² + b*u + c  →  single extremum
    coeffs = np.polyfit(u_coords, v_coords, 2)

    # Determine u range — use mesh extent if available
    if mesh_vertices is not None:
        mesh_xy_c = mesh_vertices[:, :2] - centroid_2d
        mesh_u = mesh_xy_c @ axis_u
        u_min, u_max = mesh_u.min(), mesh_u.max()
    else:
        u_min, u_max = u_coords.min(), u_coords.max()

    u_fine = np.linspace(u_min, u_max, num_eval)
    v_fine = np.polyval(coeffs, u_fine)

    # Transform back to world XY
    world_xy = centroid_2d + np.outer(u_fine, axis_u) + np.outer(v_fine, axis_v)
    curve = np.column_stack([world_xy, np.zeros(num_eval)])

    # Trim curve to mesh extents
    if mesh_vertices is not None:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(mesh_vertices[:, :2])
        from scipy.spatial import Delaunay
        hull_del = Delaunay(mesh_vertices[hull.vertices, :2])
        inside = hull_del.find_simplex(curve[:, :2]) >= 0
        # Keep the longest contiguous run of inside points
        if inside.any():
            diffs = np.diff(np.concatenate([[0], inside.astype(int), [0]]))
            starts = np.where(diffs == 1)[0]
            ends = np.where(diffs == -1)[0]
            lengths = ends - starts
            best = np.argmax(lengths)
            curve = curve[starts[best]:ends[best]]
    return curve


def _order_along_arch(pts: np.ndarray) -> np.ndarray:
    """Order points by greedy nearest-neighbour walk, starting from the
    point farthest from the centroid (an arch endpoint)."""
    centroid = pts.mean(axis=0)
    dists_to_center = np.linalg.norm(pts - centroid, axis=1)
    start = int(np.argmax(dists_to_center))

    n = len(pts)
    visited = np.zeros(n, dtype=bool)
    order = [start]
    visited[start] = True

    for _ in range(n - 1):
        cur = order[-1]
        dists = np.linalg.norm(pts - pts[cur], axis=1)
        dists[visited] = np.inf
        nxt = int(np.argmin(dists))
        order.append(nxt)
        visited[nxt] = True

    return pts[order]


# ── 5. Create equally spaced slicing planes tangent to occlusal curve ────

def _cumulative_arc_length(curve: np.ndarray) -> np.ndarray:
    diffs = np.diff(curve, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    return np.concatenate([[0], np.cumsum(seg_lens)])

def create_slicing_planes(
    curve: np.ndarray,
    spacing: float = 0.02,
    plane_half_size: float = 5.0,
) -> list[dict]:
    """Return a list of dicts with keys 'origin', 'normal', and 'corners'
    for planes placed every *spacing* along the occlusal curve, oriented
    tangent to the curve (normal = tangent direction)."""
    arc = _cumulative_arc_length(curve)
    total_len = arc[-1]
    num_planes = max(1, int(total_len / spacing))
    target_dists = np.linspace(0, total_len, num_planes, endpoint=False)

    planes = []
    n_pts = len(curve)
    for d in target_dists:
        idx = np.searchsorted(arc, d, side="right") - 1
        idx = np.clip(idx, 0, n_pts - 2)

        # Interpolate position
        frac = (d - arc[idx]) / max(arc[idx + 1] - arc[idx], 1e-12)
        origin = curve[idx] + frac * (curve[idx + 1] - curve[idx])

        # Tangent ≈ local finite difference
        tangent = curve[(idx + 1) % n_pts] - curve[(idx - 1) % n_pts]
        tangent /= np.linalg.norm(tangent) + 1e-12

        # Build a local frame:  normal = tangent,  u = up×tangent,  v = tangent×u
        up = np.array([0.0, 0.0, 1.0])
        u = np.cross(up, tangent)
        u_norm = np.linalg.norm(u)
        if u_norm < 1e-6:
            up = np.array([0.0, 1.0, 0.0])
            u = np.cross(up, tangent)
            u_norm = np.linalg.norm(u)
        u /= u_norm
        v = np.cross(tangent, u)

        h = plane_half_size
        corners = np.array([
            origin - h * u - h * v,
            origin + h * u - h * v,
            origin + h * u + h * v,
            origin - h * u + h * v,
        ])

        planes.append({"origin": origin, "normal": tangent, "corners": corners})

    return planes


# ── 5b. Connector detection ──────────────────────────────────────────────

def find_connectors(
    mesh: trimesh.Trimesh,
    planes: list[dict],
    threshold_ratio: float = 0.75,
    skip_extremes: bool = False,
) -> list[list[dict]]:
    """Identify connector regions: consecutive runs of planes where the
    Z-extent of the mesh/plane intersection falls below
    *threshold_ratio* × median Z-extent.

    If *skip_extremes* is True, connector groups that touch the first or
    last plane are discarded (they are typically mesh boundary artifacts,
    not real connectors).

    Returns a list of connector groups, each group being a list of
    consecutive planes that form one connector region.
    """
    z_extents = []
    for pl in planes:
        section = mesh.section(
            plane_origin=pl["origin"],
            plane_normal=pl["normal"],
        )
        if section is None:
            z_extents.append(0.0)
            continue
        verts = section.vertices
        z_extent = verts[:, 2].max() - verts[:, 2].min()
        z_extents.append(z_extent)

    z_extents = np.array(z_extents)
    median_z = np.median(z_extents[z_extents > 0]) if np.any(z_extents > 0) else 1.0
    threshold = threshold_ratio * median_z

    # Find connector plane indices and group consecutive runs
    is_connector = [(0 < ze < threshold) for ze in z_extents]
    groups: list[list[dict]] = []
    current_group: list[int] = []
    for i, flag in enumerate(is_connector):
        if flag:
            current_group.append(i)
        else:
            if current_group:
                groups.append(current_group)
                current_group = []
    if current_group:
        groups.append(current_group)

    if skip_extremes:
        n = len(planes)
        groups = [g for g in groups if g[0] != 0 and g[-1] != n - 1]

    # Convert index groups to plane groups
    return [[planes[i] for i in g] for g in groups]


def make_connectors_geometry(
    mesh: trimesh.Trimesh,
    connector_groups: list[list[dict]],
    num_samples: int = 5000,
) -> o3d.geometry.PointCloud:
    """Sample points on the mesh within each connector region and return
    them as a green point cloud."""
    # For each connector group, define the slab between first and last plane
    connector_points = []
    for group in connector_groups:
        origin_first = group[0]["origin"]
        normal_first = group[0]["normal"]
        origin_last = group[-1]["origin"]
        normal_last = group[-1]["normal"]

        # A vertex is inside this slab if it's between the two bounding planes
        # (on the positive side of the first plane and negative side of the last)
        d_first = mesh.vertices @ normal_first - origin_first @ normal_first
        d_last = mesh.vertices @ normal_last - origin_last @ normal_last

        # Vertices between the two planes
        inside = (d_first >= 0) & (d_last <= 0)
        if not inside.any():
            # Try flipping
            inside = (d_first <= 0) & (d_last >= 0)

        if inside.any():
            connector_points.append(mesh.vertices[inside])

    if not connector_points:
        pcd = o3d.geometry.PointCloud()
        return pcd

    all_pts = np.vstack(connector_points)

    # Also sample surface points in connector regions for denser coverage
    sampled, face_idx = trimesh.sample.sample_surface(mesh, num_samples)
    # Filter sampled points to those within any connector slab
    keep = np.zeros(len(sampled), dtype=bool)
    for group in connector_groups:
        origin_first = group[0]["origin"]
        normal_first = group[0]["normal"]
        origin_last = group[-1]["origin"]
        normal_last = group[-1]["normal"]

        d_first = sampled @ normal_first - origin_first @ normal_first
        d_last = sampled @ normal_last - origin_last @ normal_last

        slab = (d_first >= 0) & (d_last <= 0)
        if not slab.any():
            slab = (d_first <= 0) & (d_last >= 0)
        keep |= slab

    if keep.any():
        all_pts = np.vstack([all_pts, sampled[keep]])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_pts)
    pcd.paint_uniform_color([0.0, 1.0, 0.3])
    return pcd


# ── 6. Render with Open3D ─────────────────────────────────────────────────

def _trimesh_to_o3d(mesh: trimesh.Trimesh) -> o3d.geometry.TriangleMesh:
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
    o3d_mesh.compute_vertex_normals()
    return o3d_mesh

def make_mesh_geometry(mesh: trimesh.Trimesh, wireframe: bool = False):
    """Return the mesh as a solid white surface or a wireframe."""
    o3d_mesh = _trimesh_to_o3d(mesh)
    if wireframe:
        wf = o3d.geometry.LineSet.create_from_triangle_mesh(o3d_mesh)
        wf.paint_uniform_color([0.75, 0.75, 0.75])
        return wf
    o3d_mesh.paint_uniform_color([1.0, 1.0, 1.0])
    return o3d_mesh

def make_curve_geometry(curve: np.ndarray, radius: float = 0.15) -> o3d.geometry.TriangleMesh:
    """Return the curve as a tube mesh so it renders with visible thickness."""
    circle_res = 12
    theta = np.linspace(0, 2 * np.pi, circle_res, endpoint=False)

    all_verts = []
    all_tris = []

    for i in range(len(curve) - 1):
        p0, p1 = curve[i], curve[i + 1]
        tangent = p1 - p0
        length = np.linalg.norm(tangent)
        if length < 1e-12:
            continue
        tangent /= length

        up = np.array([0.0, 0.0, 1.0])
        perp = np.cross(up, tangent)
        pn = np.linalg.norm(perp)
        if pn < 1e-6:
            up = np.array([0.0, 1.0, 0.0])
            perp = np.cross(up, tangent)
            pn = np.linalg.norm(perp)
        perp /= pn
        binorm = np.cross(tangent, perp)

        ring0 = p0 + radius * (np.cos(theta)[:, None] * perp + np.sin(theta)[:, None] * binorm)
        ring1 = p1 + radius * (np.cos(theta)[:, None] * perp + np.sin(theta)[:, None] * binorm)

        base = len(all_verts)
        all_verts.extend(ring0)
        all_verts.extend(ring1)
        for j in range(circle_res):
            j1 = (j + 1) % circle_res
            v0, v1 = base + j, base + j1
            v2, v3 = base + circle_res + j, base + circle_res + j1
            all_tris.append([v0, v2, v1])
            all_tris.append([v1, v2, v3])

    tube = o3d.geometry.TriangleMesh()
    tube.vertices = o3d.utility.Vector3dVector(np.array(all_verts))
    tube.triangles = o3d.utility.Vector3iVector(np.array(all_tris))
    tube.paint_uniform_color([0.0, 0.3, 1.0])
    tube.compute_vertex_normals()
    return tube

def make_planes_geometry(planes: list[dict]) -> list[o3d.geometry.TriangleMesh]:
    quads = []
    for pl in planes:
        c = pl["corners"]
        quad = o3d.geometry.TriangleMesh()
        quad.vertices = o3d.utility.Vector3dVector(c)
        quad.triangles = o3d.utility.Vector3iVector([[0, 1, 2], [0, 2, 3]])
        quad.paint_uniform_color([1.0, 0.2, 0.2])
        quad.compute_vertex_normals()
        quads.append(quad)
    return quads

def render(*geometries) -> None:
    """Render any number of Open3D geometry objects with transparency support."""
    flat = []
    for g in geometries:
        if isinstance(g, list):
            flat.extend(g)
        else:
            flat.append(g)

    if not flat:
        print("Nothing to render.")
        return

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Occlusal Plane Slicing", width=1280, height=800)

    for g in flat:
        vis.add_geometry(g)

    opt = vis.get_render_option()
    opt.mesh_show_back_face = True
    opt.background_color = np.array([0.0, 0.0, 0.0])

    vis.run()
    vis.destroy_window()


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Occlusal-plane slicing of dental STL")
    parser.add_argument("stl", help="Path to the input STL file")
    parser.add_argument("--spacing", type=float, default=0.02,
                        help="Distance between slicing planes (default: 0.02)")
    parser.add_argument("--plane-size", type=float, default=5.0,
                        help="Half-size of rendered slice planes (default: 5)")
    parser.add_argument("--show", nargs="+", default=["mesh", "curve", "planes"],
                        choices=["mesh", "curve", "planes"],
                        help="Items to render (default: mesh curve planes)")
    parser.add_argument("--wireframe", action="store_true",
                        help="Render mesh as wireframe instead of solid")
    parser.add_argument("--threshold-ratio", type=float, default=0.75,
                        help="Z-extent ratio below which a plane is a connector (default: 0.75)")
    parser.add_argument("--skip-extremes", action="store_true",
                        help="Discard connector groups at the first/last plane")
    args = parser.parse_args()

    print(f"Loading {args.stl} …")
    mesh = load_mesh(args.stl)

    print("Aligning mesh …")
    mesh = align_mesh(mesh)

    print("Slicing at Z=0 …")
    pts = slice_at_z0(mesh)
    print(f"  → {len(pts)} intersection points")

    print("Fitting occlusal curve …")
    curve = fit_occlusal_curve(pts, mesh_vertices=mesh.vertices)

    print(f"Creating slicing planes (every {args.spacing}) …")
    planes = create_slicing_planes(curve, spacing=args.spacing,
                                   plane_half_size=args.plane_size)
    print(f"  → {len(planes)} planes")

    print("Detecting connectors …")
    connectors = find_connectors(mesh, planes,
                                  threshold_ratio=args.threshold_ratio,
                                  skip_extremes=args.skip_extremes)
    print(f"  → {len(connectors)} connector regions detected")

    print("Rendering …")
    items = []
    if "mesh" in args.show:
        items.append(make_mesh_geometry(mesh, wireframe=args.wireframe))
    if "curve" in args.show:
        items.append(make_curve_geometry(curve))
    # if "planes" in args.show:
    #     items.append(make_planes_geometry(planes))
    if connectors:
        items.append(make_connectors_geometry(mesh, connectors))
    render(*items)


if __name__ == "__main__":
    main()

