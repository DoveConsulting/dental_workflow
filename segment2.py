
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
    python segment2.py stl/0325.stl              # default 3 mm spacing
    python segment2.py stl/0325.stl --spacing 2.5
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
    num_eval: int = 500,
) -> np.ndarray:
    """Fit a parabolic arch through the cross-section centroids *pts*
    and return *num_eval* evenly sampled points (Nx3).

    The curve is guaranteed to have a single extremum (no wiggles)
    because it is a degree-2 polynomial in the arch's principal frame.
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

    # Sample uniformly along u
    u_min, u_max = u_coords.min(), u_coords.max()
    u_fine = np.linspace(u_min, u_max, num_eval)
    v_fine = np.polyval(coeffs, u_fine)

    # Transform back to world XY
    world_xy = centroid_2d + np.outer(u_fine, axis_u) + np.outer(v_fine, axis_v)
    curve = np.column_stack([world_xy, np.zeros(num_eval)])
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
    spacing_mm: float = 3.0,
    plane_half_size: float = 5.0,
) -> list[dict]:
    """Return a list of dicts with keys 'origin', 'normal', and 'corners'
    for planes placed every *spacing_mm* along the occlusal curve, oriented
    tangent to the curve (normal = tangent direction)."""
    arc = _cumulative_arc_length(curve)
    total_len = arc[-1]
    num_planes = max(1, int(total_len / spacing_mm))
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
    parser.add_argument("--spacing", type=float, default=3.0,
                        help="Distance (mm) between slicing planes (default: 3)")
    parser.add_argument("--plane-size", type=float, default=5.0,
                        help="Half-size of rendered slice planes in mm (default: 5)")
    parser.add_argument("--show", nargs="+", default=["mesh", "curve", "planes"],
                        choices=["mesh", "curve", "planes"],
                        help="Items to render (default: mesh curve planes)")
    parser.add_argument("--wireframe", action="store_true",
                        help="Render mesh as wireframe instead of solid")
    args = parser.parse_args()

    print(f"Loading {args.stl} …")
    mesh = load_mesh(args.stl)

    print("Aligning mesh …")
    mesh = align_mesh(mesh)

    print("Slicing at Z=0 …")
    pts = slice_at_z0(mesh)
    print(f"  → {len(pts)} intersection points")

    print("Fitting occlusal curve …")
    curve = fit_occlusal_curve(pts)

    print(f"Creating slicing planes (every {args.spacing} mm) …")
    planes = create_slicing_planes(curve, spacing_mm=args.spacing,
                                   plane_half_size=args.plane_size)
    print(f"  → {len(planes)} planes")

    print("Rendering …")
    items = []
    if "mesh" in args.show:
        items.append(make_mesh_geometry(mesh, wireframe=args.wireframe))
    if "curve" in args.show:
        items.append(make_curve_geometry(curve))
    # if "planes" in args.show:
    #     items.append(make_planes_geometry(planes))
    render(*items)


if __name__ == "__main__":
    main()

