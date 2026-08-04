"""
Dental STL — Outer Boundary Outline
------------------------------------
1. Load STL
2. Align mesh (translate to origin, PCA rotation: widest spread XY, least Z)
3. Sample mesh surface points and project onto the Z=0 (occlusal) plane
4. Outline outer boundary of projected points (alpha shape / concave hull)
5. Render outline as 3D curve

Usage:
    python3 segment2.py                      # launch GUI with no file
    python3 segment2.py stl/0325.stl         # launch GUI with file preloaded
    python3 segment2.py stl/0325.stl --alpha 1.5 --num-samples 10000
"""

import argparse
import sys

import numpy as np
import open3d as o3d
import trimesh
from scipy.spatial import Delaunay


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

    order = np.argsort(-eigenvalues)
    rotation = eigenvectors[:, order].T

    if np.linalg.det(rotation) < 0:
        rotation[2] *= -1

    mesh.vertices = mesh.vertices @ rotation.T
    return mesh


# ── 3. Project sampled mesh points onto the occlusal plane ───────────────

def project_mesh_to_plane(mesh: trimesh.Trimesh, num_samples: int = 10000) -> np.ndarray:
    """Sample points on the mesh surface and project them onto Z=0.
    Returns an Nx2 array of XY coordinates."""
    sampled, _ = trimesh.sample.sample_surface(mesh, num_samples)
    return sampled[:, :2]


# ── 4. Outline outer boundary (alpha shape) ──────────────────────────────

def alpha_shape_edges(points_2d: np.ndarray, alpha: float) -> list[tuple[int, int]]:
    """Compute the alpha shape of a 2D point set and return boundary edges
    as a list of (i, j) index pairs.

    *alpha* controls concavity: smaller = more concave, larger = more convex.
    If alpha is very large the result approaches the convex hull.
    """
    tri = Delaunay(points_2d)
    edges = set()
    edge_count = {}

    for simplex in tri.simplices:
        pts = points_2d[simplex]
        # Circumradius of the triangle
        a = np.linalg.norm(pts[0] - pts[1])
        b = np.linalg.norm(pts[1] - pts[2])
        c = np.linalg.norm(pts[2] - pts[0])
        s = (a + b + c) / 2.0
        area = np.sqrt(max(s * (s - a) * (s - b) * (s - c), 0))
        if area < 1e-12:
            continue
        circum_r = (a * b * c) / (4.0 * area)

        if circum_r < 1.0 / alpha:
            for i in range(3):
                edge = tuple(sorted((simplex[i], simplex[(i + 1) % 3])))
                edge_count[edge] = edge_count.get(edge, 0) + 1

    # Boundary edges appear exactly once in the filtered triangulation
    boundary_edges = [e for e, cnt in edge_count.items() if cnt == 1]
    return boundary_edges


def order_boundary(points_2d: np.ndarray, edges: list[tuple[int, int]]) -> np.ndarray:
    """Order boundary edges into a closed polygon. Returns Mx2 ordered points.
    If multiple loops exist, returns the longest one."""
    from collections import defaultdict

    adj = defaultdict(list)
    for i, j in edges:
        adj[i].append(j)
        adj[j].append(i)

    visited_edges = set()
    loops = []

    for start in adj:
        if start in visited_edges:
            continue
        loop = [start]
        visited_edges.add(start)
        current = start
        while True:
            neighbors = adj[current]
            next_node = None
            for n in neighbors:
                if n not in visited_edges:
                    next_node = n
                    break
            if next_node is None:
                break
            loop.append(next_node)
            visited_edges.add(next_node)
            current = next_node
        loops.append(loop)

    if not loops:
        return np.empty((0, 2))

    # Return the longest loop
    longest = max(loops, key=len)
    return points_2d[longest]


def compute_outline(points_2d: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Compute the outer boundary outline of 2D points.
    Returns Mx3 ordered boundary points with Z=0."""
    edges = alpha_shape_edges(points_2d, alpha)
    if not edges:
        # Fallback: use convex hull
        from scipy.spatial import ConvexHull
        hull = ConvexHull(points_2d)
        boundary_2d = points_2d[hull.vertices]
    else:
        boundary_2d = order_boundary(points_2d, edges)

    if len(boundary_2d) == 0:
        return np.empty((0, 3))

    # Close the loop
    boundary_3d = np.column_stack([boundary_2d, np.zeros(len(boundary_2d))])
    return boundary_3d


# ── 5. Render outline as 3D curve ─────────────────────────────────────────

def _trimesh_to_o3d(mesh: trimesh.Trimesh) -> o3d.geometry.TriangleMesh:
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
    o3d_mesh.compute_vertex_normals()
    return o3d_mesh


def make_mesh_geometry(mesh: trimesh.Trimesh, wireframe: bool = False):
    o3d_mesh = _trimesh_to_o3d(mesh)
    if wireframe:
        wf = o3d.geometry.LineSet.create_from_triangle_mesh(o3d_mesh)
        wf.paint_uniform_color([0.75, 0.75, 0.75])
        return wf
    o3d_mesh.paint_uniform_color([1.0, 1.0, 1.0])
    return o3d_mesh


def make_outline_geometry(outline: np.ndarray, radius: float = 0.15) -> o3d.geometry.TriangleMesh:
    """Render the outline as a tube mesh (closed loop)."""
    if len(outline) < 3:
        return o3d.geometry.TriangleMesh()

    # Close the loop by appending the first point
    closed = np.vstack([outline, outline[0:1]])

    circle_res = 12
    theta = np.linspace(0, 2 * np.pi, circle_res, endpoint=False)

    all_verts = []
    all_tris = []

    for i in range(len(closed) - 1):
        p0, p1 = closed[i], closed[i + 1]
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

    if not all_verts:
        return o3d.geometry.TriangleMesh()

    tube = o3d.geometry.TriangleMesh()
    tube.vertices = o3d.utility.Vector3dVector(np.array(all_verts))
    tube.triangles = o3d.utility.Vector3iVector(np.array(all_tris))
    tube.paint_uniform_color([0.0, 1.0, 0.3])
    tube.compute_vertex_normals()
    return tube


# ── GUI ───────────────────────────────────────────────────────────────────

class OutlineApp:
    """Open3D GUI application for computing and displaying the outer boundary."""

    def __init__(self, stl_path: str | None = None, alpha: float = 1.0,
                 num_samples: int = 10000, wireframe: bool = False):
        self._mesh: trimesh.Trimesh | None = None
        self._outline: np.ndarray | None = None

        gui = o3d.visualization.gui
        self._gui = gui

        self._app = gui.Application.instance
        self._app.initialize()

        # ── Window ──
        self._window = self._app.create_window("Outer Boundary Outline", 1400, 900)
        w = self._window

        # ── 3D Scene ──
        self._scene = gui.SceneWidget()
        self._scene.scene = o3d.visualization.rendering.Open3DScene(w.renderer)
        self._scene.scene.set_background([0.0, 0.0, 0.0, 1.0])

        # ── Settings panel ──
        em = w.theme.font_size
        self._panel = gui.Vert(0.5 * em, gui.Margins(0.5 * em, 0.5 * em,
                                                       0.5 * em, 0.5 * em))

        # Load mesh button
        load_btn = gui.Button("Load Mesh…")
        load_btn.set_on_clicked(self._on_load_mesh)
        self._panel.add_child(load_btn)

        self._file_label = gui.Label("No file loaded")
        self._panel.add_child(self._file_label)

        sep = gui.Label("─── Parameters ───")
        self._panel.add_child(sep)

        # Alpha
        self._panel.add_child(gui.Label("Alpha"))
        self._alpha_edit = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self._alpha_edit.double_value = alpha
        self._alpha_edit.set_limits(0.01, 100.0)
        self._panel.add_child(self._alpha_edit)

        # Num samples
        self._panel.add_child(gui.Label("Num samples"))
        self._num_samples_edit = gui.NumberEdit(gui.NumberEdit.INT)
        self._num_samples_edit.int_value = num_samples
        self._num_samples_edit.set_limits(100, 100000)
        self._panel.add_child(self._num_samples_edit)

        # Wireframe
        self._wireframe_cb = gui.Checkbox("Wireframe")
        self._wireframe_cb.checked = wireframe
        self._panel.add_child(self._wireframe_cb)

        # Show options
        self._show_mesh_cb = gui.Checkbox("Show mesh")
        self._show_mesh_cb.checked = True
        self._panel.add_child(self._show_mesh_cb)

        self._show_outline_cb = gui.Checkbox("Show outline")
        self._show_outline_cb.checked = True
        self._panel.add_child(self._show_outline_cb)

        self._show_points_cb = gui.Checkbox("Show projected points")
        self._show_points_cb.checked = False
        self._panel.add_child(self._show_points_cb)

        # Recompute button
        sep2 = gui.Label("")
        self._panel.add_child(sep2)
        recompute_btn = gui.Button("Recompute")
        recompute_btn.set_on_clicked(self._on_recompute)
        self._panel.add_child(recompute_btn)

        # Status
        self._status_label = gui.Label("")
        self._panel.add_child(self._status_label)

        # ── Layout ──
        w.set_on_layout(self._on_layout)
        w.add_child(self._scene)
        w.add_child(self._panel)

        # Load initial file if provided
        if stl_path:
            self._load_file(stl_path)

    def _on_layout(self, layout_context):
        r = self._window.content_rect
        panel_width = 220
        self._scene.frame = o3d.visualization.gui.Rect(
            r.x, r.y, r.width - panel_width, r.height)
        self._panel.frame = o3d.visualization.gui.Rect(
            r.get_right() - panel_width, r.y, panel_width, r.height)

    def _on_load_mesh(self):
        gui = self._gui
        dlg = gui.FileDialog(gui.FileDialog.OPEN, "Select STL file",
                             self._window.theme)
        dlg.add_filter(".stl", "STL files (.stl)")
        dlg.add_filter("", "All files")
        dlg.set_on_cancel(self._on_file_cancel)
        dlg.set_on_done(self._on_file_done)
        self._window.show_dialog(dlg)

    def _on_file_cancel(self):
        self._window.close_dialog()

    def _on_file_done(self, path):
        self._window.close_dialog()
        self._load_file(path)

    def _load_file(self, path: str):
        try:
            mesh = load_mesh(path)
            mesh = align_mesh(mesh)
            self._mesh = mesh
            self._file_label.text = path.split("/")[-1]
            self._rebuild_scene()
        except Exception as e:
            self._status_label.text = f"Error: {e}"

    def _on_recompute(self):
        if self._mesh is None:
            self._status_label.text = "No mesh loaded"
            return
        self._rebuild_scene()

    def _rebuild_scene(self):
        if self._mesh is None:
            return

        scene = self._scene.scene
        scene.clear_geometry()

        mesh = self._mesh
        alpha = self._alpha_edit.double_value
        num_samples = self._num_samples_edit.int_value
        wireframe = self._wireframe_cb.checked

        # Mesh
        if self._show_mesh_cb.checked:
            geom = make_mesh_geometry(mesh, wireframe=wireframe)
            mat = o3d.visualization.rendering.MaterialRecord()
            mat.shader = "defaultLit"
            mat.base_color = [1.0, 1.0, 1.0, 1.0]
            if wireframe:
                mat.shader = "unlitLine"
                mat.line_width = 1.0
            scene.add_geometry("mesh", geom, mat)

        # Project and compute outline
        pts_2d = project_mesh_to_plane(mesh, num_samples=num_samples)
        outline = compute_outline(pts_2d, alpha=alpha)
        self._outline = outline

        # Projected points
        if self._show_points_cb.checked:
            pts_3d = np.column_stack([pts_2d, np.zeros(len(pts_2d))])
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts_3d)
            pcd.paint_uniform_color([0.5, 0.5, 1.0])
            mat_pts = o3d.visualization.rendering.MaterialRecord()
            mat_pts.shader = "defaultUnlit"
            mat_pts.point_size = 2.0
            scene.add_geometry("projected_points", pcd, mat_pts)

        # Outline curve
        if self._show_outline_cb.checked and len(outline) > 2:
            tube = make_outline_geometry(outline)
            mat_o = o3d.visualization.rendering.MaterialRecord()
            mat_o.shader = "defaultLit"
            mat_o.base_color = [0.0, 1.0, 0.3, 1.0]
            scene.add_geometry("outline", tube, mat_o)

        self._status_label.text = f"{len(pts_2d)} pts, {len(outline)} boundary pts"

        # Fit camera
        bounds = scene.bounding_box
        self._scene.setup_camera(60.0, bounds, bounds.get_center())

    def run(self):
        self._app.run()


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Outer boundary outline of dental STL")
    parser.add_argument("stl", nargs="?", default=None,
                        help="Path to the input STL file (optional; can load from GUI)")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Alpha parameter for concave hull (default: 1.0)")
    parser.add_argument("--num-samples", type=int, default=10000,
                        help="Number of surface samples (default: 10000)")
    parser.add_argument("--wireframe", action="store_true",
                        help="Render mesh as wireframe")
    args = parser.parse_args()

    app = OutlineApp(
        stl_path=args.stl,
        alpha=args.alpha,
        num_samples=args.num_samples,
        wireframe=args.wireframe,
    )
    app.run()


if __name__ == "__main__":
    main()
