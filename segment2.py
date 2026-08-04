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
from scipy.ndimage import binary_dilation, binary_erosion


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


# ── 4. Outline outer boundary ─────────────────────────────────────────────

def compute_mesh_outline(
    mesh: trimesh.Trimesh,
    grid_res: float = 0.1,
) -> np.ndarray:
    """Compute the Z-projection outline by rasterising all mesh vertices
    onto a 2D grid, closing small gaps, and tracing the outer contour.

    *grid_res* is the cell size (mm).  Smaller = more detail, larger = smoother.

    Returns an Mx3 array of ordered boundary points with Z=0.
    """
    pts_2d = mesh.vertices[:, :2]

    pad = grid_res * 3
    xmin, ymin = pts_2d.min(axis=0) - pad
    xmax, ymax = pts_2d.max(axis=0) + pad

    nx = int(np.ceil((xmax - xmin) / grid_res)) + 1
    ny = int(np.ceil((ymax - ymin) / grid_res)) + 1

    # Rasterise vertices onto the grid
    xi = ((pts_2d[:, 0] - xmin) / grid_res).astype(int)
    yi = ((pts_2d[:, 1] - ymin) / grid_res).astype(int)
    xi = np.clip(xi, 0, nx - 1)
    yi = np.clip(yi, 0, ny - 1)

    grid = np.zeros((ny, nx), dtype=bool)
    grid[yi, xi] = True

    # Morphological close to fill small interior gaps
    struct = np.ones((3, 3), dtype=bool)
    grid = binary_dilation(grid, structure=struct, iterations=2)
    grid = binary_erosion(grid, structure=struct, iterations=2)

    # Boundary = filled cells with at least one empty 4-neighbour
    interior = binary_erosion(grid, structure=np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=bool))
    boundary = grid & ~interior

    by, bx = np.where(boundary)
    if len(bx) == 0:
        return np.empty((0, 3))

    boundary_xy = np.column_stack([
        xmin + bx * grid_res,
        ymin + by * grid_res,
    ])

    # Order boundary points by nearest-neighbour walk
    ordered = _order_nearest_neighbour(boundary_xy)
    return np.column_stack([ordered, np.zeros(len(ordered))])


def _order_nearest_neighbour(pts: np.ndarray) -> np.ndarray:
    """Order 2D points into a loop via greedy nearest-neighbour walk."""
    from scipy.spatial import cKDTree

    n = len(pts)
    if n < 3:
        return pts

    tree = cKDTree(pts)
    visited = np.zeros(n, dtype=bool)

    # Start from the point with smallest x
    start = int(np.argmin(pts[:, 0]))
    order = [start]
    visited[start] = True

    for _ in range(n - 1):
        cur = order[-1]
        # Query enough neighbours to find an unvisited one
        k = min(32, n)
        dists, idxs = tree.query(pts[cur], k=k)
        found = False
        for idx in idxs:
            if not visited[idx]:
                order.append(int(idx))
                visited[idx] = True
                found = True
                break
        if not found:
            # Brute-force fallback
            d = np.linalg.norm(pts - pts[cur], axis=1)
            d[visited] = np.inf
            nxt = int(np.argmin(d))
            order.append(nxt)
            visited[nxt] = True

    return pts[order]


# ── 5. Detect narrowing regions ───────────────────────────────────────────

def find_narrowings(
    outline: np.ndarray,
    width_ratio: float = 0.5,
    min_arc_gap: float = 0.25,
    smooth_window: int = 5,
) -> list[dict]:
    """Find narrowing regions along the closed outline curve.

    For each outline point, the "local width" is the distance to the closest
    point on the opposite side of the curve (excluding neighbours within
    *min_arc_gap* fraction of the total arc length on each side).

    A narrowing is a local minimum of the width profile whose value is below
    *width_ratio* × the median width.

    Returns a list of dicts with keys:
        'index'  – index into *outline*
        'point'  – the outline point at the narrowing
        'opposite' – the closest opposing outline point
        'width'  – the local width value
    """
    n = len(outline)
    if n < 6:
        return []

    # Cumulative arc length (closed loop)
    diffs = np.diff(outline, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    # Add closing segment
    close_len = np.linalg.norm(outline[0] - outline[-1])
    all_lens = np.append(seg_lens, close_len)
    arc = np.concatenate([[0], np.cumsum(all_lens)])
    total_arc = arc[-1]
    gap = min_arc_gap * total_arc

    # For each point, find the closest non-adjacent point
    widths = np.full(n, np.inf)
    closest_idx = np.zeros(n, dtype=int)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # Arc distance along the curve (shorter path around the loop)
            d_arc_fwd = abs(arc[j] - arc[i])
            d_arc = min(d_arc_fwd, total_arc - d_arc_fwd)
            if d_arc < gap:
                continue
            d_eucl = np.linalg.norm(outline[i] - outline[j])
            if d_eucl < widths[i]:
                widths[i] = d_eucl
                closest_idx[i] = j

    # Replace inf with 0 (points with no valid opposite)
    widths[widths == np.inf] = 0

    # Smooth the width profile
    if smooth_window > 1 and n > smooth_window:
        kernel = np.ones(smooth_window) / smooth_window
        # Pad for circular smoothing
        padded = np.concatenate([widths[-smooth_window:], widths, widths[:smooth_window]])
        smoothed = np.convolve(padded, kernel, mode='same')
        widths_smooth = smoothed[smooth_window:smooth_window + n]
    else:
        widths_smooth = widths.copy()

    # Threshold: narrowings must be below width_ratio × median
    median_w = np.median(widths_smooth[widths_smooth > 0]) if np.any(widths_smooth > 0) else 1.0
    threshold = width_ratio * median_w

    # Find local minima of smoothed width profile
    narrowings = []
    for i in range(n):
        if widths_smooth[i] <= 0 or widths_smooth[i] > threshold:
            continue
        prev_idx = (i - 1) % n
        next_idx = (i + 1) % n
        if widths_smooth[i] <= widths_smooth[prev_idx] and widths_smooth[i] <= widths_smooth[next_idx]:
            narrowings.append({
                'index': i,
                'point': outline[i],
                'opposite': outline[closest_idx[i]],
                'width': float(widths_smooth[i]),
            })

    # Deduplicate: if two narrowings point to each other, keep the one with
    # the smaller width
    if len(narrowings) > 1:
        deduped = []
        used_pairs = set()
        for nr in narrowings:
            pair = tuple(sorted((nr['index'], int(np.argmin(
                np.linalg.norm(outline - nr['opposite'], axis=1))))))
            if pair not in used_pairs:
                used_pairs.add(pair)
                deduped.append(nr)
        narrowings = deduped

    return narrowings


def make_narrowing_geometry(narrowings: list[dict]) -> o3d.geometry.LineSet:
    """Return a LineSet with lines drawn across each narrowing region."""
    points = []
    lines = []
    for nr in narrowings:
        idx = len(points)
        points.append(nr['point'])
        points.append(nr['opposite'])
        lines.append([idx, idx + 1])

    ls = o3d.geometry.LineSet()
    if points:
        ls.points = o3d.utility.Vector3dVector(np.array(points))
        ls.lines = o3d.utility.Vector2iVector(np.array(lines))
        ls.paint_uniform_color([1.0, 0.2, 0.2])
    return ls


# ── 6. Render outline as 3D curve ─────────────────────────────────────────

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

    def __init__(self, stl_path: str | None = None, grid_res: float = 0.1,
                 num_samples: int = 10000, wireframe: bool = False,
                 width_ratio: float = 0.5, min_arc_gap: float = 0.25):
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

        # Grid resolution
        self._panel.add_child(gui.Label("Grid resolution"))
        self._grid_res_edit = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self._grid_res_edit.double_value = grid_res
        self._grid_res_edit.set_limits(0.05, 5.0)
        self._panel.add_child(self._grid_res_edit)

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
        self._show_points_cb.checked = True
        self._panel.add_child(self._show_points_cb)

        self._show_plane_cb = gui.Checkbox("Show Z=0 plane")
        self._show_plane_cb.checked = True
        self._panel.add_child(self._show_plane_cb)

        self._show_narrowings_cb = gui.Checkbox("Show narrowings")
        self._show_narrowings_cb.checked = True
        self._panel.add_child(self._show_narrowings_cb)

        sep3 = gui.Label("─── Narrowing ───")
        self._panel.add_child(sep3)

        # Width ratio
        self._panel.add_child(gui.Label("Width ratio"))
        self._width_ratio_edit = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self._width_ratio_edit.double_value = width_ratio
        self._width_ratio_edit.set_limits(0.01, 2.0)
        self._panel.add_child(self._width_ratio_edit)

        # Min arc gap
        self._panel.add_child(gui.Label("Min arc gap"))
        self._min_arc_gap_edit = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self._min_arc_gap_edit.double_value = min_arc_gap
        self._min_arc_gap_edit.set_limits(0.05, 0.49)
        self._panel.add_child(self._min_arc_gap_edit)

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
        grid_res = self._grid_res_edit.double_value
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

        # Z=0 reference plane
        if self._show_plane_cb.checked:
            margin = 2.0
            xmin, ymin = mesh.vertices[:, 0].min() - margin, mesh.vertices[:, 1].min() - margin
            xmax, ymax = mesh.vertices[:, 0].max() + margin, mesh.vertices[:, 1].max() + margin
            corners = np.array([
                [xmin, ymin, 0], [xmax, ymin, 0],
                [xmax, ymax, 0], [xmin, ymax, 0],
            ])
            plane_mesh = o3d.geometry.TriangleMesh()
            plane_mesh.vertices = o3d.utility.Vector3dVector(corners)
            plane_mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2], [0, 2, 3]])
            plane_mesh.compute_vertex_normals()
            mat_plane = o3d.visualization.rendering.MaterialRecord()
            mat_plane.shader = "defaultLitTransparency"
            mat_plane.base_color = [1.0, 1.0, 0.0, 0.25]
            scene.add_geometry("z0_plane", plane_mesh, mat_plane)

        # Compute outline from rasterised vertex projection
        outline = compute_mesh_outline(mesh, grid_res=grid_res)
        self._outline = outline

        # Projected points (optional visualisation)
        if self._show_points_cb.checked:
            pts_2d = project_mesh_to_plane(mesh, num_samples=num_samples)
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

        # Narrowing lines
        if self._show_narrowings_cb.checked and len(outline) > 5:
            width_ratio = self._width_ratio_edit.double_value
            min_arc_gap = self._min_arc_gap_edit.double_value
            narrowings = find_narrowings(outline, width_ratio=width_ratio,
                                         min_arc_gap=min_arc_gap)
            ls = make_narrowing_geometry(narrowings)
            if len(narrowings) > 0:
                mat_n = o3d.visualization.rendering.MaterialRecord()
                mat_n.shader = "unlitLine"
                mat_n.line_width = 3.0
                scene.add_geometry("narrowings", ls, mat_n)
            self._status_label.text = (
                f"{len(outline)} boundary, "
                f"{len(narrowings)} narrowings")
        else:
            self._status_label.text = f"{len(outline)} boundary pts"

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
    parser.add_argument("--grid-res", type=float, default=0.1,
                        help="Grid cell size for outline extraction in mm (default: 0.1)")
    parser.add_argument("--num-samples", type=int, default=10000,
                        help="Number of surface samples (default: 10000)")
    parser.add_argument("--wireframe", action="store_true",
                        help="Render mesh as wireframe")
    parser.add_argument("--width-ratio", type=float, default=0.9,
                        help="Width ratio threshold for narrowings (default: 0.9)")
    parser.add_argument("--min-arc-gap", type=float, default=0.1,
                        help="Min arc-length fraction to skip neighbours (default: 0.1)")
    args = parser.parse_args()

    app = OutlineApp(
        stl_path=args.stl,
        grid_res=args.grid_res,
        num_samples=args.num_samples,
        wireframe=args.wireframe,
        width_ratio=args.width_ratio,
        min_arc_gap=args.min_arc_gap,
    )
    app.run()


if __name__ == "__main__":
    main()
