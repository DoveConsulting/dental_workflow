
"""
Dental STL — Occlusal-plane slicing
------------------------------------
1. Load STL
2. Align mesh (translate to origin, PCA rotation: widest spread XY, least Z)
3. Sample mesh surface points and project onto the Z=0 (occlusal) plane
4. Fit a parabolic arch through the projected points (occlusal curve)
5. Create equally spaced slicing planes tangent to the curve
6. Render mesh + slicing planes with Open3D

Usage:
    python3 segment.py                           # launch GUI with no file
    python3 segment.py stl/0325.stl              # launch GUI with file preloaded
    python3 segment.py stl/0325.stl --spacing 2.0
    python3 segment.py stl/0325.stl --spacing 1.5 --plane-size 8 --wireframe
    python3 segment.py stl/0325.stl --threshold-ratio 0.6 --skip-extremes

Arguments:
    stl                 Path to the input STL file (optional; can load from GUI)
    --spacing           Distance between slicing planes (default: 0.2)
    --plane-size        Half-size of rendered slice planes (default: 5.0)
    --wireframe         Render mesh as wireframe instead of solid surface
    --threshold-ratio   Z-extent ratio below which a plane is classified as a
                        connector region (default: 0.85)
    --no-skip-extremes  Keep connector groups that touch the first or last plane
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


# ── 3. Project sampled mesh points onto the occlusal plane ───────────────

def project_mesh_to_plane(mesh: trimesh.Trimesh, num_samples: int = 5000) -> np.ndarray:
    """Sample points on the mesh surface and project them onto the Z=0
    (occlusal) plane.  Returns an Nx3 array with Z=0."""
    sampled, _ = trimesh.sample.sample_surface(mesh, num_samples)
    pts_3d = np.column_stack([sampled[:, 0], sampled[:, 1],
                              np.zeros(len(sampled))])
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
    spacing: float = 0.2,
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
    threshold_ratio: float = 0.85,
    skip_extremes: bool = True,
    neighbor_distance: float = 10.0,
    height_method: str = "max",
    max_connector_width: float = 1.0,
) -> list[list[dict]]:
    """Identify connector regions: consecutive runs of planes where the
    maximum vertical distance from the cross-section to the occlusal curve
    falls below *threshold_ratio* × the local reference distance.

    *height_method* controls how the local reference is computed from
    neighboring planes: "median" uses the median, "max" uses the maximum.

    The local median is computed from neighboring planes within
    *neighbor_distance* units along the curve (equally divided left/right).

    If *skip_extremes* is True (default), connector groups that touch the
    first or last plane are discarded (they are typically mesh boundary
    artifacts, not real connectors).

    Returns a list of connector groups, each group being a list of
    consecutive planes that form one connector region.
    """
    distances = []
    for pl in planes:
        section = mesh.section(
            plane_origin=pl["origin"],
            plane_normal=pl["normal"],
        )
        if section is None:
            distances.append(0.0)
            continue
        verts = section.vertices
        # Max vertical distance from cross-section to the occlusal curve point
        dist = np.max(np.abs(verts[:, 2] - pl["origin"][2]))
        distances.append(dist)

    distances = np.array(distances)

    # Compute arc positions of each plane origin along the curve
    origins = np.array([pl["origin"] for pl in planes])
    arc_diffs = np.linalg.norm(np.diff(origins, axis=0), axis=1)
    arc_pos = np.concatenate([[0], np.cumsum(arc_diffs)])

    # Per-plane local median within neighbor_distance (half each side)
    half = neighbor_distance / 2.0
    is_connector = []
    for i in range(len(planes)):
        left = arc_pos[i] - half
        right = arc_pos[i] + half
        mask = (arc_pos >= left) & (arc_pos <= right) & (distances > 0)
        if mask.any():
            if height_method == "max":
                local_median = np.max(distances[mask])
            else:
                local_median = np.median(distances[mask])
        else:
            local_median = 1.0
        threshold = threshold_ratio * local_median
        is_connector.append(0 < distances[i] < threshold)
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

    # Trim groups that exceed max_connector_width
    if max_connector_width > 0:
        trimmed_groups = []
        for g in groups:
            first_origin = planes[g[0]]["origin"]
            last_origin = planes[g[-1]]["origin"]
            width = np.linalg.norm(last_origin - first_origin)
            if width <= max_connector_width:
                trimmed_groups.append(g)
            else:
                # Centre around the plane with the least Z-extent
                origins = np.array([planes[i]["origin"] for i in g])
                group_distances = distances[g]
                center_idx = int(np.argmin(group_distances))
                center_origin = origins[center_idx]
                half_w = max_connector_width / 2.0
                keep = []
                for j, idx in enumerate(g):
                    dist = np.linalg.norm(origins[j] - center_origin)
                    if dist <= half_w:
                        keep.append(idx)
                if keep:
                    trimmed_groups.append(keep)
        groups = trimmed_groups

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

# ── 6. Interactive GUI ────────────────────────────────────────────────────

class OcclusalApp:
    """Open3D GUI application with adjustable parameters and a Recompute button."""

    MENU_OPEN = 1

    def __init__(self, stl_path: str | None = None, spacing: float = 0.2,
                 plane_size: float = 7.5, threshold_ratio: float = 0.85,
                 skip_extremes: bool = True, wireframe: bool = False,
                 neighbor_distance: float = 10.0,
                 max_connector_width: float = 1.0):
        self._mesh: trimesh.Trimesh | None = None
        self._curve: np.ndarray | None = None

        gui = o3d.visualization.gui
        self._gui = gui

        self._app = gui.Application.instance
        self._app.initialize()

        # ── Window ──
        self._window = self._app.create_window("Occlusal Plane Slicing", 1400, 900)
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

        # Spacing
        self._panel.add_child(gui.Label("Spacing"))
        self._spacing_edit = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self._spacing_edit.double_value = spacing
        self._spacing_edit.set_limits(0.001, 100.0)
        self._panel.add_child(self._spacing_edit)

        # Plane half-size
        self._panel.add_child(gui.Label("Plane half-size"))
        self._plane_size_edit = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self._plane_size_edit.double_value = plane_size
        self._plane_size_edit.set_limits(0.1, 100.0)
        self._panel.add_child(self._plane_size_edit)

        # Threshold ratio
        self._panel.add_child(gui.Label("Threshold ratio"))
        self._threshold_edit = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self._threshold_edit.double_value = threshold_ratio
        self._threshold_edit.set_limits(0.01, 1.0)
        self._panel.add_child(self._threshold_edit)

        # Neighbor distance
        self._panel.add_child(gui.Label("Neighbor distance"))
        self._neighbor_dist_edit = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self._neighbor_dist_edit.double_value = neighbor_distance
        self._neighbor_dist_edit.set_limits(0.1, 1000.0)
        self._panel.add_child(self._neighbor_dist_edit)

        # Max connector width
        self._panel.add_child(gui.Label("Max connector width"))
        self._max_conn_width_edit = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self._max_conn_width_edit.double_value = max_connector_width
        self._max_conn_width_edit.set_limits(0.1, 1000.0)
        self._panel.add_child(self._max_conn_width_edit)

        # Height method
        self._panel.add_child(gui.Label("Height method"))
        self._height_max_rb = gui.RadioButton(gui.RadioButton.VERT)
        self._height_max_rb.set_items(["Max", "Median"])
        self._height_max_rb.selected_index = 0
        self._panel.add_child(self._height_max_rb)

        # Skip extremes
        self._skip_extremes_cb = gui.Checkbox("Skip extremes")
        self._skip_extremes_cb.checked = skip_extremes
        self._panel.add_child(self._skip_extremes_cb)

        # Wireframe
        self._wireframe_cb = gui.Checkbox("Wireframe")
        self._wireframe_cb.checked = wireframe
        self._panel.add_child(self._wireframe_cb)

        # Show options
        self._show_mesh_cb = gui.Checkbox("Show mesh")
        self._show_mesh_cb.checked = True
        self._panel.add_child(self._show_mesh_cb)

        self._show_curve_cb = gui.Checkbox("Show curve")
        self._show_curve_cb.checked = True
        self._panel.add_child(self._show_curve_cb)

        self._show_planes_cb = gui.Checkbox("Show planes")
        self._show_planes_cb.checked = False
        self._panel.add_child(self._show_planes_cb)

        self._show_connector_planes_only_cb = gui.Checkbox("Connector planes only")
        self._show_connector_planes_only_cb.checked = False
        self._panel.add_child(self._show_connector_planes_only_cb)

        self._show_connectors_cb = gui.Checkbox("Show connectors")
        self._show_connectors_cb.checked = True
        self._panel.add_child(self._show_connectors_cb)

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
            pts = project_mesh_to_plane(mesh)
            curve = fit_occlusal_curve(pts, mesh_vertices=mesh.vertices)
            self._mesh = mesh
            self._curve = curve
            self._file_label.text = path.split("/")[-1]
            self._status_label.text = f"{len(pts)} intersections"
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
        curve = self._curve

        spacing = self._spacing_edit.double_value
        plane_size = self._plane_size_edit.double_value
        threshold_ratio = self._threshold_edit.double_value
        neighbor_distance = self._neighbor_dist_edit.double_value
        skip_extremes = self._skip_extremes_cb.checked
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

        # Curve
        if self._show_curve_cb.checked and curve is not None:
            tube = make_curve_geometry(curve)
            mat_c = o3d.visualization.rendering.MaterialRecord()
            mat_c.shader = "defaultLit"
            mat_c.base_color = [0.0, 0.3, 1.0, 1.0]
            scene.add_geometry("curve", tube, mat_c)

        # Planes and connectors
        planes = create_slicing_planes(curve, spacing=spacing,
                                       plane_half_size=plane_size)

        height_method = "median" if self._height_max_rb.selected_index == 1 else "max"
        max_connector_width = self._max_conn_width_edit.double_value
        connectors = find_connectors(mesh, planes,
                                     threshold_ratio=threshold_ratio,
                                     skip_extremes=skip_extremes,
                                     neighbor_distance=neighbor_distance,
                                     height_method=height_method,
                                     max_connector_width=max_connector_width)
        connector_indices = set()
        for group in connectors:
            for pl in group:
                for i, p in enumerate(planes):
                    if p is pl:
                        connector_indices.add(i)
                        break

        if self._show_planes_cb.checked or self._show_connector_planes_only_cb.checked:
            quads = make_planes_geometry(planes)
            for i, q in enumerate(quads):
                is_conn = i in connector_indices
                if not self._show_planes_cb.checked and not is_conn:
                    continue
                if self._show_connector_planes_only_cb.checked and not self._show_planes_cb.checked and not is_conn:
                    continue
                mat_p = o3d.visualization.rendering.MaterialRecord()
                mat_p.shader = "defaultLit"
                if is_conn:
                    mat_p.base_color = [0.0, 1.0, 0.3, 0.6]
                else:
                    mat_p.base_color = [1.0, 0.2, 0.2, 0.6]
                scene.add_geometry(f"plane_{i}", q, mat_p)

        if self._show_connectors_cb.checked:
            if connectors:
                pcd = make_connectors_geometry(mesh, connectors)
                mat_conn = o3d.visualization.rendering.MaterialRecord()
                mat_conn.shader = "defaultUnlit"
                mat_conn.point_size = 3.0
                scene.add_geometry("connectors", pcd, mat_conn)
            self._status_label.text = (
                f"{len(planes)} planes, {len(connectors)} connectors")
        else:
            self._status_label.text = f"{len(planes)} planes"

        # Fit camera to scene
        bounds = scene.bounding_box
        self._scene.setup_camera(60.0, bounds, bounds.get_center())

    def run(self):
        self._app.run()


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Occlusal-plane slicing of dental STL")
    parser.add_argument("stl", nargs="?", default=None,
                        help="Path to the input STL file (optional; can load from GUI)")
    parser.add_argument("--spacing", type=float, default=0.2,
                        help="Distance between slicing planes (default: 0.2)")
    parser.add_argument("--plane-size", type=float, default=7.5,
                        help="Half-size of rendered slice planes (default: 7.5)")
    parser.add_argument("--wireframe", action="store_true",
                        help="Render mesh as wireframe instead of solid")
    parser.add_argument("--threshold-ratio", type=float, default=0.85,
                        help="Z-extent ratio below which a plane is a connector (default: 0.85)")
    parser.add_argument("--no-skip-extremes", action="store_true",
                        help="Keep connector groups at the first/last plane")
    parser.add_argument("--neighbor-distance", type=float, default=10.0,
                        help="Arc distance for local median neighborhood (default: 10)")
    parser.add_argument("--max-connector-width", type=float, default=1.0,
                        help="Max width of a connector region (default: 1.0)")
    args = parser.parse_args()

    app = OcclusalApp(
        stl_path=args.stl,
        spacing=args.spacing,
        plane_size=args.plane_size,
        threshold_ratio=args.threshold_ratio,
        skip_extremes=not args.no_skip_extremes,
        wireframe=args.wireframe,
        neighbor_distance=args.neighbor_distance,
        max_connector_width=args.max_connector_width,
    )
    app.run()


if __name__ == "__main__":
    main()

