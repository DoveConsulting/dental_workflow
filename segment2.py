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
    onto a 2D grid, closing small gaps, and tracing the outer contour
    using 8-connected grid adjacency (no shortcuts across the interior).

    *grid_res* is the cell size (mm).  Smaller = more detail, larger = smoother.

    Returns an Mx3 array of ordered boundary points with Z=0.
    """
    pts_2d = mesh.vertices[:, :2]

    pad = grid_res * 3
    xmin, ymin = pts_2d.min(axis=0) - pad
    xmax, ymax = pts_2d.max(axis=0) + pad

    nx = int(np.ceil((xmax - xmin) / grid_res)) + 1
    ny = int(np.ceil((ymax - ymin) / grid_res)) + 1

    # Rasterise all mesh edges onto the grid (not just vertices)
    # This ensures continuous fill even where triangles are large.
    edges = mesh.edges_unique
    p0 = pts_2d[edges[:, 0]]
    p1 = pts_2d[edges[:, 1]]
    lengths = np.linalg.norm(p1 - p0, axis=1)
    max_steps = max(int(np.ceil(lengths.max() / grid_res)), 1)

    grid = np.zeros((ny, nx), dtype=bool)
    for step in range(max_steps + 1):
        t = step / max_steps
        pts_interp = p0 + t * (p1 - p0)
        gx = ((pts_interp[:, 0] - xmin) / grid_res).astype(int)
        gy = ((pts_interp[:, 1] - ymin) / grid_res).astype(int)
        valid = (gx >= 0) & (gx < nx) & (gy >= 0) & (gy < ny)
        grid[gy[valid], gx[valid]] = True

    # Fill interior holes enclosed by the rasterised edges
    from scipy.ndimage import binary_fill_holes
    grid = binary_fill_holes(grid)

    # Morphological close to smooth jagged boundary
    struct = np.ones((3, 3), dtype=bool)
    grid = binary_dilation(grid, structure=struct, iterations=1)
    grid = binary_erosion(grid, structure=struct, iterations=1)

    # Keep only the largest connected component
    from scipy.ndimage import label as ndlabel
    labeled, num_features = ndlabel(grid)
    if num_features > 1:
        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0  # ignore background
        largest = sizes.argmax()
        grid = labeled == largest

    # Trace outer contour on the filled grid using Moore boundary tracing
    contour_rc = _trace_outer_contour(grid)
    if len(contour_rc) == 0:
        return np.empty((0, 3))

    # Convert grid (row, col) back to world XY
    outline_xy = np.column_stack([
        xmin + contour_rc[:, 1] * grid_res,
        ymin + contour_rc[:, 0] * grid_res,
    ])

    return np.column_stack([outline_xy, np.zeros(len(outline_xy))])


def _trace_outer_contour(grid: np.ndarray) -> np.ndarray:
    """Trace the outer boundary of a filled binary grid using Moore
    neighbourhood tracing.  Walks along 8-connected boundary pixels
    in order — cannot jump across the interior.

    Returns an Mx2 array of (row, col) indices forming the closed contour.
    """
    ny, nx = grid.shape

    # Find starting boundary pixel: topmost row with a filled cell,
    # leftmost filled cell in that row.  This is guaranteed to be on
    # the outer boundary.
    start = None
    for y in range(ny):
        for x in range(nx):
            if grid[y, x]:
                start = (y, x)
                break
        if start is not None:
            break

    if start is None:
        return np.empty((0, 2), dtype=int)

    # 8-connectivity clockwise: E, SE, S, SW, W, NW, N, NE
    dy = np.array([0, 1, 1, 1, 0, -1, -1, -1])
    dx = np.array([1, 1, 0, -1, -1, -1, 0, 1])

    # Boundary pixel = filled cell with at least one empty 4-neighbour
    def is_boundary(y, x):
        if not grid[y, x]:
            return False
        for ddy, ddx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            ny2, nx2 = y + ddy, x + ddx
            if ny2 < 0 or ny2 >= ny or nx2 < 0 or nx2 >= nx or not grid[ny2, nx2]:
                return True
        return False

    contour = [start]
    visited = np.zeros((ny, nx), dtype=bool)
    visited[start[0], start[1]] = True
    current = start
    # We came from the west (since start is leftmost in its row)
    last_dir = 0  # arrived heading east

    max_steps = ny * nx  # safety limit

    while len(contour) < max_steps:
        # Search clockwise starting from (last_dir + 5) % 8
        # This is ~135° back from the direction of travel, ensuring we
        # hug the boundary tightly.
        search_start = (last_dir + 5) % 8
        found_next = False

        for i in range(8):
            d = (search_start + i) % 8
            ny2 = current[0] + dy[d]
            nx2 = current[1] + dx[d]

            if ny2 < 0 or ny2 >= ny or nx2 < 0 or nx2 >= nx:
                continue
            if not grid[ny2, nx2]:
                continue

            # Check if we completed the loop (back to start)
            if (ny2, nx2) == start and len(contour) > 2:
                return np.array(contour)

            if visited[ny2, nx2]:
                continue

            if is_boundary(ny2, nx2):
                contour.append((ny2, nx2))
                visited[ny2, nx2] = True
                current = (ny2, nx2)
                last_dir = d
                found_next = True
                break

        if not found_next:
            break

    return np.array(contour)


# ── 5. Detect narrowing regions ───────────────────────────────────────────

def find_narrowings(
    outline: np.ndarray,
    min_prominence: float = 0.7,
    min_arc_gap: float = 0.25,
    sigma: float = 3.0,
) -> list[dict]:
    """Find narrowing regions along the closed outline curve.

    For each outline point, the "local width" is the distance to the closest
    point on the opposite side of the curve (excluding neighbours within
    *min_arc_gap* fraction of the total arc length on each side).

    Local minima are detected via numerical differentiation of the Gaussian-
    smoothed width profile:
      1. Smooth width(s) with a Gaussian kernel of standard deviation *sigma*
         (in index space).
      2. Compute the first derivative dw/ds using central finite differences.
      3. Locate zero-crossings where dw/ds goes from negative to positive
         (transition from decreasing to increasing width → local minimum).
      4. Confirm with second derivative d²w/ds² > 0 (concave-up).
      5. Filter by prominence: only keep minima whose depth (height of the
         nearest surrounding maximum minus the minimum value) exceeds
         *min_prominence* (in mm).

    Returns a list of dicts with keys:
        'index'      – index into *outline*
        'point'      – the outline point at the narrowing
        'opposite'   – the closest opposing outline point
        'width'      – the smoothed width value at the minimum
        'prominence' – depth of the dip relative to surrounding maxima
    """
    from scipy.ndimage import gaussian_filter1d
    from scipy.spatial import cKDTree

    n = len(outline)
    if n < 10:
        return []

    # ── Arc length parameterisation (closed loop) ──
    diffs = np.diff(outline, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    close_len = np.linalg.norm(outline[0] - outline[-1])
    all_lens = np.append(seg_lens, close_len)
    arc = np.concatenate([[0], np.cumsum(all_lens)])
    total_arc = arc[-1]
    gap = min_arc_gap * total_arc

    # ── Vectorised width computation ──
    tree = cKDTree(outline)
    k = min(n, max(64, int(n * 0.4)))
    dists_all, idxs_all = tree.query(outline, k=k)

    widths = np.full(n, np.inf)
    closest_idx = np.zeros(n, dtype=int)

    for i in range(n):
        for rank in range(1, k):
            j = idxs_all[i, rank]
            d_arc_fwd = abs(arc[j] - arc[i])
            d_arc = min(d_arc_fwd, total_arc - d_arc_fwd)
            if d_arc < gap:
                continue
            widths[i] = dists_all[i, rank]
            closest_idx[i] = j
            break

    widths[widths == np.inf] = 0

    # ── Gaussian smoothing (circular / wrap mode) ──
    w = gaussian_filter1d(widths, sigma=sigma, mode='wrap')

    # ── First derivative  dw/ds  (central differences, periodic) ──
    dw = np.zeros(n)
    for i in range(n):
        ip = (i + 1) % n
        im = (i - 1) % n
        ds = arc[ip] - arc[im]
        if i == 0:
            ds = all_lens[0] + all_lens[-1]
        if ds < 1e-12:
            continue
        dw[i] = (w[ip] - w[im]) / ds

    # ── Second derivative  d²w/ds²  (central differences, periodic) ──
    d2w = np.zeros(n)
    for i in range(n):
        ip = (i + 1) % n
        im = (i - 1) % n
        ds_p = all_lens[i] if i < n - 1 else all_lens[-1]
        ds_m = all_lens[im]
        ds_avg = (ds_p + ds_m) / 2.0
        if ds_avg < 1e-12:
            continue
        d2w[i] = (w[ip] - 2 * w[i] + w[im]) / (ds_avg * ds_avg)

    # ── Find all local minima and maxima via zero-crossings ──
    minima_indices = []
    for i in range(n):
        im = (i - 1) % n
        if dw[im] < 0 and dw[i] >= 0 and d2w[i] > 0:
            if w[i] > 0:
                minima_indices.append(i)

    maxima_indices = []
    for i in range(n):
        im = (i - 1) % n
        if dw[im] > 0 and dw[i] <= 0:
            maxima_indices.append(i)

    if not minima_indices:
        return []

    # ── Compute prominence for each minimum ──
    # Prominence = min(left_max, right_max) - min_value
    # where left_max/right_max are the highest maxima on each side
    # before reaching a lower minimum.
    def _compute_prominence(min_idx: int) -> float:
        min_val = w[min_idx]

        # Search left (decreasing index, wrapping) for the nearest maximum
        left_max = min_val
        idx = (min_idx - 1) % n
        steps = 0
        while steps < n:
            if idx in maxima_set:
                left_max = w[idx]
                break
            if idx in minima_set and w[idx] < min_val:
                break
            idx = (idx - 1) % n
            steps += 1

        # Search right (increasing index, wrapping) for the nearest maximum
        right_max = min_val
        idx = (min_idx + 1) % n
        steps = 0
        while steps < n:
            if idx in maxima_set:
                right_max = w[idx]
                break
            if idx in minima_set and w[idx] < min_val:
                break
            idx = (idx + 1) % n
            steps += 1

        ref = min(left_max, right_max)
        return ref - min_val

    minima_set = set(minima_indices)
    maxima_set = set(maxima_indices)

    # ── Filter by prominence ──
    narrowings = []
    for i in minima_indices:
        prom = _compute_prominence(i)
        if prom >= min_prominence:
            narrowings.append({
                'index': i,
                'point': outline[i],
                'opposite': outline[closest_idx[i]],
                'width': float(w[i]),
                'prominence': float(prom),
            })

    # ── Deduplicate symmetric pairs ──
    if len(narrowings) > 1:
        deduped = []
        used_pairs = set()
        for nr in narrowings:
            pair = tuple(sorted((nr['index'], int(closest_idx[nr['index']]))))
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


def compute_narrowing_sections(
    mesh: trimesh.Trimesh,
    narrowings: list[dict],
) -> list[np.ndarray]:
    """Slice the mesh with a vertical plane at each narrowing to produce
    cross-section contours that loop around the mesh surface.

    The cutting plane contains the narrowing line (point ↔ opposite)
    and the Z axis.  Its normal is the cross product of the narrowing
    direction with Z.
    """
    sections = []

    for nr in narrowings:
        direction = nr['opposite'] - nr['point']
        direction_2d = direction[:2]
        d_len = np.linalg.norm(direction_2d)
        if d_len < 1e-12:
            continue
        # normal = direction × Z  →  lies in XY, perpendicular to the narrowing line
        plane_normal = np.array([direction_2d[1], -direction_2d[0], 0.0]) / d_len

        # Plane origin at the midpoint between the narrowing pair
        origin = (nr['point'] + nr['opposite']) / 2.0

        section = mesh.section(
            plane_origin=origin,
            plane_normal=plane_normal,
        )
        if section is None:
            continue

        # Pick the path that passes closest to both narrowing endpoints
        try:
            best_path = None
            best_score = np.inf
            for path_pts in section.discrete:
                if len(path_pts) < 3:
                    continue
                pts = np.array(path_pts)
                # Minimum distance from any path vertex to each endpoint
                d_to_point = np.linalg.norm(pts - nr['point'], axis=1).min()
                d_to_opposite = np.linalg.norm(pts - nr['opposite'], axis=1).min()
                score = d_to_point + d_to_opposite
                if score < best_score:
                    best_score = score
                    best_path = pts
            if best_path is not None:
                sections.append(best_path)
        except Exception:
            pass

    return sections


def make_narrowing_rings_geometry(
    sections: list[np.ndarray],
) -> list[o3d.geometry.LineSet]:
    """Create LineSet geometry for cross-section rings at narrowings."""
    geometries = []
    for pts in sections:
        m = len(pts)
        if m < 3:
            continue
        lines = [[i, (i + 1) % m] for i in range(m)]

        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(pts)
        ls.lines = o3d.utility.Vector2iVector(np.array(lines))
        ls.paint_uniform_color([1.0, 1.0, 0.0])
        geometries.append(ls)

    return geometries


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
                 min_prominence: float = 0.7, min_arc_gap: float = 0.25):
        self._mesh: trimesh.Trimesh | None = None
        self._outline: np.ndarray | None = None
        self._camera_initialized = False

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
        self._show_mesh_cb.checked = False
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

        self._show_rings_cb = gui.Checkbox("Show narrowing rings")
        self._show_rings_cb.checked = True
        self._panel.add_child(self._show_rings_cb)

        sep3 = gui.Label("─── Narrowing ───")
        self._panel.add_child(sep3)

        # Min prominence
        self._panel.add_child(gui.Label("Min prominence"))
        self._min_prom_edit = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self._min_prom_edit.double_value = min_prominence
        self._min_prom_edit.set_limits(0.0, 20.0)
        self._panel.add_child(self._min_prom_edit)

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

        # Origin triad (XYZ axes)
        triad = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0)
        mat_triad = o3d.visualization.rendering.MaterialRecord()
        mat_triad.shader = "defaultUnlit"
        scene.add_geometry("triad", triad, mat_triad)

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

        # Narrowings
        if (self._show_narrowings_cb.checked or self._show_rings_cb.checked) and len(outline) > 5:
            min_prominence = self._min_prom_edit.double_value
            min_arc_gap = self._min_arc_gap_edit.double_value
            narrowings = find_narrowings(outline, min_prominence=min_prominence,
                                         min_arc_gap=min_arc_gap)

            # Narrowing lines
            if self._show_narrowings_cb.checked and len(narrowings) > 0:
                ls = make_narrowing_geometry(narrowings)
                mat_n = o3d.visualization.rendering.MaterialRecord()
                mat_n.shader = "unlitLine"
                mat_n.line_width = 3.0
                scene.add_geometry("narrowings", ls, mat_n)

            # Narrowing rings (cross-section contours around mesh)
            if self._show_rings_cb.checked and len(narrowings) > 0:
                sections = compute_narrowing_sections(mesh, narrowings)
                ring_geoms = make_narrowing_rings_geometry(sections)
                for i, rg in enumerate(ring_geoms):
                    mat_r = o3d.visualization.rendering.MaterialRecord()
                    mat_r.shader = "unlitLine"
                    mat_r.line_width = 4.0
                    scene.add_geometry(f"ring_{i}", rg, mat_r)

            self._status_label.text = (
                f"{len(outline)} boundary, "
                f"{len(narrowings)} narrowings")
        else:
            self._status_label.text = f"{len(outline)} boundary pts"

        # Fit camera only on first load
        if not self._camera_initialized:
            bounds = scene.bounding_box
            self._scene.setup_camera(60.0, bounds, bounds.get_center())
            self._camera_initialized = True

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
    parser.add_argument("--min-prominence", type=float, default=0.7,
                        help="Min prominence (mm) for narrowing detection (default: 0.7)")
    parser.add_argument("--min-arc-gap", type=float, default=0.1,
                        help="Min arc-length fraction to skip neighbours (default: 0.1)")
    args = parser.parse_args()

    app = OutlineApp(
        stl_path=args.stl,
        grid_res=args.grid_res,
        num_samples=args.num_samples,
        wireframe=args.wireframe,
        min_prominence=args.min_prominence,
        min_arc_gap=args.min_arc_gap,
    )
    app.run()


if __name__ == "__main__":
    main()
