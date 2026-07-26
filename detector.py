import os
import math
import numpy as np
import cv2
import open3d as o3d
from ultralytics import YOLO

# ── Configuration ──────────────────────────────────────────────
STL_DIR         = "stl"              # folder of .stl files to process (mirrors training_data_collection.py)
OUTPUT_DIR      = "detections"       # debug images + final overlays are written here
MODEL_PATH      = "weights/best.pt"  # YOLO detector trained on ToothTop / ToothBottom / Connector
IMAGE_WIDTH     = 1920
IMAGE_HEIGHT    = 1080
FOV_DEG         = 60.0
CONF_THRESHOLD  = 0.25

# STL files are almost always authored/exported in millimeters, so the
# "3 cm up / 3 cm down" box height from the spec is converted into mesh units here.
MESH_UNITS      = "mm"
UNITS_PER_CM    = {"mm": 10.0, "cm": 1.0, "m": 0.01}[MESH_UNITS]
BOX_HALF_HEIGHT = 3.0 * UNITS_PER_CM  # step 7: box spans this far above AND below Z=0

# Only labels present in this dict get segmented + colored in step 8.
# ToothBottom is deliberately excluded — it's only used as a selection
# criterion in step 5, never rendered.
LABEL_COLORS = {
    "ToothTop":  (0.0, 1.0, 0.0),   # green
    "Connector": (1.0, 1.0, 0.0),   # yellow
}

CENTER = [0.0, 0.0, 0.0]  # look-at target, mesh is always re-centered to the origin

os.makedirs(OUTPUT_DIR, exist_ok=True)

_model = None
def get_model():
    """Lazily load (and cache) the YOLO detector."""
    global _model
    if _model is None:
        _model = YOLO(MODEL_PATH)
    return _model


# ── Geometry helpers (shared logic with training_data_collection.py) ──
def align_to_principal_axes(mesh):
    """Align mesh to principal axes of inertia, widest section in XY plane.

    Computes the covariance matrix of the mesh vertices, then rotates so
    that the eigenvector with the *smallest* eigenvalue (least spread)
    maps to Z, leaving the two directions with most spread in XY.
    """
    vertices = np.asarray(mesh.vertices)
    centroid = vertices.mean(axis=0)
    centered = vertices - centroid

    cov = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # eigh returns eigenvalues ascending; rearrange columns to largest,
    # middle, smallest → X, Y, Z (smallest spread ends up on Z).
    rot = eigenvectors[:, [2, 1, 0]]

    if np.linalg.det(rot) < 0:
        rot[:, 2] = -rot[:, 2]

    mesh.rotate(rot.T, center=centroid)
    return mesh


def mesh_to_pointcloud(mesh, number_of_points=100_000):
    """Convert a mesh to a point cloud by sampling points on its surface."""
    return mesh.sample_points_uniformly(number_of_points=number_of_points)


def make_material(geometry, color=None, point_size=2.0):
    mat = o3d.visualization.rendering.MaterialRecord()
    if isinstance(geometry, o3d.geometry.PointCloud):
        mat.shader = "defaultUnlit"
        mat.point_size = point_size
    else:
        mat.shader = "defaultLit"
    mat.base_color = [*(color if color is not None else (0.95, 0.93, 0.88)), 1.0]
    return mat


# ── Step 1: load stl file ──────────────────────────────────────
def load_stl(stl_path):
    mesh = o3d.io.read_triangle_mesh(stl_path)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.95, 0.93, 0.88])
    return mesh


# ── Step 2: align mesh to principal axes, widest section in XY ─
def prepare_mesh(mesh):
    mesh.translate(-mesh.get_center())
    align_to_principal_axes(mesh)
    mesh.translate(-mesh.get_center())  # re-center after rotation
    return mesh


# ── Step 3: capture top & bottom views on black background ─────
def capture_top_bottom_views(mesh, stem):
    bbox = mesh.get_axis_aligned_bounding_box()
    radius = bbox.get_max_extent() * 1.5

    views = {
        "top":    {"eye": [0, 0,  radius], "up": [0, 1, 0]},
        "bottom": {"eye": [0, 0, -radius], "up": [0, 1, 0]},
    }

    renderer = o3d.visualization.rendering.OffscreenRenderer(IMAGE_WIDTH, IMAGE_HEIGHT)
    renderer.scene.set_background([0.0, 0.0, 0.0, 1.0])
    renderer.scene.add_geometry("mesh", mesh, make_material(mesh))

    captures = {}
    for name, params in views.items():
        renderer.setup_camera(FOV_DEG, np.array(CENTER), np.array(params["eye"]), np.array(params["up"]))

        img_path = os.path.join(OUTPUT_DIR, f"{name}_{stem}.png")
        o3d.io.write_image(img_path, renderer.render_to_image())
        print(f"  ✓ saved {img_path}")

        # depth buffer (normalized 0..1) is kept so 2D detections from this
        # exact camera pose can be unprojected back into 3D in step 7.
        depth = np.asarray(renderer.render_to_depth_image(z_in_view_space=False))

        captures[name] = {
            "image_path": img_path,
            "depth": depth,
            "eye": params["eye"],
            "up": params["up"],
        }

    renderer.scene.remove_geometry("mesh")
    del renderer
    return captures


# ── Step 4: run inference on the captured images ────────────────
def run_inference(image_path, stem, view_name):
    model = get_model()
    results = model.predict(source=image_path, conf=CONF_THRESHOLD, verbose=False)[0]

    detections = []
    for box in results.boxes:
        label = results.names[int(box.cls[0])]
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append({
            "label": label,
            "confidence": float(box.conf[0]),
            "bbox": (x1, y1, x2, y2),
        })

    debug_path = os.path.join(OUTPUT_DIR, f"inference_{view_name}_{stem}.png")
    cv2.imwrite(debug_path, results.plot())
    print(f"  ✓ saved {debug_path} ({len(detections)} detections)")
    return detections


# ── Step 5: select the view that has no 'ToothBottom' label ────
def select_valid_view(captures, detections_by_view, stem):
    for view_name in ("top", "bottom"):
        dets = detections_by_view[view_name]
        if not any(d["label"] == "ToothBottom" for d in dets):
            chosen_path = os.path.join(OUTPUT_DIR, f"chosen_{view_name}_{stem}.png")
            o3d.io.write_image(chosen_path, o3d.io.read_image(captures[view_name]["image_path"]))
            print(f"  ✓ '{view_name}' view has no ToothBottom labels — using it ({chosen_path})")
            return view_name, dets

    raise RuntimeError(
        "Both top and bottom views contain a 'ToothBottom' label; "
        "cannot select a clean view for segmentation."
    )


# ── Step 6: convert mesh to point cloud ─────────────────────────
# (mesh_to_pointcloud defined above, reused here)


# ── Step 7: pull the points inside each detection's 3D box ─────
def unproject_bbox_to_xy(bbox, depth, eye, up, width, height, fov=FOV_DEG, grid=6):
    """Sample points across a pixel bbox, unproject each hit using the
    matching depth buffer, and return the resulting XY footprint in
    world/mesh coordinates."""
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    renderer.setup_camera(fov, np.array(CENTER), np.array(eye), np.array(up))
    camera = renderer.scene.camera

    x1, y1, x2, y2 = bbox
    world_points = []
    for py in np.linspace(y1, y2, grid):
        for px in np.linspace(x1, x2, grid):
            ix = int(np.clip(round(px), 0, width - 1))
            iy = int(np.clip(round(py), 0, height - 1))
            d = depth[iy, ix]
            if d >= 1.0:  # no geometry hit at this pixel (background)
                continue
            world_points.append(camera.unproject(px, py, d, width, height))

    del renderer
    if not world_points:
        return None

    world_points = np.array(world_points)
    return (
        world_points[:, 0].min(), world_points[:, 0].max(),
        world_points[:, 1].min(), world_points[:, 1].max(),
    )


def points_in_box(points, x_range, y_range, z_half_height=BOX_HALF_HEIGHT):
    x_min, x_max = x_range
    y_min, y_max = y_range
    return (
        (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
        (points[:, 1] >= y_min) & (points[:, 1] <= y_max) &
        (points[:, 2] >= -z_half_height) & (points[:, 2] <= z_half_height)
    )


def segment_pointcloud(pcd, detections, view_capture):
    """Steps 7 & 8: crop the point cloud per-detection and color each
    segment (green = ToothTop, yellow = Connector)."""
    points = np.asarray(pcd.points)
    depth, eye, up = view_capture["depth"], view_capture["eye"], view_capture["up"]

    segments = []
    for det in detections:
        color = LABEL_COLORS.get(det["label"])
        if color is None:
            continue  # e.g. ToothBottom — not part of the overlay

        xy = unproject_bbox_to_xy(det["bbox"], depth, eye, up, IMAGE_WIDTH, IMAGE_HEIGHT)
        if xy is None:
            continue
        x_min, x_max, y_min, y_max = xy

        mask = points_in_box(points, (x_min, x_max), (y_min, y_max))
        if not mask.any():
            continue

        segment = o3d.geometry.PointCloud()
        segment.points = o3d.utility.Vector3dVector(points[mask])
        segment.paint_uniform_color(color)
        segments.append(segment)

    return segments


# ── Step 9: isometric capture of the mesh + overlaid segments ──
def capture_overlay_view(mesh, segments, stem):
    bbox = mesh.get_axis_aligned_bounding_box()
    radius = bbox.get_max_extent() * 1.8

    # top-down isometric-style vantage point
    eye = [radius * 0.7, -radius * 0.7, radius * 0.7]
    up = [0, 0, 1]

    renderer = o3d.visualization.rendering.OffscreenRenderer(IMAGE_WIDTH, IMAGE_HEIGHT)
    renderer.scene.set_background([0.0, 0.0, 0.0, 1.0])

    dim_mat = make_material(mesh, color=(0.35, 0.35, 0.35))
    renderer.scene.add_geometry("mesh", mesh, dim_mat)

    for i, segment in enumerate(segments):
        renderer.scene.add_geometry(f"segment_{i}", segment, make_material(segment, point_size=4.0))

    renderer.setup_camera(FOV_DEG, np.array(CENTER), np.array(eye), np.array(up))
    out_path = os.path.join(OUTPUT_DIR, f"overlay_{stem}.png")
    o3d.io.write_image(out_path, renderer.render_to_image())
    print(f"  ✓ saved {out_path}")

    del renderer
    return out_path


# ── Per-file pipeline (steps 1-9) ────────────────────────────────
def process_stl(stl_path):
    stem = os.path.splitext(os.path.basename(stl_path))[0]
    print(f"\nProcessing {stl_path} ...")

    mesh = load_stl(stl_path)                                     # 1
    mesh = prepare_mesh(mesh)                                     # 2
    captures = capture_top_bottom_views(mesh, stem)                # 3

    detections_by_view = {                                        # 4
        view: run_inference(data["image_path"], stem, view)
        for view, data in captures.items()
    }

    chosen_view, chosen_detections = select_valid_view(            # 5
        captures, detections_by_view, stem
    )

    pcd = mesh_to_pointcloud(mesh)                                  # 6

    segments = segment_pointcloud(                                 # 7 & 8
        pcd, chosen_detections, captures[chosen_view]
    )

    capture_overlay_view(mesh, segments, stem)                      # 9


# ── Entry point ───────────────────────────────────────────────
def main():
    stl_files = sorted(f for f in os.listdir(STL_DIR) if f.lower().endswith(".stl"))
    if not stl_files:
        print(f"No STL files found in '{STL_DIR}/'")
        return

    for stl_file in stl_files:
        process_stl(os.path.join(STL_DIR, stl_file))

    print("\nDone.")


if __name__ == "__main__":
    main()
