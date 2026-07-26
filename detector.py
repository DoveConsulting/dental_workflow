import os
import re
import math
import numpy as np
import cv2
import open3d as o3d
from ultralytics import YOLO

# ── Configuration ──────────────────────────────────────────────
STL_DIR         = "stl_test"              # folder of .stl files to process (mirrors training_data_collection.py)
OUTPUT_DIR      = "detections"       # debug images + final overlays are written here
MODEL_PATH      = "ai_model/best.pt"  # YOLO detector trained on ToothTop / ToothBottom / Connector
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
    "ToothBottom": (1.0, 0.0, 0.0),   # red
}

# ── Debug annotation styling (step 4) ───────────────────────────
LABEL_FONT_SCALE = 0.75   # multiplier for annotation label text size
LABEL_OPACITY    = 0.75   # 0.0 (invisible) .. 1.0 (fully opaque) for boxes + label text
SHOW_CONFIDENCE  = False  # append the confidence score to the annotation label

# BGR (OpenCV order) colors used to draw detections on the debug images.
# Covers all three labels, unlike LABEL_COLORS above which only covers the
# two that get segmented onto the mesh.
ANNOTATION_COLORS = {
    "ToothTop":    (0, 255, 0),    # green
    "ToothBottom": (0, 0, 255),    # red
    "Connector":   (0, 255, 255),  # yellow
}
DEFAULT_ANNOTATION_COLOR = (255, 255, 255)  # white, used for any unlisted label

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


def mesh_to_pointcloud(mesh, number_of_points=10_000):
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
def capture_top_bottom_views(mesh, out_dir):
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

        img_path = os.path.join(out_dir, f"{name}.png")
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
# Explicit abbreviations for known labels; anything else falls back to the
# initials of its capitalized words (e.g. a future 'GumLine' -> 'GL').
LABEL_ABBREVIATIONS = {
    "ToothTop":    "TT",
    "ToothBottom": "TB",
    "Connector":   "C",
}


def sanitize_label(label):
    """Abbreviated, uppercase-letters-only annotation text for a label
    (e.g. 'ToothTop' -> 'TT', 'ToothBottom' -> 'TB', 'Connector' -> 'C')."""
    if label in LABEL_ABBREVIATIONS:
        return LABEL_ABBREVIATIONS[label]

    words = re.findall(r"[A-Z][a-z]*", label)
    if words:
        return "".join(word[0] for word in words).upper()

    return re.sub(r"[^A-Za-z]", "", label).upper()


def draw_detections(image_path, detections, out_path,
                     font_scale=LABEL_FONT_SCALE,
                     opacity=LABEL_OPACITY,
                     show_confidence=SHOW_CONFIDENCE):
    """Draw bounding boxes (rotated, if the model is OBB) and abbreviated
    labels onto the captured image and save it for debugging.

    font_scale       - scales the label text size
    opacity          - 0.0 (invisible) .. 1.0 (fully opaque), applies to
                        both the box outline/fill and the label text
    show_confidence  - append the detection confidence to the label text
    """
    base = cv2.imread(image_path)
    overlay = base.copy()
    thickness = max(1, round(font_scale * 5))

    for det in detections:
        color = ANNOTATION_COLORS.get(det["label"], DEFAULT_ANNOTATION_COLOR)

        if "obb_corners" in det:
            pts = np.array(det["obb_corners"], dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(overlay, [pts], isClosed=True, color=color, thickness=thickness)
            tx, ty = det["obb_corners"][0]
            tx, ty = int(tx), int(ty)
        else:
            x1, y1, x2, y2 = map(int, det["bbox"])
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness=thickness)
            tx, ty = x1, y1

        label_text = sanitize_label(det["label"])
        if show_confidence:
            label_text = f"{label_text} {det['confidence']:.2f}"

        (text_w, text_h), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        ty = max(ty, text_h + baseline + 4)  # keep the label on-screen near the top edge
        cv2.rectangle(overlay, (tx, ty - text_h - baseline - 4), (tx + text_w + 4, ty), color, thickness=-1)
        cv2.putText(
            overlay, label_text, (tx + 2, ty - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness, cv2.LINE_AA
        )

    # Blending the whole frame is equivalent to blending just the drawn
    # regions, since overlay == base everywhere nothing was drawn.
    blended = cv2.addWeighted(overlay, opacity, base, 1 - opacity, 0)
    cv2.imwrite(out_path, blended)


def run_inference(image_path, out_dir, view_name):
    model = get_model()
    results = model.predict(source=image_path, conf=CONF_THRESHOLD, verbose=False)[0]

    # OBB (oriented bounding box) models populate results.obb and leave
    # results.boxes as None — regular detection models are the reverse.
    # OBB items expose the same .cls/.conf as Boxes, plus .xyxy (the
    # axis-aligned box enclosing the rotated box) and .xyxyxyxy (the
    # actual 4 rotated corner points, kept for a tighter crop later if needed).
    is_obb = results.obb is not None
    preds = results.obb if is_obb else results.boxes

    detections = []
    for box in preds:
        label = results.names[int(box.cls[0])]
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detection = {
            "label": label,
            "confidence": float(box.conf[0]),
            "bbox": (x1, y1, x2, y2),
        }
        if is_obb:
            detection["obb_corners"] = box.xyxyxyxy[0].tolist()  # [[x,y], [x,y], [x,y], [x,y]]
        detections.append(detection)

    debug_path = os.path.join(out_dir, f"inference_{view_name}.png")
    draw_detections(image_path, detections, debug_path)
    print(f"  ✓ saved {debug_path} ({len(detections)} detections)")
    return detections


# ── Step 5: select the view that has no 'ToothBottom' label ────
def select_valid_view(captures, detections_by_view, out_dir):
    for view_name in ("top", "bottom"):
        dets = detections_by_view[view_name]
        if not any(d["label"] == "ToothBottom" for d in dets):
            chosen_path = os.path.join(out_dir, f"chosen_{view_name}.png")
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


# ── Step 9: top, bottom & isometric captures of the mesh + overlaid segments ──
def capture_overlay_view(mesh, segments, out_dir):
    bbox = mesh.get_axis_aligned_bounding_box()
    radius = bbox.get_max_extent() * 1.8
 
    views = {
        "top":       {"eye": [0, 0,  radius], "up": [0, 1, 0]},
        "bottom":    {"eye": [0, 0, -radius], "up": [0, 1, 0]},
        "isometric": {"eye": [radius * 0.7, -radius * 0.7, radius * 0.7], "up": [0, 0, 1]},
    }
 
    renderer = o3d.visualization.rendering.OffscreenRenderer(IMAGE_WIDTH, IMAGE_HEIGHT)
    renderer.scene.set_background([0.0, 0.0, 0.0, 1.0])
 
    dim_mat = make_material(mesh, color=(0.35, 0.35, 0.35))
    renderer.scene.add_geometry("mesh", mesh, dim_mat)
 
    for i, segment in enumerate(segments):
        renderer.scene.add_geometry(f"segment_{i}", segment, make_material(segment, point_size=4.0))
 
    out_paths = {}
    for name, params in views.items():
        renderer.setup_camera(FOV_DEG, np.array(CENTER), np.array(params["eye"]), np.array(params["up"]))
        out_path = os.path.join(out_dir, f"overlay_{name}.png")
        o3d.io.write_image(out_path, renderer.render_to_image())
        print(f"  ✓ saved {out_path}")
        out_paths[name] = out_path
 
    del renderer
    return out_paths


# ── Per-file pipeline (steps 1-9) ────────────────────────────────
def process_stl(stl_path):
    stem = os.path.splitext(os.path.basename(stl_path))[0]
    print(f"\nProcessing {stl_path} ...")

    # Everything this run produces for this mesh lands under its own
    # subdirectory, named after the source file (e.g. detections/<stem>/...).
    out_dir = os.path.join(OUTPUT_DIR, stem)
    os.makedirs(out_dir, exist_ok=True)

    mesh = load_stl(stl_path)                                     # 1

    mesh = prepare_mesh(mesh)                                     # 2
    output_path = os.path.join(out_dir, "aligned.stl")
    o3d.io.write_triangle_mesh(output_path, mesh)


    captures = capture_top_bottom_views(mesh, out_dir)             # 3

    detections_by_view = {                                        # 4
        view: run_inference(data["image_path"], out_dir, view)
        for view, data in captures.items()
    }

    chosen_view, chosen_detections = select_valid_view(            # 5
        captures, detections_by_view, out_dir
    )

    pcd = mesh_to_pointcloud(mesh)                                  # 6

    segments = segment_pointcloud(                                 # 7 & 8
        pcd, chosen_detections, captures[chosen_view]
    )

    capture_overlay_view(mesh, segments, out_dir)                   # 9


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