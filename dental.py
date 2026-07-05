import os
import math
import open3d as o3d
import numpy as np

# ── Configuration ──────────────────────────────────────────────
STL_PATH       = "0035.stl"
OUTPUT_DIR     = "renders"
IMAGE_WIDTH    = 1920
IMAGE_HEIGHT   = 1080


os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load mesh ──────────────────────────────────────────────────
mesh = o3d.io.read_triangle_mesh(STL_PATH)
mesh.compute_vertex_normals()
mesh.paint_uniform_color([0.95, 0.93, 0.88])

# Center the mesh at the origin and get its bounding sphere
mesh.translate(-mesh.get_center())
bbox   = mesh.get_axis_aligned_bounding_box()
extent = bbox.get_max_extent()
radius = extent * 1.5            # camera distance from center


# ── Define the five view directions ────────────────────────────
# Each entry: (name, eye_position, up_vector)
VIEWS = {
    "front":  {"eye": [0,  0,  radius], "up": [0, 1, 0]},
    "back":   {"eye": [0,  0, -radius], "up": [0, 1, 0]},
    "left":   {"eye": [-radius, 0, 0],  "up": [0, 1, 0]},
    "right":  {"eye": [radius,  0, 0],  "up": [0, 1, 0]},
    "top":    {"eye": [0, radius,  0],  "up": [0, 0, -1]},
}

CENTER = [0.0, 0.0, 0.0]         # look-at target


# ── Capture helper using OffscreenRenderer ─────────────────────
def capture_view(mesh, eye, up, center, width, height, save_path):
    """Render the mesh from a given viewpoint and save to disk."""
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])  # white bg

    # Material
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLit"
    mat.base_color = [0.95, 0.93, 0.88, 1.0]

    renderer.scene.add_geometry("dental_mesh", mesh, mat)

    # Camera
    renderer.setup_camera(
        60.0,                       # vertical field-of-view (degrees)
        np.array(center),           # look-at point
        np.array(eye),              # camera position
        np.array(up)                # up vector
    )

    # Render and save
    img = renderer.render_to_image()
    o3d.io.write_image(save_path, img)
    print(f"  ✓ saved {save_path}")

    renderer.scene.remove_geometry("dental_mesh")
    del renderer


# ── Render each view ───────────────────────────────────────────
filename = os.path.basename(STL_PATH)
saved_paths = []

for name, params in VIEWS.items():
    path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(filename)[0]}_{name}.png")
    capture_view(
        mesh,
        eye=params["eye"],
        up=params["up"],
        center=CENTER,
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
        save_path=path,
    )
    saved_paths.append(path)

print(f"\nAll {len(saved_paths)} views saved to '{OUTPUT_DIR}/'")