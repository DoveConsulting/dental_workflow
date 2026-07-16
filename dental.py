import os
import math
import open3d as o3d
import numpy as np

# ── Configuration ──────────────────────────────────────────────
STL_DIR        = "stl"
OUTPUT_DIR     = "renders"
IMAGE_WIDTH    = 1920
IMAGE_HEIGHT   = 1080
USE_POINTCLOUD = False   # True: convert mesh → point cloud before rendering
                        # False: render the STL mesh directly


os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Define the five view directions ────────────────────────────
# Each entry: (name, eye_position, up_vector)
VIEWS_NAMES = ["front", "back", "left", "right", "top"]

CENTER = [0.0, 0.0, 0.0]         # look-at target



def mesh_to_pointcloud(mesh, number_of_points=100_000):
    """Convert a mesh to a point cloud by sampling points on its surface."""
    # resulting point cloud should have uniform density across the mesh surface
    pcd = mesh.sample_points_uniformly(number_of_points=number_of_points)
    return pcd


# ── Capture helper using OffscreenRenderer ─────────────────────
def capture_view(geometry, eye, up, center, width, height, save_path):
    """Render a mesh or point cloud from a given viewpoint and save to disk."""
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    renderer.scene.set_background([0.0, 0.0, 0.0, 1.0])  # black bg

    # Material — adapt shader to geometry type
    mat = o3d.visualization.rendering.MaterialRecord()
    if isinstance(geometry, o3d.geometry.PointCloud):
        mat.shader = "defaultUnlit"
        mat.point_size = 2.0
    else:
        mat.shader = "defaultLit"
        mat.base_color = [0.95, 0.93, 0.88, 1.0]

    renderer.scene.add_geometry("dental_geom", geometry, mat)

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

    renderer.scene.remove_geometry("dental_geom")
    del renderer


# ── Render each view ───────────────────────────────────────────
stl_files = sorted(
    f for f in os.listdir(STL_DIR) if f.lower().endswith(".stl")
)

if not stl_files:
    print(f"No STL files found in '{STL_DIR}/'")
else:
    total_saved = []
    for stl_file in stl_files:
        stl_path = os.path.join(STL_DIR, stl_file)
        stem = os.path.splitext(stl_file)[0]
        print(f"\nProcessing {stl_file} ...")

        # Load mesh
        mesh = o3d.io.read_triangle_mesh(stl_path)
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color([0.95, 0.93, 0.88])

        # Center and compute camera radius
        mesh.translate(-mesh.get_center())
        bbox   = mesh.get_axis_aligned_bounding_box()
        extent = bbox.get_max_extent()
        radius = extent * 1.5

        VIEWS = {
            "top":  {"eye": [0,  0,  radius], "up": [0, 1, 0]}, #top
            "bottom":   {"eye": [0,  0, -radius], "up": [0, 1, 0]}, #bottom
            # "left":   {"eye": [-radius, 0, 0],  "up": [0, 1, 0]},
            # "right":  {"eye": [radius,  0, 0],  "up": [0, 1, 0]},
            # "back":    {"eye": [0, radius,  0],  "up": [0, 0, -1]},#back
            # "front": {"eye": [0, -radius, 0], "up": [0, 0, 1]}, # front
        }

        geometry = mesh_to_pointcloud(mesh) if USE_POINTCLOUD else mesh

        for name, params in VIEWS.items():
            path = os.path.join(OUTPUT_DIR, f"{name}_{stem}.png")
            capture_view(
                geometry,
                eye=params["eye"],
                up=params["up"],
                center=CENTER,
                width=IMAGE_WIDTH,
                height=IMAGE_HEIGHT,
                save_path=path,
            )
            total_saved.append(path)

    print(f"\nAll {len(total_saved)} views saved to '{OUTPUT_DIR}/'")