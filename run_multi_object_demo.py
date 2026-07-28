"""Estimate poses for every CAD-backed object visible in one synthetic frame.

Object selection follows the same scene_meta validation used by the single
object adapter. Class IDs are mapped strictly to objects_metadata.csv Old_name;
there is intentionally no Object_name or directory-name fallback.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import cv2
import imageio
import nvdiffrast.torch as dr
import numpy as np
import torch
import trimesh
from scipy.optimize import linear_sum_assignment

from custom_datareader import CustomSceneReader, load_mesh_readonly
from estimater import FoundationPose, PoseRefinePredictor, ScorePredictor
from Utils import (
    draw_posed_3d_box,
    draw_xyz_axis,
    project_3d_to_2d,
    set_logging_format,
    set_seed,
)


def build_parser():
  code_dir = Path(__file__).resolve().parent
  parser = argparse.ArgumentParser(
      description="Run sequential multi-object FoundationPose on synthetic RGB-D."
  )
  parser.add_argument("--scene_dir", default=str(code_dir / "demo_data" / "cough"))
  parser.add_argument(
      "--cad_root",
      default="/media/uon/data/3d_model/peel3_scan_data_2025",
  )
  parser.add_argument(
      "--objects_metadata",
      default=None,
      help="Defaults to <scene_dir>/objects_metadata.csv.",
  )
  parser.add_argument("--camera_name", default="top_view_camera")
  parser.add_argument(
      "--frame_id",
      default=None,
      help="Frame stem to process; defaults to the first RGB frame.",
  )
  parser.add_argument("--mesh_scale", type=float, default=0.001)
  parser.add_argument("--max_image_size", type=int, default=640)
  parser.add_argument("--max_texture_size", type=int, default=4096)
  parser.add_argument(
      "--max_mask_match_distance",
      type=float,
      default=250.0,
      help="Maximum source-image pixel distance for GT center to mask-color matching.",
  )
  parser.add_argument("--est_refine_iter", type=int, default=5)
  parser.add_argument("--debug", type=int, default=2)
  parser.add_argument("--debug_dir", default=str(code_dir / "debug_cough"))
  parser.add_argument(
      "--no_gui",
      action="store_true",
      help="Accepted for single-runner CLI parity; this runner saves files only.",
  )
  return parser


def load_json(path):
  with Path(path).open("r", encoding="utf-8") as stream:
    return json.load(stream)


def get_frame_id(scene_dir, camera_name, requested):
  rgb_dir = Path(scene_dir) / "rgb" / camera_name
  files = sorted(rgb_dir.glob("*.png"))
  if not files:
    raise FileNotFoundError(f"No RGB PNG files found in {rgb_dir}")
  if requested is None:
    return files[0].stem
  path = rgb_dir / f"{requested}.png"
  if not path.is_file():
    raise FileNotFoundError(f"Requested RGB frame not found: {path}")
  return requested


def load_old_name_catalog(csv_path):
  catalog = {}
  with Path(csv_path).open("r", encoding="utf-8-sig", newline="") as stream:
    for row in csv.DictReader(stream):
      class_id = row.get("Class_name", "").strip()
      old_name = row.get("Old_name", "").strip()
      if class_id and old_name:
        catalog[class_id] = old_name
  return catalog


def get_camera(frame_metadata, camera_name):
  matches = [
      camera for camera in frame_metadata.get("cameras", [])
      if camera.get("name") == camera_name
  ]
  if len(matches) != 1:
    available = [
        camera.get("name") for camera in frame_metadata.get("cameras", [])
    ]
    raise ValueError(
        f"Expected one camera named {camera_name!r}, found {len(matches)}; "
        f"available cameras: {available}"
    )
  return matches[0]


def project_world_point(camera, world_point):
  """Project an Isaac/Usd camera-space point onto source-image pixels."""
  cam_in_world = np.asarray(camera["cam_poses"], dtype=np.float64).reshape(4, 4)
  world_point = np.asarray(world_point, dtype=np.float64).reshape(3)
  point_glcam = cam_in_world[:3, :3].T @ (
      world_point - cam_in_world[:3, 3]
  )
  depth = -point_glcam[2]
  if depth <= 0:
    return None

  K = np.asarray(camera["intrinsic_isaac"], dtype=np.float64).reshape(3, 3)
  # Isaac camera metadata uses an OpenGL camera convention: forward is -Z and
  # image Y points opposite camera Y.
  u = K[0, 2] + K[0, 0] * point_glcam[0] / depth
  v = K[1, 2] - K[1, 1] * point_glcam[1] / depth
  return np.asarray([u, v], dtype=np.float64)


def nonzero_segmentation_colors(segmentation):
  rgb = np.ascontiguousarray(segmentation[..., :3].astype(np.uint8, copy=False))
  colors = np.unique(rgb.reshape(-1, 3), axis=0)
  return [tuple(int(value) for value in color) for color in colors if np.any(color)]


def min_distance_to_color(segmentation_rgb, color, uv):
  ys, xs = np.where(np.all(segmentation_rgb == np.asarray(color), axis=-1))
  if len(xs) == 0:
    return np.inf
  distances = (xs.astype(np.float64) - uv[0]) ** 2
  distances += (ys.astype(np.float64) - uv[1]) ** 2
  return float(np.sqrt(distances.min()))


def match_class_ids_to_mask_colors(
    scene_class_ids,
    frame_metadata,
    camera,
    segmentation,
    max_distance,
):
  """Match GT-projected object centers to colored instance-mask regions."""
  object_positions = {
      item["class"]: item["translate"]
      for item in frame_metadata.get("objects", [])
      if "class" in item and "translate" in item
  }
  projected = []
  projected_classes = []
  for class_id in scene_class_ids:
    if class_id not in object_positions:
      logging.warning("Skipping %s: no object translation in frame metadata", class_id)
      continue
    uv = project_world_point(camera, object_positions[class_id])
    if uv is None:
      logging.warning("Skipping %s: object center is behind the camera", class_id)
      continue
    projected_classes.append(class_id)
    projected.append(uv)

  colors = nonzero_segmentation_colors(segmentation)
  if not projected_classes or not colors:
    return {}

  segmentation_rgb = segmentation[..., :3].astype(np.uint8, copy=False)
  costs = np.empty((len(projected_classes), len(colors)), dtype=np.float64)
  for row, uv in enumerate(projected):
    for col, color in enumerate(colors):
      costs[row, col] = min_distance_to_color(segmentation_rgb, color, uv)

  rows, cols = linear_sum_assignment(costs)
  matches = {}
  for row, col in zip(rows, cols):
    distance = costs[row, col]
    class_id = projected_classes[row]
    if distance > max_distance:
      logging.warning(
          "Skipping %s: nearest unique mask assignment is %.1f px away",
          class_id,
          distance,
      )
      continue
    matches[class_id] = colors[col]
    logging.info(
        "Mask match: %s -> RGB%s (projected distance %.1f px)",
        class_id,
        colors[col],
        distance,
    )
  return matches


def resolve_old_name_cad(cad_root, old_name):
  """Resolve only the canonical Old_name layout; never fall back."""
  object_dir = Path(cad_root).expanduser() / old_name
  mesh_file = object_dir / "edited" / f"{old_name}.obj"
  material_file = object_dir / "edited" / f"{old_name}.mtl"
  texture_file = object_dir / f"{old_name}_edited.bmp"
  missing = [
      path for path in (mesh_file, material_file, texture_file) if not path.is_file()
  ]
  if missing:
    return None, missing
  return {
      "object_dir": object_dir,
      "mesh_file": mesh_file,
      "material_file": material_file,
      "texture_file": texture_file,
  }, []


def draw_pose(original, K, pose, to_origin, bbox, extents, class_id, color):
  center_pose = pose @ np.linalg.inv(to_origin)
  linewidth = 7
  vis = draw_posed_3d_box(
      K,
      img=original,
      ob_in_cam=center_pose,
      bbox=bbox,
      line_color=color,
      linewidth=linewidth,
  )
  axis_scale = max(float(extents.max()) * 0.5, 0.01)
  vis = draw_xyz_axis(
      vis,
      ob_in_cam=center_pose,
      scale=axis_scale,
      K=K,
      thickness=linewidth,
      transparency=0,
      is_input_rgb=True,
  )
  uv = project_3d_to_2d(np.asarray([0, 0, 0, 1]), K, center_pose)
  cv2.putText(
      vis,
      class_id,
      tuple(int(value) for value in uv),
      cv2.FONT_HERSHEY_SIMPLEX,
      1.0,
      color,
      2,
      cv2.LINE_AA,
  )
  return vis


def main():
  args = build_parser().parse_args()
  if args.mesh_scale <= 0:
    raise ValueError("--mesh_scale must be positive")
  if args.max_image_size < 0 or args.max_texture_size < 0:
    raise ValueError("Image and texture size limits cannot be negative")
  if args.max_mask_match_distance < 0:
    raise ValueError("--max_mask_match_distance cannot be negative")

  set_logging_format()
  set_seed(0)
  scene_dir = Path(args.scene_dir).expanduser().resolve()
  frame_id = get_frame_id(scene_dir, args.camera_name, args.frame_id)
  frame_metadata = load_json(scene_dir / f"{frame_id}.json")
  scene_metadata = load_json(scene_dir / "scene_meta" / f"{frame_id}.json")
  scene_class_ids = list(scene_metadata.get("objects", {}).keys())
  if not scene_class_ids:
    raise ValueError(f"No objects found in scene metadata for frame {frame_id}")

  csv_path = (
      Path(args.objects_metadata).expanduser()
      if args.objects_metadata
      else scene_dir / "objects_metadata.csv"
  )
  old_names = load_old_name_catalog(csv_path)
  camera = get_camera(frame_metadata, args.camera_name)
  segmentation_path = (
      scene_dir / "masks" / args.camera_name / f"{frame_id}.png"
  )
  segmentation = imageio.imread(segmentation_path)
  if segmentation.ndim != 3 or segmentation.shape[2] < 3:
    raise ValueError(f"Expected colored segmentation at {segmentation_path}")
  mask_colors = match_class_ids_to_mask_colors(
      scene_class_ids,
      frame_metadata,
      camera,
      segmentation,
      args.max_mask_match_distance,
  )

  jobs = []
  for class_id in scene_class_ids:
    old_name = old_names.get(class_id)
    if not old_name:
      logging.warning("Skipping %s: Old_name is missing in %s", class_id, csv_path)
      continue
    cad, missing = resolve_old_name_cad(args.cad_root, old_name)
    if cad is None:
      logging.warning(
          "Skipping %s (%s): required Old_name CAD files are missing: %s",
          class_id,
          old_name,
          ", ".join(str(path) for path in missing),
      )
      continue
    if class_id not in mask_colors:
      logging.warning("Skipping %s (%s): mask color was not resolved", class_id, old_name)
      continue
    jobs.append((class_id, old_name, cad, mask_colors[class_id]))

  if not jobs:
    raise RuntimeError("No scene objects have both Old_name CAD files and a mask")
  logging.info(
      "Processing %d/%d scene objects: %s",
      len(jobs),
      len(scene_class_ids),
      [class_id for class_id, _, _, _ in jobs],
  )

  debug_camera_dir = (
      Path(args.debug_dir).expanduser().resolve() / args.camera_name
  )
  combined_dir = debug_camera_dir / "combined"
  combined_dir.mkdir(parents=True, exist_ok=True)

  # Network weights and rasterizer are object-independent and are loaded once.
  scorer = ScorePredictor()
  refiner = PoseRefinePredictor()
  glctx = dr.RasterizeCudaContext()
  combined_vis = None
  display_colors = [
      (0, 255, 0),
      (255, 128, 0),
      (255, 0, 255),
      (0, 255, 255),
      (255, 255, 0),
      (128, 128, 255),
  ]

  for object_index, (class_id, old_name, cad, mask_color) in enumerate(jobs):
    object_debug_dir = debug_camera_dir / class_id
    (object_debug_dir / "ob_in_cam").mkdir(parents=True, exist_ok=True)
    (object_debug_dir / "track_vis").mkdir(parents=True, exist_ok=True)
    logging.info("Registering %s using Old_name CAD %s", class_id, old_name)

    reader = CustomSceneReader(
        scene_dir=scene_dir,
        camera_name=args.camera_name,
        target_class=class_id,
        mask_color=mask_color,
        max_image_size=args.max_image_size,
    )
    frame_index = reader.id_strs.index(frame_id)
    rgb = reader.get_color(frame_index)
    depth = reader.get_depth(frame_index)
    mask = reader.get_mask(frame_index)
    K = reader.get_K(frame_index)

    with load_mesh_readonly(
        mesh_file=cad["mesh_file"],
        mesh_scale=args.mesh_scale,
        texture_roots=[cad["object_dir"]],
        max_texture_size=args.max_texture_size,
    ) as mesh:
      to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
      bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)
      estimator = FoundationPose(
          model_pts=mesh.vertices,
          model_normals=mesh.vertex_normals,
          mesh=mesh,
          scorer=scorer,
          refiner=refiner,
          debug_dir=str(object_debug_dir),
          debug=args.debug,
          glctx=glctx,
      )
      estimator.diameter = float(estimator.diameter)
      pose = estimator.register(
          K=K,
          rgb=rgb,
          depth=depth,
          ob_mask=mask,
          iteration=args.est_refine_iter,
      )
      np.savetxt(
          object_debug_dir / "ob_in_cam" / f"{frame_id}.txt",
          pose.reshape(4, 4),
      )

      original = reader.get_original_color(frame_index)
      original_K = reader.get_original_K(frame_index)
      line_color = display_colors[object_index % len(display_colors)]
      individual_vis = draw_pose(
          original.copy(),
          original_K,
          pose,
          to_origin,
          bbox,
          extents,
          class_id,
          line_color,
      )
      imageio.imwrite(
          object_debug_dir / "track_vis" / f"{frame_id}.png",
          individual_vis,
      )
      if combined_vis is None:
        combined_vis = original.copy()
      combined_vis = draw_pose(
          combined_vis,
          original_K,
          pose,
          to_origin,
          bbox,
          extents,
          class_id,
          line_color,
      )

    del estimator
    torch.cuda.empty_cache()

  imageio.imwrite(combined_dir / f"{frame_id}.png", combined_vis)
  logging.info("Combined result saved to %s", combined_dir / f"{frame_id}.png")


if __name__ == "__main__":
  main()
