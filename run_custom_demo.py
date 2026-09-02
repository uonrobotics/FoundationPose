"""Run the original FoundationPose demo behavior on a custom RGB-D scene."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import cv2
import imageio
import nvdiffrast.torch as dr
import numpy as np
import open3d as o3d
import trimesh

from custom_datareader import (
    CustomSceneReader,
    known_object_names_for_year,
    load_object_catalog,
    load_mesh_readonly,
    parse_path_mappings,
    resolve_cad_for_year,
)
from custom_mask_utils import (
    get_camera,
    match_class_ids_to_mask_colors,
)
from estimater import FoundationPose, PoseRefinePredictor, ScorePredictor
from Utils import (
    depth2xyzmap,
    draw_posed_3d_box,
    draw_xyz_axis,
    set_logging_format,
    set_seed,
    toOpen3dCloud,
)


def build_parser():
  code_dir = Path(__file__).resolve().parent
  parser = argparse.ArgumentParser(
      description="Run FoundationPose without modifying the source dataset."
  )
  parser.add_argument("--scene_dir", default=str(code_dir / "sample" / "cough"))
  parser.add_argument(
      "--mesh_file",
      default=None,
      help="Explicit OBJ override; takes precedence over all CAD name lookup.",
  )
  parser.add_argument(
      "--cad_root",
      default="/media/uon/data/3d_model/peel3_scan_data_2025",
      help="Root of the external CAD database as mounted inside the container.",
  )
  parser.add_argument(
      "--cad_name",
      default="paper_cup",
      help=(
          "Object name to find in objects_metadata.csv (Old_name first, then "
          "Object_name). Defaults to paper_cup."
      ),
  )
  parser.add_argument(
      "--objects_metadata",
      default=None,
      help="Defaults to <scene_dir>/objects_metadata.csv.",
  )
  parser.add_argument(
      "--target_class",
      default=None,
      help=(
          "Advanced class-ID override, primarily for --mesh_file. When using "
          "a database CAD, it must match the class resolved from --cad_name."
      ),
  )
  parser.add_argument("--camera_name", default="top_view_camera")
  parser.add_argument(
      "--mask_color",
      nargs=3,
      type=int,
      default=None,
      help="Optional RGB override; by default the instance color is detected automatically.",
  )
  parser.add_argument(
      "--max_mask_match_distance",
      type=float,
      default=250.0,
      help="Maximum source-image pixel distance for automatic mask-color matching.",
  )
  parser.add_argument(
      "--max_image_size",
      type=int,
      default=640,
      help="Resize the longest image side in memory; use 0 for native resolution.",
  )
  parser.add_argument("--mesh_scale", type=float, default=0.001)
  parser.add_argument(
      "--path_map",
      action="append",
      default=[],
      metavar="OLD=NEW",
      help="Repeatable prefix mapping for stale paths in MTL files.",
  )
  parser.add_argument(
      "--texture_root",
      action="append",
      default=[],
      help="Additional directory in which to resolve texture files.",
  )
  parser.add_argument(
      "--max_texture_size",
      type=int,
      default=4096,
      help="Resize texture in memory only; use 0 to keep its original size.",
  )
  parser.add_argument("--est_refine_iter", type=int, default=5)
  parser.add_argument("--track_refine_iter", type=int, default=2)
  parser.add_argument("--debug", type=int, default=2)
  parser.add_argument(
      "--debug_dir",
      default=str(code_dir / "outputs"),
      help="Common output root; camera_name is appended automatically.",
  )
  parser.add_argument("--no_gui", action="store_true")
  return parser


def main():
  args = build_parser().parse_args()
  if args.mesh_scale <= 0:
    raise ValueError("--mesh_scale must be positive")
  if args.max_texture_size < 0:
    raise ValueError("--max_texture_size cannot be negative")
  if args.max_image_size < 0:
    raise ValueError("--max_image_size cannot be negative")
  if args.mask_color and any(value < 0 or value > 255 for value in args.mask_color):
    raise ValueError("--mask_color values must be between 0 and 255")
  if args.max_mask_match_distance < 0:
    raise ValueError("--max_mask_match_distance cannot be negative")

  set_logging_format()
  set_seed(0)
  path_mappings = parse_path_mappings(args.path_map)
  scene_dir = Path(args.scene_dir).expanduser().resolve()
  csv_path = (
      Path(args.objects_metadata).expanduser()
      if args.objects_metadata
      else scene_dir / "objects_metadata.csv"
  )
  object_catalog = load_object_catalog(csv_path)

  # A single-object run is selected by its human-readable CAD name. Resolve
  # Old_name before Object_name, then use the matched Class_name internally
  # for scene validation and mask extraction.
  matches = [
      class_id for class_id, names in object_catalog.items()
      if names["old_name"] == args.cad_name
  ]
  matched_field = "Old_name"
  if not matches:
    matches = [
        class_id for class_id, names in object_catalog.items()
        if names["object_name"] == args.cad_name
    ]
    matched_field = "Object_name"
  if len(matches) > 1:
    raise ValueError(
        f"CAD name {args.cad_name!r} is ambiguous in {csv_path}: {matches}"
    )
  resolved_class = matches[0] if matches else None

  if args.target_class:
    if resolved_class and args.target_class != resolved_class and not args.mesh_file:
      raise ValueError(
          f"--cad_name {args.cad_name!r} resolves to {resolved_class}, but "
          f"--target_class specifies {args.target_class}"
      )
    target_class = args.target_class
  elif resolved_class:
    target_class = resolved_class
  else:
    raise ValueError(
        f"CAD name {args.cad_name!r} was not found in Old_name or Object_name "
        f"of {csv_path}; specify --target_class only when using --mesh_file"
    )

  rgb_dir = scene_dir / "rgb" / args.camera_name
  rgb_files = sorted(rgb_dir.glob("*.png"))
  if not rgb_files:
    raise FileNotFoundError(f"No RGB PNG files found in {rgb_dir}")
  registration_frame_id = rgb_files[0].stem

  debug_root = Path(args.debug_dir).expanduser().resolve()
  debug_dir = debug_root / args.camera_name
  (debug_dir / "track_vis").mkdir(parents=True, exist_ok=True)
  (debug_dir / "ob_in_cam").mkdir(parents=True, exist_ok=True)

  if args.mesh_file:
    mesh_file = Path(args.mesh_file).expanduser()
    texture_roots = list(args.texture_root)
    expected_texture_file = None
    selected_cad_name = mesh_file.stem
    known_names = set()
    logging.info("Using explicit mesh override: %s", mesh_file)
  else:
    names = object_catalog[target_class]
    cad, attempts = resolve_cad_for_year(args.cad_root, names["year"], names)
    if cad is None:
      attempted = ", ".join(
          f"{item['source_field']}={item['name']!r}"
          for item in attempts
      ) or "no non-empty Old_name/Object_name"
      raise FileNotFoundError(
          f"No complete CAD asset for {target_class} (year={names['year']}); "
          f"tried {attempted}"
      )
    mesh_file = cad["mesh_file"]
    texture_roots = [str(cad["object_dir"]), *args.texture_root]
    expected_texture_file = cad["texture_file"]
    selected_cad_name = cad["name"]
    known_names = known_object_names_for_year(object_catalog, names["year"])
    logging.info(
        "Selected %s via %s=%s; resolved CAD asset using %s=%s",
        target_class,
        matched_field,
        args.cad_name,
        cad["source_field"],
        cad["name"],
    )
  logging.info("CAD mesh: %s", mesh_file)

  if args.mask_color:
    mask_color = tuple(args.mask_color)
    logging.info("Using explicit mask color override: RGB%s", mask_color)
  else:
    frame_id = registration_frame_id
    with (
        scene_dir / "conf" / f"{frame_id}.json"
    ).open("r", encoding="utf-8") as stream:
      frame_metadata = json.load(stream)
    scene_meta_path = scene_dir / "scene_meta" / f"{frame_id}.json"
    with scene_meta_path.open("r", encoding="utf-8") as stream:
      scene_metadata = json.load(stream)
    scene_class_ids = list(scene_metadata.get("objects", {}).keys())
    segmentation_path = (
        scene_dir / "masks" / args.camera_name / f"{frame_id}.png"
    )
    segmentation = imageio.imread(segmentation_path)
    if segmentation.ndim != 3 or segmentation.shape[2] < 3:
      raise ValueError(f"Expected colored segmentation at {segmentation_path}")
    mask_colors = match_class_ids_to_mask_colors(
        scene_class_ids,
        frame_metadata,
        get_camera(frame_metadata, args.camera_name),
        segmentation,
        args.max_mask_match_distance,
    )
    if target_class not in mask_colors:
      raise ValueError(
          f"Could not automatically match {target_class} to a color in "
          f"{segmentation_path}; use --mask_color R G B to override"
      )
    mask_color = mask_colors[target_class]
    logging.info("Automatically selected mask color RGB%s", mask_color)

  reader = CustomSceneReader(
      scene_dir=args.scene_dir,
      camera_name=args.camera_name,
      target_class=target_class,
      mask_color=mask_color,
      max_image_size=args.max_image_size,
  )
  logging.info(
      "Loaded %d frame(s), resolution=%dx%d, camera=%s, target=%s",
      len(reader), reader.W, reader.H, args.camera_name, target_class,
  )

  with load_mesh_readonly(
      mesh_file=mesh_file,
      identity_name=selected_cad_name,
      known_names=known_names,
      mesh_scale=args.mesh_scale,
      path_mappings=path_mappings,
      texture_roots=texture_roots,
      max_texture_size=args.max_texture_size,
      expected_texture_file=expected_texture_file,
      reject_nonstandard_texture=True,
  ) as mesh:
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)
    logging.info("Mesh extents after scale: %s meters", extents)

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    estimator = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        debug_dir=str(debug_dir),
        debug=args.debug,
        glctx=glctx,
    )
    # compute_mesh_diameter returns a NumPy float64. In the upstream crop
    # helper that scalar can promote offsets to float64 while K remains
    # float32. Keep this compatibility fix local to the custom runner.
    estimator.diameter = float(estimator.diameter)
    logging.info("Estimator initialization done")

    for i in range(len(reader)):
      frame_id = reader.id_strs[i]
      logging.info("Frame %s", frame_id)
      color = reader.get_color(i)
      depth = reader.get_depth(i)
      K = reader.get_K(i)
      if i == 0:
        mask = reader.get_mask(i)
        pose = estimator.register(
            K=K,
            rgb=color,
            depth=depth,
            ob_mask=mask,
            iteration=args.est_refine_iter,
        )
        if args.debug >= 3:
          transformed = mesh.copy()
          transformed.apply_transform(pose)
          transformed.export(debug_dir / "model_tf.obj")
          xyz_map = depth2xyzmap(depth, K)
          valid = depth >= 0.001
          cloud = toOpen3dCloud(xyz_map[valid], color[valid])
          o3d.io.write_point_cloud(str(debug_dir / "scene_complete.ply"), cloud)
      else:
        pose = estimator.track_one(
            rgb=color,
            depth=depth,
            K=K,
            iteration=args.track_refine_iter,
        )

      np.savetxt(debug_dir / "ob_in_cam" / f"{frame_id}.txt", pose.reshape(4, 4))
      if args.debug >= 1:
        center_pose = pose @ np.linalg.inv(to_origin)
        vis = draw_posed_3d_box(K, img=color, ob_in_cam=center_pose, bbox=bbox)
        axis_scale = max(float(extents.max()) * 0.5, 0.01)
        vis = draw_xyz_axis(
            color,
            ob_in_cam=center_pose,
            scale=axis_scale,
            K=K,
            thickness=3,
            transparency=0,
            is_input_rgb=True,
        )
        if not args.no_gui:
          cv2.imshow("FoundationPose", vis[..., ::-1])
          cv2.waitKey(1)
        if args.debug >= 2:
          # Inference uses resized RGB-D to reduce the scorer's GPU memory
          # footprint, but ob_in_cam is a metric 3D pose and is independent of
          # image resolution. Render that same pose with the source RGB and
          # source intrinsic matrix so the saved user-facing result remains at
          # the dataset's original resolution. The resized visualization is
          # intentionally not saved; model diagnostics such as color.png,
          # depth.png, and vis_score.png still describe the inference input.
          original_color = reader.get_original_color(i)
          original_K = reader.get_original_K(i)
          original_vis = draw_posed_3d_box(
              original_K,
              img=original_color,
              ob_in_cam=center_pose,
              bbox=bbox,
              line_color=(0, 255, 0),
              linewidth=max(3, int(round(3 / reader.scale))),
          )
          original_vis = draw_xyz_axis(
              original_vis,
              ob_in_cam=center_pose,
              scale=axis_scale,
              K=original_K,
              thickness=max(3, int(round(3 / reader.scale))),
              transparency=0,
              is_input_rgb=True,
          )
          imageio.imwrite(
              debug_dir / "track_vis" / f"{frame_id}.png",
              original_vis,
          )

  if not args.no_gui:
    cv2.destroyAllWindows()
  logging.info("Results saved to %s", debug_dir)


if __name__ == "__main__":
  main()
