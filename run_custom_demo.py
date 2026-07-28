"""Run the original FoundationPose demo behavior on a custom RGB-D scene."""

from __future__ import annotations

import argparse
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
    load_mesh_readonly,
    parse_path_mappings,
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
  parser.add_argument("--scene_dir", default=str(code_dir / "demo_data" / "cough"))
  parser.add_argument(
      "--mesh_file",
      default=None,
      help="Explicit OBJ override; otherwise cad_root/cad_name/edited/cad_name.obj.",
  )
  parser.add_argument(
      "--cad_root",
      default="/media/uon/data/3d_model/peel3_scan_data_2025",
      help="Root of the external CAD database as mounted inside the container.",
  )
  parser.add_argument("--cad_name", default="paper_cup")
  parser.add_argument("--target_class", default="obj_120")
  parser.add_argument("--camera_name", default="top_view_camera")
  parser.add_argument("--mask_color", nargs=3, type=int, default=(255, 25, 25))
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
      default=str(code_dir / "debug_cough"),
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
  if any(value < 0 or value > 255 for value in args.mask_color):
    raise ValueError("--mask_color values must be between 0 and 255")

  set_logging_format()
  set_seed(0)
  path_mappings = parse_path_mappings(args.path_map)
  cad_object_dir = Path(args.cad_root).expanduser() / args.cad_name
  if args.mesh_file:
    mesh_file = Path(args.mesh_file).expanduser()
    texture_roots = list(args.texture_root)
  else:
    # The Peel3 database stores edited geometry/material below edited/, while
    # the corresponding *_edited.bmp texture remains in the object directory.
    mesh_file = cad_object_dir / "edited" / f"{args.cad_name}.obj"
    texture_roots = [str(cad_object_dir), *args.texture_root]
  logging.info("CAD mesh: %s", mesh_file)

  debug_root = Path(args.debug_dir).expanduser().resolve()
  debug_dir = debug_root / args.camera_name
  (debug_dir / "track_vis").mkdir(parents=True, exist_ok=True)
  (debug_dir / "ob_in_cam").mkdir(parents=True, exist_ok=True)

  reader = CustomSceneReader(
      scene_dir=args.scene_dir,
      camera_name=args.camera_name,
      target_class=args.target_class,
      mask_color=args.mask_color,
      max_image_size=args.max_image_size,
  )
  logging.info(
      "Loaded %d frame(s), resolution=%dx%d, camera=%s, target=%s",
      len(reader), reader.W, reader.H, args.camera_name, args.target_class,
  )

  with load_mesh_readonly(
      mesh_file=mesh_file,
      mesh_scale=args.mesh_scale,
      path_mappings=path_mappings,
      texture_roots=texture_roots,
      max_texture_size=args.max_texture_size,
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
