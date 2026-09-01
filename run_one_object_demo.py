"""Scene의 frame마다, 그 frame에 있는 객체 1개의 6D pose를 계산한다.

run_multi_object_demo.py와 거의 같고, 객체가 항상 1개라는 점만 다르다.
카메라 한 대만 쓰고, 추적 없이 frame마다 새로 계산한다.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import imageio
import nvdiffrast.torch as dr
import numpy as np
import torch
import trimesh

from custom_datareader import (
    CustomSceneReader,
    append_issue_jsonl,
    atomic_write_bytes,
    detect_domain,
    draw_pose,
    get_frame_ids,
    load_json,
    load_mesh_readonly,
    load_object_catalog,
    resolve_cad_by_year,
    validate_frame_files,
)
from custom_mask_utils import (
    get_camera,
    match_class_ids_to_mask_colors,
    match_class_ids_to_mask_colors_from_semantics_mapping,
)
from estimater import FoundationPose, PoseRefinePredictor, ScorePredictor
from Utils import set_logging_format, set_seed


def build_parser():
  parser = argparse.ArgumentParser(
      description=(
          "Batch 6D pose estimation over one scene, one object per frame, "
          "no temporal tracking."
      )
  )
  parser.add_argument(
      "--dataset-root",
      type=Path,
      required=True,
      help="Top-level dataset folder that holds objects_metadata.csv, e.g. "
      "/media/uon/data1/gemini",
  )
  parser.add_argument(
      "--scene",
      required=True,
      help="Scene path relative to --dataset-root, e.g. "
      "real_v1/home/LivingRoom_Kitchen/dining_table",
  )
  parser.add_argument(
      "--camera_name",
      default="top_view_camera",
      help="Pose is only meaningful in one camera's own coordinate frame "
      "(no inter-camera calibration yet), so only one camera is processed "
      "per run.",
  )
  parser.add_argument(
      "--cad-root",
      default="/media/uon/data1/3d_model",
      help="Parent of the year-specific CAD folders "
      "(<cad-root>/peel3_scan_data_<Year>), matched to each object's own "
      "catalog Year.",
  )
  parser.add_argument(
      "--objects-metadata",
      type=Path,
      default=None,
      help="Override the catalog CSV path (default: "
      "<dataset-root>/objects_metadata.csv).",
  )
  parser.add_argument(
      "--frame_id",
      default=None,
      help="Optional frame stem; when omitted, process every RGB frame.",
  )
  parser.add_argument("--mesh_scale", type=float, default=0.001)
  parser.add_argument("--max_image_size", type=int, default=640)
  parser.add_argument("--max_texture_size", type=int, default=4096)
  parser.add_argument(
      "--max_mask_match_distance",
      type=float,
      default=250.0,
      help="Virtual domain only: maximum source-image pixel distance for "
      "GT center to mask-color matching.",
  )
  parser.add_argument("--est_refine_iter", type=int, default=5)
  parser.add_argument(
      "--debug",
      type=int,
      default=2,
      help="FoundationPose internal debug verbosity, written under "
      "6d_pose_debug/<frame_id>/.",
  )
  parser.add_argument(
      "--save-diagnostics",
      action=argparse.BooleanOptionalAction,
      default=True,
      help="Write a per-frame pose visualization under diagnostics/ "
      "(default: enabled).",
  )
  return parser


def discover_work_items(scene_dir, camera_name, requested_frame_id):
  """Build an ordered frame_id worklist for one camera."""
  ordered = []
  for frame_id in get_frame_ids(scene_dir, camera_name, requested_frame_id):
    try:
      sort_key = int(frame_id)
    except ValueError:
      print(
          f"run-one-object-demo: skipping malformed frame id {frame_id!r}",
          file=sys.stderr,
      )
      continue
    ordered.append((sort_key, frame_id))
  # 숫자로 정렬: 문자열로 정렬하면 "10000"이 "9999"보다 앞에 온다.
  ordered.sort()
  return [frame_id for _, frame_id in ordered]


def frame_is_done(scene_dir, frame_id):
  pose_txt = scene_dir / "6d_pose" / f"{frame_id}.txt"
  pose_json = scene_dir / "6d_pose_json" / f"{frame_id}.json"
  return pose_txt.is_file() and pose_json.is_file()


def log_issue(issues_path, *, camera_name, frame_id, class_id, issue, message):
  append_issue_jsonl(issues_path, {
      "logged_at_utc": datetime.now(timezone.utc).isoformat(),
      "camera_name": camera_name,
      "frame_id": frame_id,
      "class_id": class_id,
      "issue": issue,
      "message": message,
  })


_ANSI_RESET = "\033[0m"
_ANSI_BOLD_GREEN = "\033[1;32m"
_ANSI_BOLD_RED = "\033[1;31m"


def _highlight(text, ansi_code):
  """TTY에 출력될 때만 색을 입힌다 (파일로 리다이렉트되면 평문 그대로)."""
  if not sys.stdout.isatty():
    return text
  return f"{ansi_code}{text}{_ANSI_RESET}"


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

  dataset_root = args.dataset_root.expanduser().resolve()
  metadata_path = (
      args.objects_metadata.expanduser().resolve()
      if args.objects_metadata is not None
      else dataset_root / "objects_metadata.csv"
  )
  if not metadata_path.is_file():
    raise FileNotFoundError(f"objects_metadata.csv not found: {metadata_path}")
  scene_dir = (dataset_root / args.scene).resolve()
  camera_name = args.camera_name

  object_catalog = load_object_catalog(metadata_path)
  work_items = discover_work_items(scene_dir, camera_name, args.frame_id)
  if not work_items:
    logging.info("No frames to process in %s [%s]", scene_dir, camera_name)
    return

  issues_path = scene_dir / "inference_meta" / "foundationpose" / "cad_asset_issues.jsonl"

  scorer = ScorePredictor()
  refiner = PoseRefinePredictor()
  glctx = dr.RasterizeCudaContext()
  cad_cache = {}

  processed = skipped = failed = 0

  for frame_id in work_items:
    if frame_is_done(scene_dir, frame_id):
      skipped += 1
      continue

    class_id = None
    try:
      domain = detect_domain(scene_dir, frame_id)
      mask_dir_name = "inst_seg" if domain == "real" else "masks"
      depth_suffix = ".png" if domain == "real" else ".npy"
      validate_frame_files(scene_dir, camera_name, [frame_id], domain)

      scene_metadata = load_json(scene_dir / "scene_meta" / f"{frame_id}.json")
      scene_objects = scene_metadata.get("objects", {})
      if not scene_objects:
        logging.info("%s: no objects in scene_meta; skipping", frame_id)
        skipped += 1
        continue
      if len(scene_objects) > 1:
        raise ValueError(
            f"{frame_id} has {len(scene_objects)} objects; use "
            "run_multi_object_demo.py for multi-object frames"
        )
      class_id, object_info = next(iter(scene_objects.items()))
      object_name = object_info.get("object_name")

      names = object_catalog.get(class_id)
      if names is None:
        raise ValueError(f"{class_id} is missing from {metadata_path}")

      if class_id in cad_cache:
        cad, attempts = cad_cache[class_id]
      else:
        cad, attempts = resolve_cad_by_year(args.cad_root, names["year"], names)
        cad_cache[class_id] = (cad, attempts)
      if cad is None:
        attempted = ", ".join(
            f"{item['source_field']}={item['name']!r}" for item in attempts
        ) or "no candidates"
        raise FileNotFoundError(
            f"No CAD asset for {class_id} (year={names['year']}); "
            f"tried {attempted}"
        )

      if domain == "virtual":
        frame_metadata = load_json(scene_dir / "conf" / f"{frame_id}.json")
        segmentation_path = scene_dir / mask_dir_name / camera_name / f"{frame_id}.png"
        segmentation = imageio.imread(segmentation_path)
        if segmentation.ndim != 3 or segmentation.shape[2] < 3:
          raise ValueError(f"Expected colored segmentation at {segmentation_path}")
        mask_colors = match_class_ids_to_mask_colors(
            [class_id],
            frame_metadata,
            get_camera(frame_metadata, camera_name),
            segmentation,
            args.max_mask_match_distance,
        )
      else:
        mask_colors = match_class_ids_to_mask_colors_from_semantics_mapping(
            [class_id], scene_dir, camera_name, frame_id, mask_dir_name
        )
      if class_id not in mask_colors:
        raise ValueError(
            f"{class_id}: mask color not resolved for frame {frame_id}"
        )

      reader = CustomSceneReader(
          scene_dir=scene_dir,
          camera_name=camera_name,
          target_class=class_id,
          mask_color=mask_colors[class_id],
          max_image_size=args.max_image_size,
          validation_frame_id=frame_id,
          mask_dir_name=mask_dir_name,
          depth_suffix=depth_suffix,
      )
      frame_index = reader.id_strs.index(frame_id)
      rgb = reader.get_color(frame_index)
      depth = reader.get_depth(frame_index)
      mask = reader.get_mask(frame_index)
      K = reader.get_K(frame_index)

      pose_debug_dir = scene_dir / "6d_pose_debug" / frame_id
      texture_diagnostics = []
      with load_mesh_readonly(
          mesh_file=cad["mesh_file"],
          identity_name=cad["name"],
          mesh_scale=args.mesh_scale,
          texture_roots=[cad["object_dir"]],
          max_texture_size=args.max_texture_size,
          expected_texture_file=cad["texture_file"],
          texture_diagnostics=texture_diagnostics,
          reject_nonstandard_texture=True,
      ) as mesh:
        to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
        bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)
        estimator = FoundationPose(
            model_pts=mesh.vertices,
            model_normals=mesh.vertex_normals,
            mesh=mesh,
            scorer=scorer,
            refiner=refiner,
            debug_dir=str(pose_debug_dir),
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

        pose_matrix = pose.reshape(4, 4)
        pose_txt_path = scene_dir / "6d_pose" / f"{frame_id}.txt"
        txt_buffer = io.StringIO()
        np.savetxt(txt_buffer, pose_matrix)
        atomic_write_bytes(pose_txt_path, txt_buffer.getvalue().encode("utf-8"))

        pose_json_path = scene_dir / "6d_pose_json" / f"{frame_id}.json"
        pose_json_payload = json.dumps(
            {
                "object_id": class_id,
                "object_name": object_name,
                "ob_in_cam": pose_matrix.tolist(),
            },
            indent=2,
        ).encode("utf-8")
        atomic_write_bytes(pose_json_path, pose_json_payload)

        if args.save_diagnostics:
          original = reader.get_original_color(frame_index)
          original_K = reader.get_original_K(frame_index)
          vis = draw_pose(
              original, original_K, pose, to_origin, bbox, extents,
              class_id, (0, 255, 0),
          )
          track_vis_path = (
              scene_dir / "diagnostics" / "foundationpose" / "track_vis"
              / f"{frame_id}.png"
          )
          track_vis_path.parent.mkdir(parents=True, exist_ok=True)
          imageio.imwrite(track_vis_path, vis)

      del estimator
      for issue in texture_diagnostics:
        log_issue(
            issues_path,
            camera_name=camera_name,
            frame_id=frame_id,
            class_id=class_id,
            issue=issue.get("issue", "texture_diagnostic"),
            message=str(issue),
        )
      torch.cuda.empty_cache()

      processed += 1
      logging.info("%s [%s] %s: pose saved", frame_id, camera_name, class_id)

    except Exception as error:  # noqa: BLE001 - per-frame isolation is required
      failed += 1
      log_issue(
          issues_path,
          camera_name=camera_name,
          frame_id=frame_id,
          class_id=class_id,
          issue=type(error).__name__,
          message=str(error),
      )
      logging.warning("%s [%s]: FAILED (%s)", frame_id, camera_name, error)
      torch.cuda.empty_cache()
      continue

  summary_line = (
      f"Processed {processed}, skipped {skipped}, failed {failed} "
      f"(of {len(work_items)} work item(s)). Scene: {scene_dir} [{camera_name}]"
  )
  logging.info(_highlight(summary_line, _ANSI_BOLD_RED if failed else _ANSI_BOLD_GREEN))


if __name__ == "__main__":
  main()
