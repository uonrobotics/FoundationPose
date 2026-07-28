"""Match synthetic scene object IDs to colors in instance segmentation images."""

from __future__ import annotations

import logging

import numpy as np
from scipy.optimize import linear_sum_assignment


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


def _nonzero_segmentation_colors(segmentation):
  rgb = np.ascontiguousarray(segmentation[..., :3].astype(np.uint8, copy=False))
  colors = np.unique(rgb.reshape(-1, 3), axis=0)
  return [tuple(int(value) for value in color) for color in colors if np.any(color)]


def _min_distance_to_color(segmentation_rgb, color, uv):
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

  colors = _nonzero_segmentation_colors(segmentation)
  if not projected_classes or not colors:
    return {}

  segmentation_rgb = segmentation[..., :3].astype(np.uint8, copy=False)
  costs = np.empty((len(projected_classes), len(colors)), dtype=np.float64)
  for row, uv in enumerate(projected):
    for col, color in enumerate(colors):
      costs[row, col] = _min_distance_to_color(segmentation_rgb, color, uv)

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
