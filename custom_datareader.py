"""Readers and asset helpers for custom FoundationPose scenes.

The source dataset is treated as read-only. Format conversion (colored
segmentation to a binary mask, mesh scaling, and MTL path resolution) happens
in memory or in a temporary directory.
"""

from __future__ import annotations

import json
import os
import shlex
import tempfile
from contextlib import contextmanager
from pathlib import Path

import cv2
import imageio
import numpy as np
import trimesh
from PIL import Image


class CustomSceneReader:
  """Read RGB PNG, float32 NPY depth, colored masks, and frame JSON metadata."""

  def __init__(
      self,
      scene_dir,
      camera_name="top_view_camera",
      target_class="obj_120",
      mask_color=(255, 25, 25),
      max_image_size=640,
  ):
    self.scene_dir = Path(scene_dir).expanduser().resolve()
    self.camera_name = camera_name
    self.target_class = target_class
    self.mask_color = np.asarray(mask_color, dtype=np.uint8).reshape(3)
    camera_rgb_dir = self.scene_dir / "rgb" / self.camera_name
    self.camera_subdirs = camera_rgb_dir.is_dir()
    self.data_dirs = {
        folder: (
            self.scene_dir / folder / self.camera_name
            if self.camera_subdirs
            else self.scene_dir / folder
        )
        for folder in ("rgb", "depth", "masks")
    }
    if self.camera_subdirs:
      missing = [
          str(path) for folder, path in self.data_dirs.items()
          if folder != "rgb" and not path.is_dir()
      ]
      if missing:
        raise FileNotFoundError(
            "Camera-specific RGB layout requires matching depth and masks "
            f"directories; missing: {missing}"
        )

    self.color_files = sorted(self.data_dirs["rgb"].glob("*.png"))
    if not self.color_files:
      raise FileNotFoundError(f"No RGB PNG files found in {self.data_dirs['rgb']}")

    self.id_strs = [path.stem for path in self.color_files]
    first = cv2.imread(str(self.color_files[0]), cv2.IMREAD_UNCHANGED)
    if first is None:
      raise ValueError(f"Could not read RGB image: {self.color_files[0]}")
    self.source_H, self.source_W = first.shape[:2]
    if max_image_size and max(self.source_H, self.source_W) > max_image_size:
      self.scale = float(max_image_size) / max(self.source_H, self.source_W)
    else:
      self.scale = 1.0
    self.H = int(round(self.source_H * self.scale))
    self.W = int(round(self.source_W * self.scale))

    scene_meta_path = self.scene_dir / "scene_meta" / f"{self.id_strs[0]}.json"
    if scene_meta_path.exists():
      with scene_meta_path.open("r", encoding="utf-8") as stream:
        scene_meta = json.load(stream)
      objects = scene_meta.get("objects", {})
      if self.target_class not in objects:
        raise ValueError(
            f"Target {self.target_class!r} is absent from {scene_meta_path}; "
            f"available targets: {sorted(objects)}"
        )

    # Preserve run_demo.py compatibility for callers which access reader.K.
    self.K = self.get_K(0)
    self._validate_frame(0)

  def __len__(self):
    return len(self.color_files)

  def _path(self, folder, i, suffix):
    return self.data_dirs[folder] / f"{self.id_strs[i]}{suffix}"

  def _frame_metadata(self, i):
    path = self.scene_dir / f"{self.id_strs[i]}.json"
    if not path.exists():
      raise FileNotFoundError(f"Frame metadata not found: {path}")
    with path.open("r", encoding="utf-8") as stream:
      return json.load(stream)

  def get_K(self, i):
    K = self.get_original_K(i)
    K[:2] *= self.scale
    return K

  def get_original_K(self, i):
    metadata = self._frame_metadata(i)
    cameras = metadata.get("cameras", [])
    matches = [camera for camera in cameras if camera.get("name") == self.camera_name]
    if len(matches) != 1:
      available = [camera.get("name") for camera in cameras]
      raise ValueError(
          f"Expected one camera named {self.camera_name!r}, found {len(matches)}; "
          f"available cameras: {available}"
      )
    K = np.asarray(matches[0]["intrinsic_isaac"], dtype=np.float32).reshape(3, 3)
    output_size = matches[0].get("output_size")
    if output_size is not None and tuple(output_size) != (self.source_W, self.source_H):
      raise ValueError(
          f"Camera output_size {tuple(output_size)} does not match RGB "
          f"source size {(self.source_W, self.source_H)}"
      )
    return K

  def get_original_color(self, i):
    color = imageio.imread(self.color_files[i])
    if color.ndim != 3 or color.shape[2] < 3:
      raise ValueError(f"RGB image must have at least 3 channels: {self.color_files[i]}")
    return np.ascontiguousarray(color[..., :3].astype(np.uint8, copy=False))

  def get_color(self, i):
    color = self.get_original_color(i)
    if self.scale != 1.0:
      color = cv2.resize(color, (self.W, self.H), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(color)

  def get_depth(self, i):
    path = self._path("depth", i, ".npy")
    if not path.exists():
      raise FileNotFoundError(f"Depth file not found: {path}")
    depth = np.load(path, allow_pickle=False)
    if depth.dtype != np.float32:
      depth = depth.astype(np.float32)
    if depth.ndim != 2:
      raise ValueError(f"Depth must be HxW, got {depth.shape} from {path}")
    if self.scale != 1.0:
      depth = cv2.resize(depth, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
    depth = np.ascontiguousarray(depth)
    depth[~np.isfinite(depth) | (depth < 0.001)] = 0
    return depth

  def get_mask(self, i):
    path = self._path("masks", i, ".png")
    if not path.exists():
      raise FileNotFoundError(f"Segmentation file not found: {path}")
    segmentation = imageio.imread(path)
    if segmentation.ndim != 3 or segmentation.shape[2] < 3:
      raise ValueError(f"Colored segmentation must have at least 3 channels: {path}")
    rgb = segmentation[..., :3].astype(np.uint8, copy=False)
    mask = np.all(rgb == self.mask_color, axis=-1).astype(np.uint8)
    if self.scale != 1.0:
      mask = cv2.resize(mask, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
    return np.ascontiguousarray(mask.astype(bool))

  def _validate_frame(self, i):
    color = self.get_color(i)
    depth = self.get_depth(i)
    mask = self.get_mask(i)
    if color.shape[:2] != depth.shape or depth.shape != mask.shape:
      raise ValueError(
          f"Frame {self.id_strs[i]} dimensions disagree: RGB={color.shape[:2]}, "
          f"depth={depth.shape}, mask={mask.shape}"
      )
    valid = mask & (depth >= 0.001)
    if mask.sum() == 0:
      raise ValueError(
          f"Mask color {tuple(int(v) for v in self.mask_color)} was not found "
          f"in frame {self.id_strs[i]}"
      )
    if valid.sum() < 4:
      raise ValueError(f"Frame {self.id_strs[i]} has fewer than 4 valid masked depth pixels")


def parse_path_mappings(values):
  """Parse repeated OLD=NEW path prefix mappings."""
  mappings = []
  for value in values or []:
    if "=" not in value:
      raise ValueError(f"Path mapping must be OLD=NEW, got: {value!r}")
    old, new = value.split("=", 1)
    if not old or not new:
      raise ValueError(f"Path mapping must have non-empty OLD and NEW: {value!r}")
    mappings.append((old.rstrip("/"), str(Path(new).expanduser().resolve()).rstrip("/")))
  return mappings


def _resolve_texture_path(raw_path, mesh_dir, path_mappings, texture_roots):
  expanded = os.path.expandvars(os.path.expanduser(raw_path))
  candidates = [Path(expanded)]

  for old, new in path_mappings:
    if expanded == old or expanded.startswith(old + "/"):
      suffix = expanded[len(old):].lstrip("/")
      candidates.append(Path(new) / suffix)

  # Dataset exports commonly retain a stale absolute map_Kd path while placing
  # the texture beside the OBJ. Prefer that local basename before broad roots.
  candidates.append(mesh_dir / Path(expanded).name)
  for root in texture_roots:
    root = Path(root).expanduser().resolve()
    candidates.extend([
        root / Path(expanded).name,
        root / Path(expanded).parent.name / Path(expanded).name,
    ])

  for candidate in candidates:
    if candidate.is_file():
      return candidate.resolve()
  rendered = "\n  ".join(str(candidate) for candidate in candidates)
  raise FileNotFoundError(f"Could not resolve texture {raw_path!r}; tried:\n  {rendered}")


def _runtime_mtl(original_mtl, runtime_dir, path_mappings, texture_roots):
  output_lines = []
  with original_mtl.open("r", encoding="utf-8") as stream:
    for line in stream:
      stripped = line.strip()
      if not stripped or stripped.startswith("#"):
        output_lines.append(line)
        continue
      tokens = shlex.split(stripped, comments=False, posix=True)
      if tokens and tokens[0].lower() == "map_kd" and len(tokens) >= 2:
        # The current dataset has no map options. Taking the last token also
        # handles the common option-bearing form, e.g. "map_Kd -s 1 1 1 x.png".
        source = _resolve_texture_path(
            tokens[-1], original_mtl.parent, path_mappings, texture_roots
        )
        link = runtime_dir / source.name
        if not link.exists():
          link.symlink_to(source)
        output_lines.append(f"map_Kd {link.name}\n")
      else:
        output_lines.append(line)
  runtime_mtl = runtime_dir / original_mtl.name
  runtime_mtl.write_text("".join(output_lines), encoding="utf-8")
  return runtime_mtl


@contextmanager
def load_mesh_readonly(
    mesh_file,
    mesh_scale=0.001,
    path_mappings=None,
    texture_roots=None,
    max_texture_size=4096,
):
  """Load a textured OBJ without editing OBJ, MTL, or texture source files."""
  mesh_file = Path(mesh_file).expanduser().resolve()
  if not mesh_file.is_file():
    raise FileNotFoundError(f"Mesh not found: {mesh_file}")
  original_mtl = mesh_file.with_suffix(".mtl")
  if not original_mtl.is_file():
    raise FileNotFoundError(f"MTL not found beside mesh: {original_mtl}")

  with tempfile.TemporaryDirectory(prefix="foundationpose_asset_") as temp:
    runtime_dir = Path(temp)
    (runtime_dir / mesh_file.name).symlink_to(mesh_file)
    _runtime_mtl(
        original_mtl,
        runtime_dir,
        path_mappings=path_mappings or [],
        texture_roots=texture_roots or [],
    )
    mesh = trimesh.load(runtime_dir / mesh_file.name, process=False)
    if isinstance(mesh, trimesh.Scene):
      if len(mesh.geometry) != 1:
        raise ValueError(
            f"Expected one mesh geometry, found {len(mesh.geometry)} in {mesh_file}"
        )
      mesh = next(iter(mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh):
      raise TypeError(f"Expected Trimesh, got {type(mesh).__name__} from {mesh_file}")

    mesh.apply_scale(float(mesh_scale))
    material = getattr(mesh.visual, "material", None)
    image = getattr(material, "image", None)
    uv = getattr(mesh.visual, "uv", None)
    if image is None or uv is None:
      raise ValueError(
          f"Textured mesh loading failed for {mesh_file}; verify its mtllib, "
          "map_Kd, UV coordinates, and texture path mappings"
      )
    if image is not None and max_texture_size and max(image.size) > max_texture_size:
      scale = float(max_texture_size) / max(image.size)
      resized = tuple(max(1, int(round(value * scale))) for value in image.size)
      material.image = image.resize(resized, Image.Resampling.LANCZOS)
    yield mesh
