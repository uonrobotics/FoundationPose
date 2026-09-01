"""Readers and asset helpers for custom FoundationPose scenes.

The source dataset is treated as read-only. Format conversion (colored
segmentation to a binary mask, mesh scaling, and MTL path resolution) happens
in memory or in a temporary directory.
"""

from __future__ import annotations

import csv
import json
import os
import shlex
import tempfile
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

import cv2
import imageio
import numpy as np
import trimesh
from PIL import Image
from trimesh import resolvers

from Utils import draw_posed_3d_box, draw_xyz_axis, project_3d_to_2d


class NonstandardMapKdError(ValueError):
  """Raised when strict CAD loading rejects a nonstandard texture name."""


def load_object_catalog(csv_path):
  """Load CAD lookup names keyed by the synthetic scene class ID."""
  csv_path = Path(csv_path).expanduser()
  catalog = {}
  with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
    for row in csv.DictReader(stream):
      class_id = row.get("Class_name", "").strip()
      if not class_id:
        continue
      catalog[class_id] = {
          "old_name": row.get("Old_name", "").strip(),
          "object_name": row.get("Object_name", "").strip(),
          "year": row.get("Year", "").strip(),
      }
  return catalog


# mesh/mtl 위치는 모든 연도가 동일. 텍스처는 위치를 강제하지 않고 mtl 안 map_Kd를 따른다.
_CAD_MESH_TEMPLATE = "edited/{name}.obj"
_CAD_MATERIAL_TEMPLATE = "edited/{name}.mtl"


def _texture_from_mtl(mesh_file, material_file, object_dir):
  """Return the texture path for the OBJ's own identity material — the one
  whose texture filename (minus an optional _edited suffix) matches
  object_dir's own name. Not face count: a combined-scan sibling can have
  more faces than the object's own part. None if unresolvable."""
  if not material_file.is_file() or not Path(mesh_file).is_file():
    return None
  used_material_names = _used_material_names(mesh_file)
  if not used_material_names:
    return None
  textures = _used_material_textures(material_file, used_material_names)
  distinct_textures = set(textures.values())
  if not distinct_textures:
    return None
  if len(distinct_textures) == 1:
    texture_name = next(iter(distinct_textures))
  else:
    identity_name = object_dir.name
    matches = {
        texture for texture in distinct_textures
        if Path(texture).stem == identity_name
        or Path(texture).stem.removesuffix("_edited") == identity_name
    }
    if len(matches) != 1:
      return None
    texture_name = next(iter(matches))
  next_to_mtl = material_file.parent / texture_name
  at_object_root = object_dir / texture_name
  return next_to_mtl if next_to_mtl.is_file() else at_object_root


def resolve_cad_for_year(cad_root, year, names):
  """Resolve a CAD asset by whichever of Old_name/Object_name matches (in
  the real catalog, at most one of the two ever does — there's no actual
  priority between them)."""
  cad_root = Path(cad_root).expanduser()
  attempts = []
  seen = set()
  for source_field, name in (
      ("Old_name", names.get("old_name", "")),
      ("Object_name", names.get("object_name", "")),
  ):
    if not name or name in seen:
      continue
    seen.add(name)
    object_dir = cad_root / name
    mesh_file = object_dir / _CAD_MESH_TEMPLATE.format(name=name)
    material_file = object_dir / _CAD_MATERIAL_TEMPLATE.format(name=name)
    texture_file = _texture_from_mtl(mesh_file, material_file, object_dir)
    missing = [path for path in (mesh_file, material_file) if not path.is_file()]
    if texture_file is None or not texture_file.is_file():
      missing.append(texture_file or material_file)
    attempts.append({
        "source_field": source_field,
        "name": name,
        "missing": missing,
    })
    if not missing:
      return {
          "source_field": source_field,
          "name": name,
          "object_dir": object_dir,
          "mesh_file": mesh_file,
          "material_file": material_file,
          "texture_file": texture_file,
      }, attempts
  return None, attempts


def resolve_cad_by_year(cad_root, year, names):
  """Resolve a CAD asset from ``<cad_root>/peel3_scan_data_<year>/``.

  Real-domain objects are scanned into a year-specific folder
  (2024/2025/2026 under one shared ``cad_root``); this only searches the
  folder matching the object's own catalog year, never other years.
  """
  year_root = Path(cad_root).expanduser() / f"peel3_scan_data_{year}"
  if not year_root.is_dir():
    return None, [{
        "source_field": "cad_root",
        "name": str(year_root),
        "missing": [year_root],
    }]
  return resolve_cad_for_year(year_root, year, names)


def append_cad_asset_issues(log_path, issues, frame_id, class_id, cad_name):
  """Write compact, de-duplicated CAD texture issues for the current run."""
  if not issues:
    return
  log_path = Path(log_path)
  if log_path.is_file():
    with log_path.open("r", encoding="utf-8") as stream:
      records = json.load(stream)
  else:
    records = []
  for issue in issues:
    if issue["issue"] == "nonstandard_map_kd":
      message = (
          "MTL texture name does not match the object's own identity name. "
          "This object was skipped."
      )
    else:
      message = (
          "The texture referenced by map_Kd does not exist. "
          "This object was skipped."
      )
    record = {
        "frame_id": frame_id,
        "class_id": class_id,
        "cad_name": cad_name,
        "issue": issue["issue"],
        "expected_texture": issue["expected_texture"],
        "resolved_texture": issue["resolved_texture"],
        "message": message,
    }
    if record not in records:
      records.append(record)
  with log_path.open("w", encoding="utf-8") as stream:
    json.dump(records, stream, ensure_ascii=False, indent=2)
    stream.write("\n")


def append_issue_jsonl(log_path, record):
  """Append one JSON record as a line, creating parent dirs as needed."""
  log_path = Path(log_path)
  log_path.parent.mkdir(parents=True, exist_ok=True)
  with log_path.open("a", encoding="utf-8") as stream:
    stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def atomic_write_bytes(path, payload):
  """Write bytes via a temp file + rename so a crash mid-write can never
  leave a corrupted (partially written) file behind."""
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  temp_path = path.parent / f".{path.name}.{uuid4().hex}.tmp"
  try:
    descriptor = os.open(temp_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o664)
    with os.fdopen(descriptor, "wb") as stream:
      stream.write(payload)
      stream.flush()
      os.fsync(stream.fileno())
    os.replace(temp_path, path)
  finally:
    temp_path.unlink(missing_ok=True)


def load_json(path):
  with Path(path).open("r", encoding="utf-8") as stream:
    return json.load(stream)


def get_frame_ids(scene_dir, camera_name, requested):
  rgb_dir = Path(scene_dir) / "rgb" / camera_name
  files = sorted(rgb_dir.glob("*.png"))
  if not files:
    raise FileNotFoundError(f"No RGB PNG files found in {rgb_dir}")
  if requested is None:
    return [path.stem for path in files]
  path = rgb_dir / f"{requested}.png"
  if not path.is_file():
    raise FileNotFoundError(f"Requested RGB frame not found: {path}")
  return [requested]


def detect_domain(scene_dir, frame_id):
  """Real captures mark themselves with conf.json's "domain": "real"."""
  conf_path = Path(scene_dir) / "conf" / f"{frame_id}.json"
  with conf_path.open("r", encoding="utf-8") as stream:
    conf = json.load(stream)
  return "real" if conf.get("domain") == "real" else "virtual"


def validate_frame_files(scene_dir, camera_name, frame_ids, domain):
  """Fail before model loading when any selected frame input is incomplete."""
  mask_dir_name = "inst_seg" if domain == "real" else "masks"
  depth_suffix = ".png" if domain == "real" else ".npy"
  required = []
  for frame_id in frame_ids:
    required.extend([
        scene_dir / "depth" / camera_name / f"{frame_id}{depth_suffix}",
        scene_dir / mask_dir_name / camera_name / f"{frame_id}.png",
        scene_dir / "conf" / f"{frame_id}.json",
        scene_dir / "scene_meta" / f"{frame_id}.json",
    ])
    if domain == "real":
      required.append(
          scene_dir / mask_dir_name / camera_name
          / f"semantics_mapping_{frame_id}.json"
      )
  missing = [path for path in required if not path.is_file()]
  if missing:
    rendered = "\n  ".join(str(path) for path in missing)
    raise FileNotFoundError(
        "Selected RGB frames have missing corresponding input files:\n  "
        f"{rendered}"
    )


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


class CustomSceneReader:
  """Read RGB PNG, depth (NPY or PNG), colored masks, and frame JSON metadata.

  Virtual (Isaac-sim) scenes store depth as float32-meter .npy and colored
  masks under a "masks" folder. Real captures store depth as uint16-mm .png
  (aligned to the color frame) and colored masks under an "inst_seg" folder;
  pass depth_suffix=".png" and mask_dir_name="inst_seg" for those.
  """

  def __init__(
      self,
      scene_dir,
      camera_name="top_view_camera",
      target_class="obj_120",
      mask_color=(255, 25, 25),
      max_image_size=640,
      validation_frame_id=None,
      mask_dir_name="masks",
      depth_suffix=".npy",
  ):
    self.scene_dir = Path(scene_dir).expanduser().resolve()
    self.camera_name = camera_name
    self.target_class = target_class
    self.mask_color = np.asarray(mask_color, dtype=np.uint8).reshape(3)
    self.mask_dir_name = mask_dir_name
    self.depth_suffix = depth_suffix
    camera_rgb_dir = self.scene_dir / "rgb" / self.camera_name
    self.camera_subdirs = camera_rgb_dir.is_dir()
    folder_dir_names = {"rgb": "rgb", "depth": "depth", "masks": mask_dir_name}
    self.data_dirs = {
        key: (
            self.scene_dir / dirname / self.camera_name
            if self.camera_subdirs
            else self.scene_dir / dirname
        )
        for key, dirname in folder_dir_names.items()
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

    if validation_frame_id is None:
      validation_i = 0
    else:
      try:
        validation_i = self.id_strs.index(str(validation_frame_id))
      except ValueError as error:
        raise ValueError(
            f"Validation frame {validation_frame_id!r} is absent from "
            f"{self.data_dirs['rgb']}"
        ) from error

    scene_meta_path = (
        self.scene_dir / "scene_meta" / f"{self.id_strs[validation_i]}.json"
    )
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
    self.K = self.get_K(validation_i)
    self._validate_frame(validation_i)

  def __len__(self):
    return len(self.color_files)

  def _path(self, folder, i, suffix):
    return self.data_dirs[folder] / f"{self.id_strs[i]}{suffix}"

  def _frame_metadata(self, i):
    path = self.scene_dir / "conf" / f"{self.id_strs[i]}.json"
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
    intrinsic = matches[0].get("intrinsic_isaac", matches[0].get("intrinsic"))
    if intrinsic is None:
      raise KeyError(
          f"Camera {self.camera_name!r} entry has neither 'intrinsic_isaac' "
          "nor 'intrinsic'"
      )
    K = np.asarray(intrinsic, dtype=np.float32).reshape(3, 3)
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
    path = self._path("depth", i, self.depth_suffix)
    if not path.exists():
      raise FileNotFoundError(f"Depth file not found: {path}")
    if self.depth_suffix == ".npy":
      depth = np.load(path, allow_pickle=False).astype(np.float32, copy=False)
    else:
      # Real captures save aligned uint16 depth in millimeters.
      depth = imageio.imread(path).astype(np.float32, copy=False) / 1000.0
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


def _resolve_texture_path(
    raw_path,
    mesh_dir,
    path_mappings,
    texture_roots,
    expected_texture_file=None,
    texture_diagnostics=None,
    reject_nonstandard_texture=False,
):
  # 항상 로컬(mesh_dir/texture_roots) 파일명 기준으로만 찾는다 — map_Kd의 원본
  # 절대경로는 후보로 쓰지 않는다.
  expanded = os.path.expandvars(os.path.expanduser(raw_path))
  candidates = []

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
      resolved = candidate.resolve()
      if (
          expected_texture_file is not None
          and Path(raw_path).name != Path(expected_texture_file).name
      ):
        diagnostic = {
            "issue": "nonstandard_map_kd",
            "map_kd": raw_path,
            "expected_texture": Path(expected_texture_file).name,
            "resolved_texture": str(resolved),
        }
        if texture_diagnostics is not None:
          texture_diagnostics.append(diagnostic)
        if reject_nonstandard_texture:
          raise NonstandardMapKdError(
              f"Nonstandard map_Kd texture name: {Path(raw_path).name!r}; "
              f"expected {Path(expected_texture_file).name!r}"
          )
      return resolved
  if texture_diagnostics is not None:
    texture_diagnostics.append({
        "issue": "missing_map_kd_texture",
        "map_kd": raw_path,
        "expected_texture": (
            Path(expected_texture_file).name
            if expected_texture_file is not None
            else None
        ),
        "resolved_texture": None,
    })
  rendered = "\n  ".join(str(candidate) for candidate in candidates)
  raise FileNotFoundError(f"Could not resolve texture {raw_path!r}; tried:\n  {rendered}")


def _used_material_names(mesh_file):
  """Collect material names an OBJ actually assigns to faces via usemtl.

  Exported MTLs can carry leftover materials from editing history (e.g. a
  Blender material pointing at a texture that was never re-exported next to
  the asset). Those are never loaded by trimesh, so they must not block
  loading the materials the mesh actually uses.
  """
  names = set()
  with Path(mesh_file).open("r", encoding="utf-8", errors="replace") as stream:
    for line in stream:
      stripped = line.strip()
      if not stripped.startswith("usemtl"):
        continue
      tokens = stripped.split(None, 1)
      if len(tokens) == 2:
        names.add(tokens[1].strip())
  return names


def _used_material_textures(mtl_file, used_material_names):
  """Map each used material name to its map_Kd basename (works on the
  original MTL or the runtime one — only the basename is compared)."""
  textures = {}
  current_material = None
  with mtl_file.open("r", encoding="utf-8", errors="replace") as stream:
    for line in stream:
      stripped = line.strip()
      tokens = stripped.split(None, 1)
      if len(tokens) != 2:
        continue
      if tokens[0].lower() == "newmtl":
        current_material = tokens[1]
      elif tokens[0].lower() == "map_kd" and current_material in used_material_names:
        textures[current_material] = Path(stripped.split()[-1]).name
  return textures


def _used_texture_names(mtl_file, used_material_names):
  """Collect the distinct map_Kd basenames that used_material_names
  reference — see _used_material_textures."""
  return set(_used_material_textures(mtl_file, used_material_names).values())


def known_object_names_for_year(object_catalog, year):
  """Collect every Old_name/Object_name registered for a catalog year —
  used to tell a real sibling asset (combined into one scan) apart from an
  incidental part that has no separate asset of its own."""
  names = set()
  for entry in object_catalog.values():
    if entry.get("year") != str(year):
      continue
    for key in ("old_name", "object_name"):
      value = entry.get(key, "")
      if value:
        names.add(value)
  return names


def _identity_material_or_raise(original_mtl, identity_name, known_names, used_material_names):
  """Pin the identity material by name match (see _texture_from_mtl). Any
  other used material is then a second texture in the same scan — if its
  name is another registered object (known_names), this is a combined scan
  of two separate assets, so raise rather than silently keep only one.
  Otherwise it's an incidental part with no asset of its own to lose —
  drop it. Returns (possibly narrowed used_material_names, identity
  material name or None if untouched)."""
  textures = _used_material_textures(original_mtl, used_material_names)
  if len(set(textures.values())) <= 1:
    return used_material_names, None

  matches = [
      material for material, texture in textures.items()
      if Path(texture).stem == identity_name
      or Path(texture).stem.removesuffix("_edited") == identity_name
  ]
  if len(matches) != 1:
    raise ValueError(
        f"Expected exactly one material whose texture matches "
        f"{identity_name!r}, found {len(matches)} in {original_mtl} "
        f"(textures: {sorted(set(textures.values()))})"
    )
  identity_material = matches[0]

  for material, texture in textures.items():
    if material == identity_material:
      continue
    other_name = Path(texture).stem.removesuffix("_edited")
    if other_name in known_names:
      raise ValueError(
          f"{original_mtl} combines {identity_name!r} with another "
          f"registered object {other_name!r} in one scan — can't load "
          "as a single part"
      )
  return {identity_material}, identity_material


def _runtime_mtl(
    original_mtl,
    runtime_dir,
    path_mappings,
    texture_roots,
    expected_texture_file=None,
    texture_diagnostics=None,
    reject_nonstandard_texture=False,
    used_material_names=None,
):
  output_lines = []
  current_material = None
  with original_mtl.open("r", encoding="utf-8") as stream:
    for line in stream:
      stripped = line.strip()
      if not stripped or stripped.startswith("#"):
        output_lines.append(line)
        continue
      tokens = shlex.split(stripped, comments=False, posix=True)
      if tokens and tokens[0].lower() == "newmtl" and len(tokens) >= 2:
        current_material = tokens[1]
        output_lines.append(line)
        continue
      if tokens and tokens[0].lower() == "map_kd" and len(tokens) >= 2:
        if (
            used_material_names is not None
            and current_material not in used_material_names
        ):
          # Unreferenced material: leave its map_Kd untouched instead of
          # resolving/validating a texture the mesh will never load.
          output_lines.append(line)
          continue
        # The current dataset has no map options. Taking the last token also
        # handles the common option-bearing form, e.g. "map_Kd -s 1 1 1 x.png".
        source = _resolve_texture_path(
            tokens[-1],
            original_mtl.parent,
            path_mappings,
            texture_roots,
            expected_texture_file=expected_texture_file,
            texture_diagnostics=texture_diagnostics,
            reject_nonstandard_texture=reject_nonstandard_texture,
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


def validate_mtl_textures(
    mesh_file,
    identity_name,
    known_names,
    path_mappings=None,
    texture_roots=None,
    expected_texture_file=None,
    texture_diagnostics=None,
    reject_nonstandard_texture=False,
):
  """Validate MTL texture references without loading mesh geometry."""
  mesh_file = Path(mesh_file).expanduser().resolve()
  if not mesh_file.is_file():
    raise FileNotFoundError(f"Mesh not found: {mesh_file}")
  original_mtl = mesh_file.with_suffix(".mtl")
  if not original_mtl.is_file():
    raise FileNotFoundError(f"MTL not found beside mesh: {original_mtl}")
  used_material_names, _ = _identity_material_or_raise(
      original_mtl, identity_name, known_names, _used_material_names(mesh_file),
  )
  with tempfile.TemporaryDirectory(prefix="foundationpose_asset_check_") as temp:
    _runtime_mtl(
        original_mtl,
        Path(temp),
        path_mappings=path_mappings or [],
        texture_roots=texture_roots or [],
        expected_texture_file=expected_texture_file,
        texture_diagnostics=texture_diagnostics,
        reject_nonstandard_texture=reject_nonstandard_texture,
        used_material_names=used_material_names,
    )


@contextmanager
def load_mesh_readonly(
    mesh_file,
    identity_name,
    known_names,
    mesh_scale=0.001,
    path_mappings=None,
    texture_roots=None,
    max_texture_size=4096,
    expected_texture_file=None,
    texture_diagnostics=None,
    reject_nonstandard_texture=False,
):
  """Load a textured OBJ without editing OBJ, MTL, or texture source files.
  See _identity_material_or_raise for multi-part handling."""
  mesh_file = Path(mesh_file).expanduser().resolve()
  if not mesh_file.is_file():
    raise FileNotFoundError(f"Mesh not found: {mesh_file}")
  original_mtl = mesh_file.with_suffix(".mtl")
  if not original_mtl.is_file():
    raise FileNotFoundError(f"MTL not found beside mesh: {original_mtl}")

  with tempfile.TemporaryDirectory(prefix="foundationpose_asset_") as temp:
    runtime_dir = Path(temp)
    (runtime_dir / mesh_file.name).symlink_to(mesh_file)
    used_material_names, identity_material = _identity_material_or_raise(
        original_mtl, identity_name, known_names, _used_material_names(mesh_file),
    )
    _runtime_mtl(
        original_mtl,
        runtime_dir,
        path_mappings=path_mappings or [],
        texture_roots=texture_roots or [],
        expected_texture_file=expected_texture_file,
        texture_diagnostics=texture_diagnostics,
        reject_nonstandard_texture=reject_nonstandard_texture,
        used_material_names=used_material_names,
    )
    # 최신 trimesh는 runtime_dir 밖을 가리키는 심링크(텍스처)를 거부하므로 허용 처리
    resolver = resolvers.FilePathResolver(str(runtime_dir), allow_anywhere=True)
    mesh = trimesh.load(
        runtime_dir / mesh_file.name, process=False, resolver=resolver
    )
    if isinstance(mesh, trimesh.Scene):
      if identity_material is not None:
        # 작은 보조 부품은 버리고 본체(identity) 지오메트리만 쓴다.
        mesh = mesh.geometry[identity_material]
      elif len(mesh.geometry) == 1:
        mesh = next(iter(mesh.geometry.values()))
      else:
        # 부품이 여러 개지만 전부 같은 텍스처를 공유하는 경우, 합쳐서 쓴다.
        mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
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
