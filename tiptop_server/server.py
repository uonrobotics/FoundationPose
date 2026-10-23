"""HTTP pose server for TiPToP, run with the FoundationPose conda Python.

`--check` only reports missing prerequisites. Mesh files are read only; model/debug scratch stays under /tmp.
"""

import argparse
import importlib.machinery
import importlib.util
import io
import json
import socket
import sys
import time
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PORT = 13346
ENV_HINT = "see readme_custom.md, 'Installation'"
# Run directories loaded by the upstream predict_score.py and predict_pose_refine.py
WEIGHT_RUNS = ("2024-01-11-20-02-45", "2023-10-28-18-33-37")
# Imported by this script itself, and by the FoundationPose sources
SERVER_MODULES = ("numpy", "torch", "nvdiffrast")
SOURCE_MODULES = ("pytorch3d", "trimesh", "cv2", "scipy", "omegaconf", "open3d", "imageio", "transformations")

Finding = tuple[str, str]  # (level, message) with level "ok", "warn" or "error"


def port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # The same option HTTPServer binds with
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def preflight(
    asset_root: Path, port: int, port_busy_level: str, extra_findings: Callable[[Path], list[Finding]] | None = None
) -> list[Finding]:
    """Report what is missing before serving. Level "error" means serving cannot work; imports no GPU library."""
    findings = [("ok", f"FoundationPose source: {ROOT}")]
    missing_weights = [
        ROOT / "weights" / run / name
        for run in WEIGHT_RUNS
        for name in ("model_best.pth", "config.yml")
        if not (ROOT / "weights" / run / name).is_file()
    ]
    findings += [
        (
            "error",
            f"FoundationPose weights missing: {path}; download the checkpoints (see readme.md, 'Data prepare')",
        )
        for path in missing_weights
    ] or [("ok", f"FoundationPose weights: {', '.join(WEIGHT_RUNS)}")]
    build = ROOT / "mycpp" / "build"
    if any((build / f"mycpp{suffix}").is_file() for suffix in importlib.machinery.EXTENSION_SUFFIXES):
        findings.append(("ok", f"mycpp extension built for Python {sys.version.split()[0]}"))
    elif importlib.util.find_spec("mycpp") is None:
        findings.append(
            (
                "error",
                f"mycpp extension not built for Python {sys.version.split()[0]} in {build}; "
                f"run `bash build_all_conda.sh` in {ROOT} with the foundationpose env active ({ENV_HINT})",
            )
        )
    for name in SERVER_MODULES:
        if importlib.util.find_spec(name) is None:
            findings.append(("error", f"Python module '{name}' not found for {sys.executable}; {ENV_HINT}"))
    missing = [name for name in SOURCE_MODULES if importlib.util.find_spec(name) is None]
    if missing:
        findings.append(("warn", f"Python modules not found: {', '.join(missing)}; {ENV_HINT}"))
    elif all(importlib.util.find_spec(name) for name in SERVER_MODULES):
        findings.append(("ok", f"Python modules present for {sys.executable}"))
    if not asset_root.is_dir():
        findings.append(("warn", f"Asset root not found: {asset_root}; every mesh request would be rejected"))
    elif extra_findings:
        findings += extra_findings(asset_root)
    if not port_is_free(port):
        findings.append(
            (
                port_busy_level,
                f"Port {port} is in use; a server may already be running "
                f"(curl http://127.0.0.1:{port}/health) or pass --port",
            )
        )
    return findings


def serve(asset_root: Path, port: int):
    sys.path.insert(0, str(ROOT))
    import numpy as np
    import nvdiffrast.torch as dr
    from custom_datareader import load_mesh_readonly
    from estimater import FoundationPose, PoseRefinePredictor, ScorePredictor

    scorer, refiner = ScorePredictor(), PoseRefinePredictor()
    context = dr.RasterizeCudaContext()
    estimators = {}
    asset_root = asset_root.resolve()

    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, result):
            body = json.dumps(result, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self.reply(200 if self.path == "/health" else 404, dict(status="healthy", model_loaded=True))

        def do_POST(self):
            if self.path != "/estimate":
                return self.reply(404, dict(error="Unknown endpoint"))
            try:
                started = time.perf_counter()
                size = int(self.headers.get("Content-Length", 0))
                if not 0 < size < 32 * 1024 * 1024:
                    raise ValueError("Invalid pose packet size")
                packet = np.load(io.BytesIO(self.rfile.read(size)), allow_pickle=False)
                path = Path(str(packet["mesh_path"])).resolve()
                if not path.is_relative_to(asset_root) or path.suffix != ".obj":
                    raise ValueError("Mesh must be an OBJ in the configured asset root")
                class_id, scale = str(packet["class_id"]), float(packet["mesh_scale"])
                if not np.isfinite(scale) or scale <= 0:
                    raise ValueError("Invalid mesh scale")
                key = (str(path), scale, class_id)
                if key not in estimators:
                    with load_mesh_readonly(
                        path,
                        identity_name=class_id,
                        known_names=[class_id],
                        mesh_scale=scale,
                        texture_roots=[path.parent, path.parent.parent],
                        max_texture_size=2048,
                    ) as mesh:
                        estimators[key] = FoundationPose(
                            model_pts=mesh.vertices,
                            model_normals=mesh.vertex_normals,
                            mesh=mesh,
                            scorer=scorer,
                            refiner=refiner,
                            glctx=context,
                            debug=0,
                            debug_dir="/tmp/tiptop-foundationpose-runtime",
                        )
                estimator = estimators[key]
                # The existing data_gen demo normalizes this scalar too: a numpy
                # float64 makes the crop-offset tensor double beside float poses.
                estimator.diameter = float(estimator.diameter)
                rgb, depth, mask, k = (packet[name] for name in ["rgb", "depth", "mask", "intrinsics"])
                if (
                    rgb.dtype != np.uint8
                    or rgb.shape != (*depth.shape, 3)
                    or mask.shape != depth.shape
                    or k.shape != (3, 3)
                    or not np.isfinite(k).all()
                    or min(k[0, 0], k[1, 1]) <= 0
                    or np.count_nonzero(mask & np.isfinite(depth) & (depth > 0)) < 10
                ):
                    raise ValueError("Invalid aligned RGB/depth/mask/intrinsics")
                pose = estimator.register(
                    K=k.astype(np.float32), rgb=rgb, depth=depth.astype(np.float32), ob_mask=mask, iteration=5
                )
                self.reply(
                    200,
                    dict(
                        camera_from_cad=pose.tolist(),
                        class_id=class_id,
                        elapsed_s=time.perf_counter() - started,
                        ranking_score=float(estimator.scores[0]),
                        score_is_probability=False,
                    ),
                )
            except Exception as error:
                self.reply(400, dict(error_type=type(error).__name__, error=str(error)))

    server = HTTPServer(("127.0.0.1", port), Handler)
    print(f"FOUNDATIONPOSE_READY port={port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main(default_asset_root: Path | None = None, extra_findings: Callable[[Path], list[Finding]] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=default_asset_root,
        required=default_asset_root is None,
        help="Only OBJ meshes under this directory are served",
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--check", action="store_true", help="Report missing prerequisites and exit; no GPU use")
    args = parser.parse_args()
    findings = preflight(args.asset_root, args.port, "warn" if args.check else "error", extra_findings)
    for level, message in findings:
        if level != "ok" or args.check:
            print(f"[{level.upper()}] {message}", flush=True)
    failed = any(level == "error" for level, _ in findings)
    if args.check:
        print(f"FoundationPose check: {'FAILED' if failed else 'OK'}", flush=True)
    if args.check or failed:
        sys.exit(1 if failed else 0)
    serve(args.asset_root, args.port)


if __name__ == "__main__":
    main()
