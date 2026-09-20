#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gs2ply.py -- Convert a 3D Gaussian Splatting model into a plain RGB point cloud PLY
that MeshLab / CloudCompare / Open3D can display directly.

Why this is needed
------------------
A trained 3DGS model (.ply) does NOT store RGB colors. Each Gaussian stores:
    x y z                    center position
    nx ny nz                 (unused normals, all zero)
    f_dc_0..2                DC spherical-harmonics coefficient  (base color)
    f_rest_0..44             higher-order SH coefficients        (view-dependent color)
    opacity                  stored as logit, real opacity = sigmoid(opacity)
    scale_0..2               stored as log,    real scale   = exp(scale)
    rot_0..3                 rotation quaternion

Color conversion follows the official gaussian-splatting code exactly:
    rgb = clamp( eval_sh(dir) + 0.5 , 0, 1 )
where eval_sh with degree 0 reduces to  rgb = clamp(0.5 + 0.2820948 * f_dc, 0, 1).

Usage examples
--------------
# 1) fastest: view-independent base color (what most "point cloud" previews show)
python gs2ply.py trained_model.ply

# 2) evaluate full SH from a default viewpoint (camera at origin by default)
python gs2ply.py trained_model.ply --color sh
python gs2ply.py trained_model.ply --color sh --camera-pos 4,0.5,-3

# 3) drop transparent floaters and huge sky gaussians, cap at 3M points
python gs2ply.py trained_model.ply --min-opacity 0.1 --max-scale 1.5 --max-points 3000000

# 4) batch / other formats
python gs2ply.py model.splat -o out.ply
python gs2ply.py a.ply b.ply c.ply
"""
import argparse
import os
import sys
import time

import numpy as np

# ----------------------------------------------------------------------------
# Spherical harmonics constants -- identical to gaussian-splatting utils/sh_utils.py
# ----------------------------------------------------------------------------
SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5900435899266435,
]
SH_C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -1.445305721320277,
]

_PLY_TYPE_MAP = {
    "char": "i1", "int8": "i1",
    "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2",
    "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4",
    "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4",
    "double": "f8", "float64": "f8",
}


# ----------------------------------------------------------------------------
# PLY reading
# ----------------------------------------------------------------------------
def _read_ply_header(path):
    """Parse a PLY header. Returns (fmt, elements, data_offset_bytes)."""
    with open(path, "rb") as f:
        lines = []
        offset = 0
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: unexpected EOF inside header")
            offset += len(line)
            text = line.decode("latin-1").strip()
            lines.append(text)
            if text == "end_header":
                break
        if not lines or lines[0] != "ply":
            raise ValueError(f"{path}: not a PLY file")

    fmt = None
    elements = []  # list of (name, count, [(prop_name, np_dtype), ...])
    cur = None
    for text in lines[1:]:
        parts = text.split()
        if not parts:
            continue
        key = parts[0]
        if key == "format":
            fmt = parts[1]
        elif key == "element":
            cur = {"name": parts[1], "count": int(parts[2]), "props": []}
            elements.append(cur)
        elif key == "property":
            if cur is None:
                raise ValueError(f"{path}: property before element in header")
            if parts[1] == "list":
                # e.g. property list uchar int vertex_indices (mesh faces)
                cur["props"].append(("__list__", parts[2], parts[3]))
            else:
                cur["props"].append((parts[2], _PLY_TYPE_MAP[parts[1]], None))
    return fmt, elements, offset


def read_ply_vertices(path, element_name="vertex"):
    """Read one element of a PLY file into a dict of numpy arrays.

    Handles binary_little_endian / binary_big_endian / ascii.
    Properties of other elements are skipped correctly (fixed-size elements only).
    """
    fmt, elements, offset = _read_ply_header(path)
    if fmt not in ("binary_little_endian", "binary_big_endian", "ascii"):
        raise ValueError(f"{path}: unsupported format '{fmt}'")

    target = next((e for e in elements if e["name"] == element_name), None)
    if target is None:
        raise ValueError(f"{path}: no element named '{element_name}'")

    prefix_bytes = 0
    for e in elements:
        if e is target:
            break
        if any(kind == "__list__" for _, kind, _ in e["props"]):
            raise ValueError(
                f"{path}: '{e['name']}' uses list properties; it must come after "
                f"'{element_name}' for this tool to read it")
        item_size = sum(np.dtype(t).itemsize for _, t, _ in e["props"])
        prefix_bytes += e["count"] * item_size

    dtype = np.dtype([(name, t) for name, t, _ in target["props"]])
    if fmt.startswith("binary"):
        endian = "<" if fmt == "binary_little_endian" else ">"
        dtype = dtype.newbyteorder(endian)
        with open(path, "rb") as f:
            f.seek(offset + prefix_bytes)
            raw = f.read(target["count"] * dtype.itemsize)
        if len(raw) < target["count"] * dtype.itemsize:
            raise ValueError(f"{path}: file truncated")
        arr = np.frombuffer(raw, dtype=dtype, count=target["count"])
        return {name: arr[name].copy() for name in dtype.names}

    # ascii path: read every token of the vertex element sequentially
    with open(path, "rb") as f:
        f.seek(offset + prefix_bytes)
        tokens = f.read().split()
    nprops = len(dtype.names)
    count = target["count"]
    need = count * nprops
    if len(tokens) < need:
        raise ValueError(f"{path}: ascii vertex data truncated")
    cols = np.array(tokens[:need], dtype=np.float64).reshape(count, nprops)
    out = {}
    for i, name in enumerate(dtype.names):
        out[name] = cols[:, i].astype(dtype[name].base.type)
    return out


def read_splat(path):
    """Read antimatter15 `.splat` format (32 bytes per gaussian) -> same dict shape."""
    raw = np.fromfile(path, dtype=np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("sx", "<f4"), ("sy", "<f4"), ("sz", "<f4"),
        ("r", "u1"), ("g", "u1"), ("b", "u1"), ("a", "u1"),
        ("rot0", "u1"), ("rot1", "u1"), ("rot2", "u1"), ("rot3", "u1"),
    ]))
    return {
        "x": raw["x"], "y": raw["y"], "z": raw["z"],
        "red": raw["r"].astype(np.float64),
        "green": raw["g"].astype(np.float64),
        "blue": raw["b"].astype(np.float64),
        "opacity": raw["a"].astype(np.float64) / 255.0,   # already activated
        "scale_0": raw["sx"], "scale_1": raw["sy"], "scale_2": raw["sz"],  # linear
    }


def _find_rgb_names(names):
    low = {n.lower(): n for n in names}
    out = []
    for ch in ("red", "green", "blue"):
        nm = low.get(ch) or next((k for k in low if ch in k), None)
        if nm is None:
            return None
        out.append(nm)
    return out


def load_gaussians(path):
    """Load any supported input into a dict of raw (unactivated) fields."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".splat":
        return read_splat(path), "splat"
    fields = read_ply_vertices(path)
    names = set(fields)
    required = {"x", "y", "z"}
    if not required.issubset(names):
        raise ValueError(f"{path}: missing position properties {required - names}")
    if {"f_dc_0", "f_dc_1", "f_dc_2"}.issubset(names):
        kind = "3dgs"
    elif _find_rgb_names(fields) is not None:
        kind = "rgbfply"  # already a colored point cloud -> passthrough
    else:
        raise ValueError(
            f"{path}: no f_dc_* (3DGS) and no red/green/blue found; "
            "not a 3D Gaussian PLY nor a colored point cloud")
    return fields, kind


# ----------------------------------------------------------------------------
# SH -> RGB
# ----------------------------------------------------------------------------
def eval_sh(degree, sh, dirs):
    """Evaluate SH exactly like gaussian-splatting (utils/sh_utils.py / forward.cu).

    sh:   (N, K, 3)  K=(degree+1)^2, last dim = RGB
    dirs: (N, 3)     normalized direction from camera to point
    returns (N, 3), values centered around 0 (add 0.5 afterwards).
    """
    sh3 = sh  # (N, K, 3), last axis = RGB
    result = SH_C0 * sh3[:, 0, :]
    x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]
    if degree > 0:
        result = (result
                  - SH_C1 * y[:, None] * sh3[:, 1, :]
                  + SH_C1 * z[:, None] * sh3[:, 2, :]
                  - SH_C1 * x[:, None] * sh3[:, 3, :])
    if degree > 1:
        xx, yy, zz = x * x, y * y, z * z
        xy, yz, xz = x * y, y * z, x * z
        result = (result
                  + SH_C2[0] * xy[:, None] * sh3[:, 4, :]
                  + SH_C2[1] * yz[:, None] * sh3[:, 5, :]
                  + SH_C2[2] * (2.0 * zz - xx - yy)[:, None] * sh3[:, 6, :]
                  + SH_C2[3] * xz[:, None] * sh3[:, 7, :]
                  + SH_C2[4] * (2.0 * xx - yy - zz)[:, None] * sh3[:, 8, :])
    if degree > 2:
        result = (result
                  + SH_C3[0] * yz[:, None] * (3 * xx - yy)[:, None] * sh3[:, 9, :]
                  + SH_C3[1] * xy[:, None] * z[:, None] * sh3[:, 10, :]
                  + SH_C3[2] * xy[:, None] * (4 * zz - xx - yy)[:, None] * sh3[:, 11, :]
                  + SH_C3[3] * z[:, None] * (4 * zz - xx - yy)[:, None] * sh3[:, 12, :]
                  + SH_C3[4] * xz[:, None] * (4 * zz - xx - yy)[:, None] * sh3[:, 13, :]
                  + SH_C3[5] * x[:, None] * (4 * xx - yy - zz)[:, None] * sh3[:, 14, :]
                  + SH_C3[6] * x[:, None] * (xx - 3 * yy)[:, None] * sh3[:, 15, :])
    return result


def detect_sh_degree(n_rest):
    """Number of f_rest properties -> SH degree (3 -> deg 3 => 45, 2 => 24, ...)."""
    for deg in (3, 2, 1):
        if n_rest >= 3 * ((deg + 1) ** 2 - 1):
            return deg
    return 0


def sh_coefficients(fields):
    """Pack f_dc / f_rest into (N, K, 3) with K=(deg+1)^2, RGB on the last axis."""
    f_dc = np.stack([fields["f_dc_0"], fields["f_dc_1"], fields["f_dc_2"]], axis=1)  # (N,3)
    rest_names = sorted(
        (n for n in fields if n.startswith("f_rest_")),
        key=lambda n: int(n.split("_")[-1]))
    n_rest = len(rest_names)
    deg = detect_sh_degree(n_rest)
    k = (deg + 1) ** 2          # total bands incl. DC; f_rest holds k-1 per channel
    n = f_dc.shape[0]
    sh = np.zeros((n, k, 3), dtype=np.float64)
    sh[:, 0, :] = f_dc
    if n_rest:
        # official trainer saves f_rest channel-major: [R(k-1) G(k-1) B(k-1)]
        k_rest = k - 1
        rest = np.stack([fields[nm] for nm in rest_names[:3 * k_rest]], axis=1)
        rest = rest.reshape(n, 3, k_rest).transpose(0, 2, 1)   # (N, k_rest, 3)
        sh[:, 1:, :] = rest
    return sh, deg


def compute_rgb(fields, kind, color_mode, camera_pos, view_dir, sh_degree=None):
    """Return rgb in [0,1], shape (N,3)."""
    if kind == "rgbfply":
        cn = _find_rgb_names(fields)
        return np.stack([fields[cn[0]], fields[cn[1]], fields[cn[2]]], axis=1)

    if kind == "splat":  # .splat already stores 0..255 colors
        return np.stack([fields["red"], fields["green"], fields["blue"]], axis=1) / 255.0

    sh, stored_deg = sh_coefficients(fields)
    deg = stored_deg if sh_degree is None else min(sh_degree, stored_deg)

    if deg == 0 or color_mode == "dc":
        rgb = 0.5 + SH_C0 * sh[:, 0, :]
    else:
        pts = np.stack([fields["x"], fields["y"], fields["z"]], axis=1)
        if view_dir is not None:
            d = np.asarray(view_dir, dtype=np.float64)
            dirs = np.broadcast_to(d / max(np.linalg.norm(d), 1e-12), pts.shape)
        else:
            dirs = pts - np.asarray(camera_pos, dtype=np.float64)[None, :]
            norms = np.linalg.norm(dirs, axis=1, keepdims=True)
            norms[norms < 1e-12] = 1.0
            dirs = dirs / norms
        rgb = eval_sh(deg, sh, dirs) + 0.5
    return np.clip(rgb, 0.0, 1.0)


# ----------------------------------------------------------------------------
# Filters
# ----------------------------------------------------------------------------
def apply_filters(fields, kind, rgb, min_opacity, max_scale, max_points, seed):
    n = rgb.shape[0]
    keep = np.ones(n, dtype=bool)

    if min_opacity > 0.0 and "opacity" in fields:
        op = fields["opacity"].astype(np.float64)
        if kind == "3dgs":
            op = 1.0 / (1.0 + np.exp(-op))          # sigmoid (stored as logit)
        keep &= op >= min_opacity

    if max_scale is not None:
        scales = [fields[k] for k in ("scale_0", "scale_1", "scale_2") if k in fields]
        if scales:
            s = np.max(np.stack(scales, axis=1), axis=1).astype(np.float64)
            if kind == "3dgs":
                s = np.exp(s)                        # stored as log
            keep &= s <= max_scale
        else:
            print("  [warn] --max-scale ignored: input has no scale_* properties")

    n_kept = int(keep.sum())
    if n_kept == 0:
        raise SystemExit("Filters removed every Gaussian; relax --min-opacity / --max-scale.")

    idx = np.flatnonzero(keep)
    if max_points and n_kept > max_points:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(idx, size=max_points, replace=False))
        print(f"  subsampled {n_kept:,} -> {max_points:,} points (seed {seed})")

    out = {"x": fields["x"][idx], "y": fields["y"][idx], "z": fields["z"][idx]}
    out["rgb"] = np.clip(np.rint(rgb[idx] * 255.0), 0, 255).astype(np.uint8)
    return out


# ----------------------------------------------------------------------------
# PLY writing
# ----------------------------------------------------------------------------
def write_colored_ply(path, x, y, z, rgb_u8, ascii_out=False):
    n = x.shape[0]
    header = (
        "ply\n"
        f"format {'ascii' if ascii_out else 'binary_little_endian'} 1.0\n"
        "comment Created by gs2ply (3DGS -> RGB point cloud)\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n")
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        if ascii_out:
            data = np.column_stack([x, y, z, rgb_u8])
            np.savetxt(f, data, fmt="%.6f %.6f %.6f %d %d %d")
        else:
            rec = np.empty(n, dtype=np.dtype([
                ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                ("r", "u1"), ("g", "u1"), ("b", "u1")]))
            rec["x"], rec["y"], rec["z"] = x, y, z
            rec["r"], rec["g"], rec["b"] = rgb_u8[:, 0], rgb_u8[:, 1], rgb_u8[:, 2]
            f.write(rec.tobytes())


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def parse_vec3(text):
    vals = [float(v) for v in text.replace(";", ",").split(",")]
    if len(vals) != 3:
        raise argparse.ArgumentTypeError("expected three numbers like 4,0.5,-3")
    return vals


def convert_one(in_path, out_path, args):
    t0 = time.time()
    print(f"[gs2ply] {in_path}")
    fields, kind = load_gaussians(in_path)
    n = fields["x"].shape[0]
    print(f"  loaded {n:,} primitives ({kind}) in {time.time() - t0:.1f}s")

    rgb = compute_rgb(fields, kind, args.color, args.camera_pos, args.view_dir, args.sh_degree)
    data = apply_filters(fields, kind, rgb, args.min_opacity, args.max_scale,
                         args.max_points, args.seed)

    if out_path is None:
        out_path = os.path.splitext(in_path)[0] + "_rgb.ply"
    write_colored_ply(out_path, data["x"], data["y"], data["z"], data["rgb"], args.ascii)

    mean = data["rgb"].mean(axis=0) / 255.0
    ext = np.stack([data["x"], data["y"], data["z"]], axis=1)
    bbox = ext.max(axis=0) - ext.min(axis=0)
    print(f"  wrote {out_path}  ({data['x'].shape[0]:,} points, "
          f"{os.path.getsize(out_path) / 1e6:.1f} MB, {time.time() - t0:.1f}s)")
    print(f"  mean RGB = ({mean[0]:.2f},{mean[1]:.2f},{mean[2]:.2f}), "
          f"bbox = ({bbox[0]:.2f}, {bbox[1]:.2f}, {bbox[2]:.2f})")


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Convert a 3D Gaussian Splatting PLY / .splat into an RGB-colored "
                    "point cloud PLY viewable in MeshLab / CloudCompare / Open3D.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("inputs", nargs="+", help="input .ply (3DGS) or .splat file(s)")
    p.add_argument("-o", "--output", help="output PLY (single input only; default: *_rgb.ply)")
    p.add_argument("--color", choices=["dc", "sh"], default="dc",
                   help="dc = view-independent base color from f_dc (fast, most robust); "
                        "sh = evaluate full SH for one default viewpoint")
    p.add_argument("--camera-pos", type=parse_vec3, default=(0.0, 0.0, 0.0),
                   help="with --color sh: viewpoint position; direction = point - camera")
    p.add_argument("--view-dir", type=parse_vec3, default=None,
                   help="with --color sh: use a constant view direction (overrides --camera-pos)")
    p.add_argument("--sh-degree", type=int, default=None, choices=[0, 1, 2, 3],
                   help="cap SH degree (default: use everything stored in the file)")
    p.add_argument("--min-opacity", type=float, default=0.0,
                   help="drop gaussians with sigmoid(opacity) below this (try 0.05-0.2)")
    p.add_argument("--max-scale", type=float, default=None,
                   help="drop gaussians whose largest exp(scale) exceeds this (removes sky/floaters)")
    p.add_argument("--max-points", type=int, default=None,
                   help="random-subsample to at most this many points (MeshLab likes <=2-3M)")
    p.add_argument("--ascii", action="store_true", help="write ASCII instead of binary PLY")
    p.add_argument("--seed", type=int, default=0, help="random seed for --max-points")
    args = p.parse_args(argv)

    if args.output and len(args.inputs) > 1:
        p.error("-o/--output only works with a single input file")
    for in_path in args.inputs:
        convert_one(in_path, args.output, args)


if __name__ == "__main__":
    main()
