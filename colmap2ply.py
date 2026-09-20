#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
colmap2ply.py -- Convert a COLMAP sparse reconstruction (points3D) into an
RGB-colored point cloud PLY, and optionally export camera frustums as a
wireframe OBJ (the colored pyramids you see in COLMAP GUI / papers).

The COLMAP point cloud already stores RGB, so this is "just" a format
conversion -- but MeshLab cannot open points3D.bin directly, and the camera
wireframes are only visible inside COLMAP itself.

Input can be:
    a reconstruction folder  (contains points3D.bin / points3D.txt [+ cameras/images])
    a sparse folder          (auto-descends into sparse/0, sparse/1, ...)
    or points3D.bin / points3D.txt directly

Usage
-----
python colmap2ply.py path/to/sparse/0
python colmap2ply.py path/to/sparse/0 --cameras          # also write <out>_cameras.obj
python colmap2ply.py path/to/sparse/0 -o mycloud.ply
python colmap2ply.py dataset --split                     # one PLY per sparse/<i>
"""
import argparse
import glob
import os
import struct
import sys

import numpy as np

from gs2ply import write_colored_ply

# COLMAP camera model ids -> number of parameters
CAMERA_MODEL_NUM_PARAMS = {
    0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12,
    7: 4, 8: 4, 9: 5, 10: 12, 11: 12,
}
CAMERA_MODEL_NAMES = {
    0: "SIMPLE_PINHOLE", 1: "PINHOLE", 2: "SIMPLE_RADIAL", 3: "RADIAL",
    4: "OPENCV", 5: "OPENCV_FISHEYE", 6: "FULL_OPENCV", 7: "FOV",
    8: "SIMPLE_RADIAL_FISHEYE", 9: "RADIAL_FISHEYE", 10: "THIN_PRISM_FISHEYE",
    11: "RAD_TAN_THIN_PRISM_FISHEYE",
}


# ----------------------------------------------------------------------------
# binary readers (format documented in COLMAP scripts/python/read_write_model.py)
# ----------------------------------------------------------------------------
def read_points3D_bin(path):
    with open(path, "rb") as f:
        buf = f.read()
    n = struct.unpack_from("<Q", buf, 0)[0]
    off = 8
    ids = np.empty(n, dtype=np.int64)
    xyz = np.empty((n, 3), dtype=np.float64)
    rgb = np.empty((n, 3), dtype=np.uint8)
    track = np.empty(n, dtype=np.int64)
    for i in range(n):
        pid, = struct.unpack_from("<Q", buf, off); off += 8
        x, y, z = struct.unpack_from("<3d", buf, off); off += 24
        r, g, b = struct.unpack_from("<3B", buf, off); off += 3
        off += 8                      # reprojection error
        tlen, = struct.unpack_from("<Q", buf, off); off += 8
        off += 8 * tlen               # track entries: (uint32 image_id, uint32 idx2d)
        ids[i] = pid
        xyz[i] = (x, y, z)
        rgb[i] = (r, g, b)
        track[i] = tlen
    return ids, xyz, rgb, track


def read_points3D_txt(path):
    # COLMAP txt lines are "ID X Y Z R G B ERR image_id idx2d image_id idx2d ..."
    pts = []
    with open(path, "r", encoding="latin-1") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            pts.append([float(parts[0]), float(parts[1]), float(parts[2]),
                        float(parts[3]), float(parts[4]), float(parts[5]),
                        float(parts[6]), len(parts)])  # col7 = #tokens incl. track
    arr = np.asarray(pts, dtype=np.float64).reshape(-1, 8)
    ids = arr[:, 0].astype(np.int64)
    xyz = arr[:, 1:4]
    rgb = np.rint(arr[:, 4:7]).astype(np.uint8)
    # token count == 7 + 2*track_len  ->  track_len = (tokens-7)//2
    track = ((arr[:, 7] - 7) // 2).astype(np.int64)
    return ids, xyz, rgb, track


def read_images_bin(path):
    with open(path, "rb") as f:
        buf = f.read()
    n = struct.unpack_from("<Q", buf, 0)[0]
    off = 8
    out = {}
    for _ in range(n):
        iid, = struct.unpack_from("<I", buf, off); off += 4
        q = np.array(struct.unpack_from("<4d", buf, off)); off += 32
        t = np.array(struct.unpack_from("<3d", buf, off)); off += 24
        cam_id, = struct.unpack_from("<I", buf, off); off += 4
        # COLMAP stores the image name as a NUL-terminated byte string.
        name_end = buf.find(b"\0", off)
        if name_end < 0:
            raise ValueError(f"invalid images.bin: unterminated image name at offset {off}")
        name = buf[off:name_end].decode("utf-8", "replace")
        off = name_end + 1
        np2d, = struct.unpack_from("<Q", buf, off); off += 8
        off += 24 * np2d               # (double x, double y, int64 point3d_id)
        out[iid] = (q, t, cam_id, name)
    return out


def read_images_txt(path):
    out = {}
    lines = [ln.strip() for ln in open(path, "r", encoding="latin-1")]
    lines = [ln for ln in lines if ln and not ln.startswith("#")]
    i = 0
    while i < len(lines):
        parts = lines[i].split()
        # images.txt puts the 2D points on a second line; that line is much longer
        if len(parts) >= 9:
            iid = int(parts[0])
            q = np.array([float(v) for v in parts[1:5]])
            t = np.array([float(v) for v in parts[5:8]])
            cam_id = int(parts[8])
            name = parts[9] if len(parts) > 9 else f"image{iid}"
            out[iid] = (q, t, cam_id, name)
        i += 2  # skip the 2D-points line
    return out


def read_cameras_bin(path):
    with open(path, "rb") as f:
        buf = f.read()
    n = struct.unpack_from("<Q", buf, 0)[0]
    off = 8
    out = {}
    for _ in range(n):
        cid, model = struct.unpack_from("<Ii", buf, off); off += 8
        w, h = struct.unpack_from("<QQ", buf, off); off += 16
        npar = CAMERA_MODEL_NUM_PARAMS[model]
        params = np.array(struct.unpack_from(f"<{npar}d", buf, off)); off += 8 * npar
        out[cid] = (CAMERA_MODEL_NAMES[model], w, h, params)
    return out


def read_cameras_txt(path):
    out = {}
    for line in open(path, "r", encoding="latin-1"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        cid = int(parts[0])
        model = parts[1]
        w, h = int(parts[2]), int(parts[3])
        params = np.array([float(v) for v in parts[4:]])
        out[cid] = (model, w, h, params)
    return out


# ----------------------------------------------------------------------------
# geometry helpers
# ----------------------------------------------------------------------------
def qvec2rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * z * x + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * z * y - 2 * x * w],
        [2 * z * x - 2 * y * w, 2 * z * y + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
    ])


PALETTE = np.array([
    [31, 119, 180], [255, 127, 14], [44, 160, 44], [214, 39, 40],
    [148, 103, 189], [140, 86, 75], [227, 119, 194], [127, 127, 127],
    [188, 189, 34], [23, 190, 207],
], dtype=np.uint8)


def write_camera_obj(path, images, cameras, depth):
    """Frustum wireframe: apex at camera center, base = image corners at `depth`."""
    verts = []       # list of (x, y, z, r, g, b)
    lines = []       # 1-based vertex index pairs
    for k, (iid, (q, t, cam_id, _name)) in enumerate(sorted(images.items())):
        R = qvec2rotmat(q)
        C = -R.T @ t
        color = PALETTE[k % len(PALETTE)]
        base_idx = len(verts) + 1
        verts.append((*C, *color))                    # apex
        if cam_id in cameras:
            _model, w, h, params = cameras[cam_id]
            f = params[0]
            cx = params[2] if len(params) >= 3 else (w - 1) / 2.0
            cy = params[3] if len(params) >= 4 else (h - 1) / 2.0
        else:                                          # intrinsics missing -> guess
            w, h, f, cx, cy = 1920, 1080, 1500.0, 960.0, 540.0
        corners = []
        for px, py in ((0, 0), (w, 0), (w, h), (0, h)):
            d = np.array([(px - cx) / f, (py - cy) / f, 1.0])
            d = d / np.linalg.norm(d) * depth
            corners.append(C + R.T @ d)
        for c in corners:
            verts.append((*c, *color))
        for j in range(4):
            lines.append((base_idx, base_idx + 1 + j))                 # apex -> base
            lines.append((base_idx + 1 + j, base_idx + 1 + (j + 1) % 4))  # base ring
    with open(path, "w", encoding="ascii") as f:
        f.write("# camera frustums exported by colmap2ply.py\n")
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {v[3]} {v[4]} {v[5]}\n")
        for a, b in lines:
            f.write(f"l {a} {b}\n")


# ----------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------
def locate_model(input_path):
    """Accept a dataset root, a 'sparse' folder, a sparse/NN folder or a points3D file."""
    p = input_path
    if os.path.isfile(p):
        return [p]
    if not os.path.isdir(p):
        raise SystemExit(f"input not found: {p}")
    for sub in ("sparse/0", "sparse", "sparse_0"):
        cand = os.path.join(p, sub)
        if os.path.isdir(cand):
            p = cand
            break
    hits = []
    for name in ("points3D.bin", "points3D.txt"):
        cand = os.path.join(p, name)
        if os.path.isfile(cand):
            hits.append(cand)
    if not hits:
        # maybe user passed the dataset root with several sparse folders
        many = sorted(glob.glob(os.path.join(p, "sparse", "*", "points3D.*")))
        if many:
            raise SystemExit(
                f"multiple models found under {p}: pass one explicitly, e.g. "
                + ", ".join(os.path.dirname(m) for m in many[:4]))
        raise SystemExit(f"no points3D.bin/points3D.txt under {p}")
    return [hits[0]]


def convert(points_path, args):
    print(f"[colmap2ply] {points_path}")
    if points_path.endswith(".bin"):
        ids, xyz, rgb, track = read_points3D_bin(points_path)
    else:
        ids, xyz, rgb, track = read_points3D_txt(points_path)
    n = xyz.shape[0]
    print(f"  points: {n:,}   mean RGB = "
          f"({rgb[:,0].mean():.0f},{rgb[:,1].mean():.0f},{rgb[:,2].mean():.0f})")

    keep = np.ones(n, dtype=bool)
    if args.min_track > 0:
        keep &= track >= args.min_track
        print(f"  --min-track {args.min_track}: kept {int(keep.sum()):,}")
    idx = np.flatnonzero(keep)
    if args.max_points and idx.size > args.max_points:
        rng = np.random.default_rng(args.seed)
        idx = np.sort(rng.choice(idx, size=args.max_points, replace=False))
        print(f"  subsampled -> {args.max_points:,}")

    x, y, z = xyz[idx, 0], xyz[idx, 1], xyz[idx, 2]
    col = rgb[idx]

    out_ply = args.output or os.path.splitext(points_path)[0] + "_rgb.ply"
    write_colored_ply(out_ply, x, y, z, col, args.ascii)
    print(f"  wrote {out_ply} ({x.shape[0]:,} points)")

    if args.cameras:
        folder = os.path.dirname(points_path)
        images = {}
        cameras = {}
        folder = os.path.dirname(points_path)
        for name in ("images", "cameras"):
            b = os.path.join(folder, f"{name}.bin")
            t = os.path.join(folder, f"{name}.txt")
            if name == "images":
                if os.path.isfile(b):
                    images = read_images_bin(b)
                elif os.path.isfile(t):
                    images = read_images_txt(t)
            else:
                if os.path.isfile(b):
                    cameras = read_cameras_bin(b)
                elif os.path.isfile(t):
                    cameras = read_cameras_txt(t)
        if not images:
            print("  [warn] no images.bin/txt next to points3D -- cannot draw cameras")
            return
        if not args.cam_scale:
            diag = float(np.linalg.norm(xyz.max(axis=0) - xyz.min(axis=0))) if n else 10.0
            depth = 0.08 * diag
        else:
            depth = args.cam_scale
        out_obj = os.path.splitext(out_ply)[0] + "_cameras.obj"
        write_camera_obj(out_obj, images, cameras, depth)
        print(f"  wrote {out_obj} ({len(images)} frustums, depth={depth:.2f})")


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Convert a COLMAP sparse model (points3D) into an RGB point-cloud "
                    "PLY, optionally exporting camera frustums as a wireframe OBJ.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("input", help="sparse/0 folder, dataset root, or points3D.bin/.txt")
    p.add_argument("-o", "--output", help="output PLY (default: <points3D>_rgb.ply)")
    p.add_argument("--cameras", action="store_true",
                   help="also export camera frustums as <out>_cameras.obj "
                        "(needs images.bin/txt + cameras.bin/txt next to points3D)")
    p.add_argument("--cam-scale", type=float, default=None,
                   help="frustum depth in world units (default: 8%% of bbox diagonal)")
    p.add_argument("--min-track", type=int, default=0,
                   help="drop points seen in fewer than this many images (try 2-3)")
    p.add_argument("--max-points", type=int, default=None,
                   help="random-subsample to at most this many points")
    p.add_argument("--ascii", action="store_true", help="write ASCII PLY")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    for points_path in locate_model(args.input):
        convert(points_path, args)


if __name__ == "__main__":
    main()
