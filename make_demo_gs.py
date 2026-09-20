#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_demo_gs.py -- Generate a synthetic 3D Gaussian Splatting PLY so you can
test the whole gs2ply pipeline without a trained model.

The scene contains a colorful ground plane, a "photo wall", confetti, and a
glossy sphere whose first-order SH coefficients are deliberately strong, so
`--color dc` and `--color sh` give visibly different results (that is the
whole point of the SH step).

    python make_demo_gs.py                    # -> demo/demo_3dgs.ply (~150k gaussians)
    python make_demo_gs.py --n 600000 -o big.ply
"""
import argparse
import os

import numpy as np

C0 = 0.28209479177387814


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def log_scale(s):
    return np.log(np.clip(s, 1e-8, None))


def write_3dgs_ply(path, xyz, rgb01, sh_rest, opacity01, scale, quat):
    """Write a standard gaussian-splatting PLY (degree 3: 45 f_rest values)."""
    n = xyz.shape[0]
    props = (["x", "y", "z", "nx", "ny", "nz"]
             + [f"f_dc_{i}" for i in range(3)]
             + [f"f_rest_{i}" for i in range(45)]
             + ["opacity", "scale_0", "scale_1", "scale_2",
                "rot_0", "rot_1", "rot_2", "rot_3"])
    header = ["ply", "format binary_little_endian 1.0",
              "comment synthetic 3DGS created by make_demo_gs.py",
              f"element vertex {n}"]
    header += [f"property float {p}" for p in props]
    header.append("end_header")

    f_dc = (rgb01 - 0.5) / C0                       # invert the DC color equation
    dtype = np.dtype([(p, "<f4") for p in props])
    rec = np.zeros(n, dtype=dtype)
    rec["x"], rec["y"], rec["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    rec["f_dc_0"], rec["f_dc_1"], rec["f_dc_2"] = f_dc[:, 0], f_dc[:, 1], f_dc[:, 2]
    for i in range(45):
        rec[f"f_rest_{i}"] = sh_rest[:, i]
    rec["opacity"] = logit(opacity01)
    for i in range(3):
        rec[f"scale_{i}"] = log_scale(scale[:, i])
    for i in range(4):
        rec[f"rot_{i}"] = quat[:, i]

    with open(path, "wb") as f:
        f.write(("\n".join(header) + "\n").encode("ascii"))
        f.write(rec.tobytes())
    print(f"wrote {path}  ({n:,} gaussians, {os.path.getsize(path)/1e6:.1f} MB)")


def build_scene(n_total, rng):
    # ---- 1) ground plane: green->yellow gradient  (60%) --------------------
    n_ground = int(n_total * 0.60)
    gx = rng.uniform(-4, 4, n_ground)
    gz = rng.uniform(-4, 4, n_ground)
    gy = rng.uniform(-0.02, 0.02, n_ground)
    t = (gx + gz + 8) / 16.0
    ground_rgb = np.stack([0.25 + 0.55 * t, 0.65 - 0.10 * t, 0.20 + 0.15 * t], axis=1)
    ground = dict(xyz=np.stack([gx, gy, gz], axis=1), rgb=ground_rgb,
                  scale=np.full((n_ground, 3), 0.035), color_bias=0.0)

    # ---- 2) back wall: colorful vertical stripes  (25%) --------------------
    n_wall = int(n_total * 0.25)
    wx = rng.uniform(-4, 4, n_wall)
    wy = rng.uniform(0, 2.6, n_wall)
    wz = rng.uniform(3.9, 4.0, n_wall)
    stripe = np.floor((wx + 4) / 8 * 10).astype(int) % 5
    wall_palette = np.array([
        [0.85, 0.30, 0.25], [0.95, 0.75, 0.30], [0.30, 0.65, 0.90],
        [0.90, 0.90, 0.92], [0.55, 0.35, 0.75]])
    wall_rgb = wall_palette[stripe] * rng.uniform(0.85, 1.0, (n_wall, 1))
    wall = dict(xyz=np.stack([wx, wy, wz], axis=1), rgb=wall_rgb,
                scale=np.full((n_wall, 3), 0.04), color_bias=0.0)

    # ---- 3) glossy sphere: strong view-dependent SH  (10%) -----------------
    n_sph = int(n_total * 0.10)
    # sample on a sphere surface (r=0.8, centered at (0,1,0))
    v = rng.normal(size=(n_sph, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    sph_xyz = v * 0.8 + np.array([[0.0, 1.0, 0.0]])
    base = np.array([0.75, 0.18, 0.15])[None, :]           # red base
    sph_rgb = np.repeat(np.clip(base + rng.normal(0, 0.03, (n_sph, 3)), 0, 1), 1, axis=0)
    # view-dependent part: want a highlight when view dir ~ +z (default camera
    # placed at origin looking along +z). 3DGS band-1: color -= C1*y*sh[1];
    # += C1*z*sh[2]; -= C1*x*sh[3]  ->  put energy into sh[2] (=z term).
    rest = np.zeros((n_sph, 45))
    C1 = 0.4886025119029199
    bright = np.clip(0.9 * v[:, 2] ** 3, 0, 1)             # front of the sphere glows
    rest[:, 0 + 2] = bright * 2.0 / C1                     # sh[2] multiplies +C1*z  (R channel)
    rest[:, 15 + 2] = bright * 1.0 / C1                    # ... G channel
    rest[:, 30 + 2] = bright * 0.6 / C1                    # ... B channel
    sph = dict(xyz=sph_xyz, rgb=sph_rgb,
               scale=np.full((n_sph, 3), 0.03), color_bias=0.0)
    sph["sh_rest"] = rest

    # ---- 4) confetti  (5%) --------------------------------------------------
    n_conf = n_total - n_ground - n_wall - n_sph
    cx = rng.uniform(-3.5, 3.5, n_conf)
    cy = rng.uniform(0.2, 2.2, n_conf)
    cz = rng.uniform(-3.5, 3.5, n_conf)
    conf_rgb = rng.uniform(0.05, 1.0, (n_conf, 3))
    conf = dict(xyz=np.stack([cx, cy, cz], axis=1), rgb=conf_rgb,
                scale=rng.uniform(0.01, 0.03, (n_conf, 3)), color_bias=0.0)
    conf["sh_rest"] = np.zeros((n_conf, 45))

    parts = [ground, wall, sph, conf]
    xyz = np.concatenate([p["xyz"] for p in parts])
    rgb = np.clip(np.concatenate([p["rgb"] for p in parts]), 0, 1)
    rest = np.concatenate([p.get("sh_rest", np.zeros((len(p["xyz"]), 45))) for p in parts])
    scale = np.concatenate([p["scale"] for p in parts])
    n = xyz.shape[0]

    opacity = np.full(n, 0.95)
    # a few low-opacity "floaters" so opacity filtering has something to do
    n_floater = max(1, n // 50)
    flo = rng.integers(0, n, n_floater)
    opacity[flo] = rng.uniform(0.01, 0.08, n_floater)
    scale[flo] = rng.uniform(0.3, 0.8, (n_floater, 3))     # also oversized -> max-scale test
    xyz[flo] += rng.uniform(-1, 1, (n_floater, 3)) + np.array([[0.0, 3.5, 0.0]])

    # random unit quaternions
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return xyz, rgb, rest, opacity, scale, q


def main():
    p = argparse.ArgumentParser(description="Create a synthetic 3DGS PLY for testing")
    p.add_argument("-o", "--output", default="demo/demo_3dgs.ply")
    p.add_argument("--n", type=int, default=150000, help="number of gaussians")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    xyz, rgb, rest, opacity, scale, quat = build_scene(args.n, rng)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    write_3dgs_ply(args.output, xyz, rgb, rest, opacity, scale, quat)


if __name__ == "__main__":
    main()
