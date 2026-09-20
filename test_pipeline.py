#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_pipeline.py -- smoke tests for the gs2ply project. Run with no args:

    python test_pipeline.py

All tests are self-contained (generate fixtures in a temp dir).
"""
import os
import struct
import sys
import tempfile

import numpy as np

import gs2ply
import make_demo_gs
import colmap2ply

TMP = tempfile.mkdtemp(prefix="gs2ply_test_")
PASS = 0


def check(name, cond, info=""):
    global PASS
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  ({info})" if info else ""))
    if cond:
        PASS += 1
    else:
        sys.exit(f"TEST FAILED: {name}")


# ---------------------------------------------------------------------------
print("[1] SH evaluation matches 3DGS reference values")
dirs = np.array([[0.0, 0.0, 1.0]])
# band-0 only
sh0 = np.zeros((1, 1, 3))
sh0[0, 0, 1] = 1.0                      # G channel DC = 1
rgb = gs2ply.eval_sh(0, sh0, dirs) + 0.5
check("band0: 0.5 + C0 * f_dc", np.allclose(rgb, [[0.5, 0.5 + gs2ply.SH_C0, 0.5]]))

# band-1: sh[...,2] enters with +C1*z, dir=+z -> +C1
sh1 = np.zeros((1, 4, 3))
sh1[0, 2, 0] = 1.0
rgb = gs2ply.eval_sh(1, sh1, dirs) + 0.5
check("band1: +C1 * z * sh[2]", np.allclose(rgb, [[0.5 + gs2ply.SH_C1, 0.5, 0.5]]))

# band-1 with dir=-y: sh[...,1] enters with -C1*y -> +C1
dirs2 = np.array([[0.0, -1.0, 0.0]])
sh1b = np.zeros((1, 4, 3))
sh1b[0, 1, 2] = 1.0
rgb = gs2ply.eval_sh(1, sh1b, dirs2) + 0.5
check("band1: -C1 * y * sh[1] with y=-1", np.allclose(rgb, [[0.5, 0.5, 0.5 + gs2ply.SH_C1]]))

# band-2 diagonal: sh[...,6] multiplies C2[2]*(2zz-xx-yy); dir=+z -> 2*C2[2]
sh2 = np.zeros((1, 9, 3))
sh2[0, 6, 0] = 1.0
rgb = gs2ply.eval_sh(2, sh2, dirs) + 0.5
check("band2: C2[2]*(2zz-xx-yy) at +z",
      np.allclose(rgb, [[0.5 + 2 * gs2ply.SH_C2[2], 0.5, 0.5]]))

# ---------------------------------------------------------------------------
print("[2] DC round-trip on synthetic 3DGS PLY (write -> read -> convert == source colors)")
demo = os.path.join(TMP, "tiny.ply")
rng = np.random.default_rng(0)
n = 5000
src_rgb = rng.uniform(0, 1, (n, 3))
xyz = rng.uniform(-2, 2, (n, 3))
rest = rng.normal(0, 0.5, (n, 45))       # nonzero rest must NOT affect dc mode
make_demo_gs.write_3dgs_ply(demo, xyz, src_rgb, rest,
                            np.full(n, 0.9), np.full((n, 3), 0.03),
                            np.tile([1, 0, 0, 0], (n, 1)).astype(float))
fields, kind = gs2ply.load_gaussians(demo)
check("detected as 3dgs", kind == "3dgs")
rgb_dc = gs2ply.compute_rgb(fields, kind, "dc", (0, 0, 0), None)
err = np.abs(rgb_dc - src_rgb).max()
check("dc colors recover source rgb (max err < 1e-6)", err < 1e-6, f"max err {err:.2e}")

rgb_sh = gs2ply.compute_rgb(fields, kind, "sh", (0, 0, 0), None)
check("sh mode differs from dc when f_rest != 0", np.abs(rgb_sh - rgb_dc).max() > 1e-3)
rgb_sh_free = gs2ply.compute_rgb(fields, kind, "sh", (0, 0, 0), (0, 0, 1))
check("sh with constant --view-dir runs", rgb_sh_free.shape == (n, 3))

# clamp check: crazy DC values must land in [0,1]
fields["f_dc_0"][:] = 50.0
rgb_c = gs2ply.compute_rgb(fields, kind, "dc", (0, 0, 0), None)
check("dc clamp to [0,1]", rgb_c.min() >= 0.0 and rgb_c.max() <= 1.0 and rgb_c[0, 0] == 1.0)

# ---------------------------------------------------------------------------
print("[3] filters and writer")
fields, kind = gs2ply.load_gaussians(demo)
rgb = gs2ply.compute_rgb(fields, kind, "dc", (0, 0, 0), None)
out = gs2ply.apply_filters(fields, kind, rgb, min_opacity=0.5, max_scale=0.05,
                           max_points=None, seed=0)
check("min-opacity + max-scale keep everything here", out["x"].shape[0] == n)
out = gs2ply.apply_filters(fields, kind, rgb, min_opacity=0.0, max_scale=None,
                           max_points=1000, seed=0)
check("max-points subsamples", out["x"].shape[0] == 1000)
col = os.path.join(TMP, "out.ply")
gs2ply.write_colored_ply(col, out["x"], out["y"], out["z"], out["rgb"])
back = gs2ply.read_ply_vertices(col)
check("binary PLY round-trip (positions)",
      np.allclose(back["x"], out["x"], atol=1e-6))
check("binary PLY round-trip (colors)",
      np.array_equal(np.stack([back["red"], back["green"], back["blue"]], axis=1),
                     out["rgb"]))
col_a = os.path.join(TMP, "out_ascii.ply")
gs2ply.write_colored_ply(col_a, out["x"], out["y"], out["z"], out["rgb"], ascii_out=True)
back_a = gs2ply.read_ply_vertices(col_a)
check("ascii PLY round-trip", np.allclose(back_a["x"], out["x"], atol=1e-5))

# already-colored ply passthrough
check("colored-ply passthrough detected", gs2ply.load_gaussians(col)[1] == "rgbfply")

# ---------------------------------------------------------------------------
print("[4] .splat format")
splat_path = os.path.join(TMP, "tiny.splat")
rng2 = np.random.default_rng(1)
m = 300
rec = np.zeros(m, dtype=np.dtype([
    ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
    ("sx", "<f4"), ("sy", "<f4"), ("sz", "<f4"),
    ("r", "u1"), ("g", "u1"), ("b", "u1"), ("a", "u1"),
    ("q0", "u1"), ("q1", "u1"), ("q2", "u1"), ("q3", "u1")]))
rec["x"] = rng2.uniform(-1, 1, m); rec["y"] = rng2.uniform(-1, 1, m); rec["z"] = rng2.uniform(-1, 1, m)
rec["sx"] = rec["sy"] = rec["sz"] = 0.02
rec["r"] = rng2.integers(0, 256, m); rec["g"] = rng2.integers(0, 256, m); rec["b"] = rng2.integers(0, 256, m)
rec["a"] = 255
rec.tofile(splat_path)
fields_s, kind_s = gs2ply.load_gaussians(splat_path)
check(".splat detected", kind_s == "splat")
rgb_s = gs2ply.compute_rgb(fields_s, kind_s, "dc", (0, 0, 0), None)
check(".splat colors preserved",
      np.allclose(rgb_s * 255,
                  np.stack([rec["r"], rec["g"], rec["b"]], axis=1), atol=0.51))

# ---------------------------------------------------------------------------
print("[5] COLMAP bin/txt parsing (fixture round-trip)")
def make_colmap_fixture(folder):
    os.makedirs(folder, exist_ok=True)
    rng3 = np.random.default_rng(2)
    npts = 200
    ids = np.arange(1, npts + 1, dtype=np.int64)
    xyz = rng3.uniform(-3, 3, (npts, 3))
    rgb = rng3.integers(0, 256, (npts, 3)).astype(np.uint8)
    track = rng3.integers(1, 5, npts)
    # --- points3D.bin
    with open(os.path.join(folder, "points3D.bin"), "wb") as f:
        f.write(struct.pack("<Q", npts))
        for i in range(npts):
            f.write(struct.pack("<Q", ids[i]))
            f.write(struct.pack("<3d", *xyz[i]))
            f.write(struct.pack("<3B", *rgb[i]))
            f.write(struct.pack("<d", 0.5))
            f.write(struct.pack("<Q", track[i]))
            f.write(struct.pack("<II", 1, 2) * track[i])
    # --- points3D.txt
    with open(os.path.join(folder, "points3D.txt"), "w") as f:
        f.write("# dummy header comment\n")
        for i in range(npts):
            trk = " ".join(["1 2"] * int(track[i]))
            f.write(f"{ids[i]} {xyz[i][0]} {xyz[i][1]} {xyz[i][2]} "
                    f"{rgb[i][0]} {rgb[i][1]} {rgb[i][2]} 0.5 {trk}\n")
    # --- images.txt (2 images with quaternions) & cameras.txt
    with open(os.path.join(folder, "images.txt"), "w") as f:
        f.write("# image list\n")
        f.write("1 1 0 0 0 0 0 -5 1 img1.jpg 0 0 -1\n0 0\n")
        f.write("2 0.9239 0 0 0.3827 0 0 -5 1 img2.jpg 0 0 -1\n0 0\n")
    with open(os.path.join(folder, "cameras.txt"), "w") as f:
        f.write("1 SIMPLE_PINHOLE 1920 1080 1500 960 540\n")
    # --- images.bin + cameras.bin, exactly per COLMAP spec:
    #     image names are NUL-terminated; collection counts are uint64.
    imgs = {1: (np.array([1., 0, 0, 0]), np.array([0., 0, -5]), 1, "img1.jpg"),
            2: (np.array([0.9239, 0, 0, 0.3827]), np.array([0., 0, -5]), 1, "img2.jpg")}
    with open(os.path.join(folder, "images.bin"), "wb") as f:
        f.write(struct.pack("<Q", len(imgs)))
        for iid, (q, t, cid, name) in imgs.items():
            f.write(struct.pack("<I", iid))
            f.write(struct.pack("<4d", *q))
            f.write(struct.pack("<3d", *t))
            f.write(struct.pack("<I", cid))
            nb = name.encode("utf-8")
            f.write(nb + b"\0")
            f.write(struct.pack("<Q", 1))                 # 1 point2D
            f.write(struct.pack("<ddq", 1.0, 2.0, -1))
    with open(os.path.join(folder, "cameras.bin"), "wb") as f:
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<Ii", 1, 1))                 # id, PINHOLE
        f.write(struct.pack("<QQ", 1920, 1080))
        f.write(struct.pack("<4d", 1500.0, 1500.0, 960.0, 540.0))
    return xyz, rgb

fx = os.path.join(TMP, "sparse0")
xyz_ref, rgb_ref = make_colmap_fixture(fx)
ids_b, xyz_b, rgb_b, trk_b = colmap2ply.read_points3D_bin(os.path.join(fx, "points3D.bin"))
ids_t, xyz_t, rgb_t, trk_t = colmap2ply.read_points3D_txt(os.path.join(fx, "points3D.txt"))
check("bin == txt (xyz)", np.allclose(xyz_b, xyz_t, atol=1e-9) and np.allclose(xyz_b, xyz_ref, atol=1e-12))
check("bin == txt (rgb)", np.array_equal(rgb_b, rgb_t) and np.array_equal(rgb_b, rgb_ref))
check("bin == txt (track)", np.array_equal(trk_b, trk_t))

# camera readers + frustum writer
imgs = colmap2ply.read_images_txt(os.path.join(fx, "images.txt"))
cams = colmap2ply.read_cameras_txt(os.path.join(fx, "cameras.txt"))
check("images parsed", len(imgs) == 2 and np.allclose(imgs[1][1], [0, 0, -5]))

# Standard COLMAP binary variants must parse identically.
imgs_b = colmap2ply.read_images_bin(os.path.join(fx, "images.bin"))
cams_b = colmap2ply.read_cameras_bin(os.path.join(fx, "cameras.bin"))
check("images.bin == images.txt (quat/t/cam)",
      len(imgs_b) == len(imgs) and
      all(np.allclose(imgs_b[i][0], imgs[i][0]) and np.allclose(imgs_b[i][1], imgs[i][1])
          and imgs_b[i][2] == imgs[i][2] for i in imgs))
check("images.bin names recovered",
      imgs_b[1][3] == "img1.jpg" and imgs_b[2][3] == "img2.jpg")
check("cameras.bin params",
      cams_b[1][1] == 1920 and np.allclose(cams_b[1][3][:4], [1500, 1500, 960, 540]))
obj_path = os.path.join(fx, "cameras.obj")
colmap2ply.write_camera_obj(obj_path, imgs, cams, depth=1.0)
obj_lines = open(obj_path).read().splitlines()
check("frustum obj written (2 cams x 5 verts x 8 lines)",
      sum(1 for l in obj_lines if l.startswith("v ")) == 10
      and sum(1 for l in obj_lines if l.startswith("l ")) == 16)

class A:  # minimal args namespace for convert()
    output = os.path.join(TMP, "colmap_out.ply")
    cameras = False
    cam_scale = None
    min_track = 2
    max_points = None
    ascii = False
    seed = 0
colmap2ply.convert(os.path.join(fx, "points3D.txt"), A())
import gs2ply as _g
fields_c = _g.read_ply_vertices(A.output)
kept = trk_t >= 2
check("colmap end-to-end colors match source",
      np.array_equal(np.stack([fields_c["red"], fields_c["green"], fields_c["blue"]], axis=1),
                     rgb_t[kept]))

print(f"\nAll {PASS} checks passed.  (fixtures in {TMP})")
