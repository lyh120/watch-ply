#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
view_ply.py -- Quick interactive viewer for colored point-cloud PLYs, built on
Open3D. Handy when MeshLab/CloudCompare is not installed.

    pip install open3d
    python view_ply.py model_rgb.ply
    python view_ply.py model_rgb.ply --point-size 3
"""
import argparse
import os
import sys

import numpy as np

from gs2ply import read_ply_vertices


def load_colored_ply(path):
    fields = read_ply_vertices(path)
    xyz = np.stack([fields["x"], fields["y"], fields["z"]], axis=1).astype(np.float64)
    rgb = None
    for name in ("red", "r"):
        if name in fields:
            rgb = np.stack([fields[name], fields["green" if name == "red" else "g"],
                            fields["blue" if name == "red" else "b"]], axis=1)
            break
    if rgb is not None:
        rgb = rgb.astype(np.float64)
        if rgb.max() > 1.0:
            rgb /= 255.0
    return xyz, rgb


def main():
    p = argparse.ArgumentParser(description="Interactive Open3D viewer for colored PLYs")
    p.add_argument("input", help="colored point-cloud PLY")
    p.add_argument("--point-size", type=float, default=2.0)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=800)
    args = p.parse_args()

    try:
        import open3d as o3d
    except ImportError:
        sys.exit("open3d is not installed. Run:  pip install open3d\n"
                 "(or open the PLY in MeshLab / CloudCompare instead)")

    xyz, rgb = load_colored_ply(args.input)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if rgb is not None:
        pcd.colors = o3d.utility.Vector3dVector(rgb)
    else:
        print("[view_ply] file has no RGB; showing geometry only")

    vis = o3d.visualization.Visualizer()
    vis.create_window(f"view_ply - {os.path.basename(args.input)}",
                      width=args.width, height=args.height)
    vis.add_geometry(pcd)
    opt = vis.get_render_option()
    opt.point_size = args.point_size
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
