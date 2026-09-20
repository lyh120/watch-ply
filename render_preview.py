#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
render_preview.py -- Offline sanity-check: render colored point-cloud PLYs to a
PNG with matplotlib (no GUI needed), so you can verify a conversion at a glance.

    python render_preview.py a_rgb.ply b_rgb.ply ...
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from view_ply import load_colored_ply


def main():
    p = argparse.ArgumentParser()
    p.add_argument("inputs", nargs="+")
    p.add_argument("-o", "--output", default="demo/preview.png")
    p.add_argument("--s", type=float, default=0.5, help="marker size")
    p.add_argument("--camera-obj", help="optional camera-frustum OBJ from colmap2ply.py")
    p.add_argument("--trim-percent", type=float, default=0.0,
                   help="hide this percentile of position outliers on each side")
    args = p.parse_args()

    size = (10 * len(args.inputs), 5.5) if args.camera_obj else (7 * len(args.inputs), 7)
    fig = plt.figure(figsize=size, facecolor="#0d1117")
    for i, path in enumerate(args.inputs):
        xyz, rgb = load_colored_ply(path)
        if args.trim_percent:
            lo = np.percentile(xyz, args.trim_percent, axis=0)
            hi = np.percentile(xyz, 100 - args.trim_percent, axis=0)
            keep = np.all((xyz >= lo) & (xyz <= hi), axis=1)
            xyz, rgb = xyz[keep], rgb[keep]

        if args.camera_obj:
            ax = fig.add_subplot(1, len(args.inputs), i + 1, projection="3d")
            ax.set_facecolor("#0d1117")
            ax.scatter(xyz[:, 0], xyz[:, 2], -xyz[:, 1], c=rgb, s=args.s,
                       marker=".", linewidths=0, depthshade=False)
            verts, colors, lines = [], [], []
            with open(args.camera_obj, "r", encoding="ascii") as f:
                for line in f:
                    parts = line.split()
                    if parts[:1] == ["v"]:
                        verts.append([float(v) for v in parts[1:4]])
                        colors.append([float(v) / 255 for v in parts[4:7]])
                    elif parts[:1] == ["l"]:
                        lines.append([int(v) - 1 for v in parts[1:3]])
            verts = np.asarray(verts)
            for a, b in lines:
                color = colors[a]
                ax.plot(verts[[a, b], 0], verts[[a, b], 2], -verts[[a, b], 1],
                        color=color, linewidth=0.65, alpha=0.7)
            all_xyz = np.vstack([np.column_stack([xyz[:, 0], xyz[:, 2], -xyz[:, 1]]),
                                 np.column_stack([verts[:, 0], verts[:, 2], -verts[:, 1]])])
            lower, upper = all_xyz.min(axis=0), all_xyz.max(axis=0)
            span = np.maximum(upper - lower, 1e-6)
            margin = span * 0.05
            ax.set_xlim(lower[0] - margin[0], upper[0] + margin[0])
            ax.set_ylim(lower[1] - margin[1], upper[1] + margin[1])
            ax.set_zlim(lower[2] - margin[2], upper[2] + margin[2])
            ax.set_box_aspect(span, zoom=1.45)
            ax.view_init(elev=22, azim=-62)
            ax.set_axis_off()
        else:
            ax = fig.add_subplot(1, len(args.inputs), i + 1)
            ax.set_facecolor("#0d1117")
            ax.scatter(xyz[:, 0], xyz[:, 2], c=rgb, s=args.s, marker=".", linewidths=0)
            ax.set_aspect("equal")
            ax.invert_yaxis()          # top-down view of the scene (x-z plane)
            ax.invert_xaxis()
            ax.tick_params(colors="#8b949e")
        ax.set_title("COLMAP sparse point cloud + camera poses", color="#f0f6fc",
                     fontsize=13, pad=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fig.savefig(args.output, dpi=110)
    print("wrote", args.output)


if __name__ == "__main__":
    main()
