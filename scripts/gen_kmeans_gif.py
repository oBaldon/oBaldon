#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class Frame:
    centers: np.ndarray  # (k,2)
    labels: np.ndarray   # (n,)
    inertia: float


def seed_from_today() -> int:
    d = date.today()
    return (d.year * 10000 + d.month * 100 + d.day) % (2**32 - 1)


def gen_points(rng: np.random.Generator, n: int = 260) -> np.ndarray:
    means = np.array([[-2.0, -1.5], [2.2, 1.7], [-1.0, 2.3], [2.5, -2.0]])
    covs = [
        np.array([[0.45, 0.12], [0.12, 0.35]]),
        np.array([[0.35, -0.10], [-0.10, 0.40]]),
        np.array([[0.30, 0.08], [0.08, 0.30]]),
        np.array([[0.40, -0.06], [-0.06, 0.30]]),
    ]
    pts = []
    for _ in range(n):
        j = int(rng.integers(0, len(means)))
        pts.append(rng.multivariate_normal(means[j], covs[j]))
    return np.array(pts, dtype=float)


def kmeans_frames(X: np.ndarray, k: int, iters: int, rng: np.random.Generator) -> List[Frame]:
    n = X.shape[0]
    centers = X[rng.choice(n, size=k, replace=False)].copy()
    frames: List[Frame] = []
    for _ in range(iters):
        d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = d2.argmin(axis=1)

        new_centers = centers.copy()
        for j in range(k):
            m = labels == j
            if m.any():
                new_centers[j] = X[m].mean(axis=0)

        inertia = float(np.take_along_axis(d2, labels[:, None], axis=1).sum())
        frames.append(Frame(centers=new_centers.copy(), labels=labels.copy(), inertia=inertia))
        centers = new_centers
    return frames


def normalize_to_rect(X: np.ndarray, x0: float, y0: float, w: float, h: float, pad: float = 20.0) -> np.ndarray:
    xmin, ymin = X.min(axis=0)
    xmax, ymax = X.max(axis=0)
    sx = (w - 2 * pad) / (xmax - xmin + 1e-9)
    sy = (h - 2 * pad) / (ymax - ymin + 1e-9)
    s = min(sx, sy)
    Yn = (X - np.array([xmin, ymin])) * s
    Yn[:, 0] = x0 + pad + Yn[:, 0]
    Yn[:, 1] = y0 + h - pad - Yn[:, 1]
    return Yn


def main() -> None:
    Path("assets").mkdir(parents=True, exist_ok=True)
    out = Path("assets/kmeans.gif")

    seed = seed_from_today()
    rng = np.random.default_rng(seed)

    W, H = 980, 360  # grande
    k = 4
    iters = 22

    # layout: painel esquerdo (info) + painel direito (plot)
    card_pad = 18
    inner_x = card_pad
    inner_y = card_pad
    inner_w = W - 2 * card_pad
    inner_h = H - 2 * card_pad

    left_w = 360
    gap = 18
    plot_x0 = inner_x + left_w + gap
    plot_y0 = inner_y + 22
    plot_w = inner_w - left_w - gap - 18
    plot_h = inner_h - 44

    X = gen_points(rng)
    frames = kmeans_frames(X, k, iters, rng)

    # normaliza pontos e centróides para o retângulo do plot
    Xv = normalize_to_rect(X, plot_x0, plot_y0, plot_w, plot_h, pad=24.0)
    centers_all = np.vstack([f.centers for f in frames])
    concat = np.vstack([X, centers_all])
    concat_v = normalize_to_rect(concat, plot_x0, plot_y0, plot_w, plot_h, pad=24.0)
    centers_v = concat_v[X.shape[0]:].reshape(len(frames), k, 2)

    colors = [(37, 99, 235), (22, 163, 74), (245, 158, 11), (239, 68, 68)]
    bg = (11, 18, 32)
    card = (15, 23, 42)
    stroke = (31, 41, 55)
    text = (229, 231, 235)
    muted = (148, 163, 184)

    images: List[Image.Image] = []
    inertia0 = frames[0].inertia
    inertiaN = frames[-1].inertia

    for t, fr in enumerate(frames):
        im = Image.new("RGB", (W, H), bg)
        dr = ImageDraw.Draw(im)

        # card
        dr.rounded_rectangle([inner_x, inner_y, inner_x + inner_w, inner_y + inner_h], radius=16, fill=card, outline=stroke, width=2)

        # títulos
        dr.text((inner_x + 22, inner_y + 16), "K-means", fill=text)
        dr.text((inner_x + 120, inner_y + 20), f"k={k} | iteração={t+1}/{len(frames)} | seed={seed}", fill=muted)

        # barra "inertia"
        bar_x, bar_y, bar_w, bar_h = inner_x + 22, inner_y + 50, left_w - 44, 12
        dr.rounded_rectangle([bar_x, bar_y, bar_x + bar_w, bar_y + bar_h], radius=8, fill=(17, 24, 39), outline=stroke)
        prog = 1.0 - (fr.inertia - inertiaN) / (inertia0 - inertiaN + 1e-9)
        fill_w = int(bar_w * max(0.0, min(1.0, prog)))
        dr.rounded_rectangle([bar_x, bar_y, bar_x + fill_w, bar_y + bar_h], radius=8, fill=(34, 197, 94))

        # legenda
        lx, ly = inner_x + 22, inner_y + 86
        for j in range(k):
            dr.rectangle([lx, ly + j * 22 - 10, lx + 12, ly + j * 22 + 2], fill=colors[j])
            dr.text((lx + 18, ly + j * 22 - 12), f"cluster {j}", fill=muted)

        # plot background
        dr.rounded_rectangle([plot_x0 - 12, plot_y0 - 12, plot_x0 + plot_w + 12, plot_y0 + plot_h + 12], radius=14, fill=(11, 18, 32), outline=stroke)

        # pontos
        r = 3
        for i, (px, py) in enumerate(Xv):
            c = colors[int(fr.labels[i])]
            dr.ellipse([px - r, py - r, px + r, py + r], fill=c)

        # centróides
        for j in range(k):
            cx, cy = centers_v[t, j]
            dr.ellipse([cx - 10, cy - 10, cx + 10, cy + 10], fill=colors[j], outline=(229, 231, 235), width=2)

        images.append(im)

    # salvar gif
    images[0].save(
        out,
        save_all=True,
        append_images=images[1:],
        duration=140,
        loop=0,
        optimize=True,
    )
    print(f"[ok] wrote {out}")


if __name__ == "__main__":
    main()