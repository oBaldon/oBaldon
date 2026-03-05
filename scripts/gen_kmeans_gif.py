#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw


# =========================
# Configuração (ajuste aqui)
# =========================
CANVAS_W, CANVAS_H = 980, 360  # grande
K = 4
N_POINTS = 260

ITERS = 26           # iterações reais
TWEEN_STEPS = 10     # mais tween = menos "quadro a quadro"
HOLD_LAST = 16
FRAME_MS = 55        # ~18 fps perceptivo

POINT_R = 3
CENTER_R = 10

# Layout
CARD_PAD = 18
LEFT_W = 360
GAP = 18

# Visual
BG = (11, 18, 32)
CARD = (15, 23, 42)
STROKE = (31, 41, 55)
TEXT = (229, 231, 235)
MUTED = (148, 163, 184)
BAR_BG = (17, 24, 39)
BAR_FILL = (34, 197, 94)

COLORS = [
    (37, 99, 235),   # azul
    (22, 163, 74),   # verde
    (245, 158, 11),  # amarelo
    (239, 68, 68),   # vermelho
]


@dataclass(frozen=True)
class KMFrame:
    centers: np.ndarray  # (k,2)
    labels: np.ndarray   # (n,)
    inertia: float


def seed_from_today() -> int:
    d = date.today()
    return (d.year * 10000 + d.month * 100 + d.day) % (2**32 - 1)


def gen_points(rng: np.random.Generator, n: int) -> np.ndarray:
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


def kmeans_frames(X: np.ndarray, k: int, iters: int, rng: np.random.Generator) -> List[KMFrame]:
    n = X.shape[0]
    centers = X[rng.choice(n, size=k, replace=False)].copy()
    frames: List[KMFrame] = []

    for _ in range(iters):
        d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = d2.argmin(axis=1)

        new_centers = centers.copy()
        for j in range(k):
            m = labels == j
            if m.any():
                new_centers[j] = X[m].mean(axis=0)

        inertia = float(np.take_along_axis(d2, labels[:, None], axis=1).sum())
        frames.append(KMFrame(centers=new_centers.copy(), labels=labels.copy(), inertia=inertia))
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


def lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return (1.0 - t) * a + t * b


def lerp_color(c0: Tuple[int, int, int], c1: Tuple[int, int, int], t: float) -> Tuple[int, int, int]:
    return (
        int(round((1.0 - t) * c0[0] + t * c1[0])),
        int(round((1.0 - t) * c0[1] + t * c1[1])),
        int(round((1.0 - t) * c0[2] + t * c1[2])),
    )


def main() -> None:
    Path("assets").mkdir(parents=True, exist_ok=True)
    out = Path("assets/kmeans.gif")

    seed = seed_from_today()
    rng = np.random.default_rng(seed)

    inner_x, inner_y = CARD_PAD, CARD_PAD
    inner_w, inner_h = CANVAS_W - 2 * CARD_PAD, CANVAS_H - 2 * CARD_PAD

    plot_x0 = inner_x + LEFT_W + GAP
    plot_y0 = inner_y + 22
    plot_w = inner_w - LEFT_W - GAP - 18
    plot_h = inner_h - 44

    X = gen_points(rng, N_POINTS)
    frames = kmeans_frames(X, K, ITERS, rng)

    # pontos sempre no painel correto
    Xv = normalize_to_rect(X, plot_x0, plot_y0, plot_w, plot_h, pad=24.0)

    # centróides normalizados por frame
    centers_all = np.vstack([f.centers for f in frames])
    concat = np.vstack([X, centers_all])
    concat_v = normalize_to_rect(concat, plot_x0, plot_y0, plot_w, plot_h, pad=24.0)
    centers_v = concat_v[X.shape[0]:].reshape(len(frames), K, 2)

    inertia0 = float(frames[0].inertia)
    inertiaN = float(frames[-1].inertia)

    images: List[Image.Image] = []

    # Monta timeline com tween e cross-fade de labels (principal ganho)
    for i in range(len(frames) - 1):
        c0 = centers_v[i]
        c1 = centers_v[i + 1]
        l0 = frames[i].labels
        l1 = frames[i + 1].labels
        in0 = float(frames[i].inertia)
        in1 = float(frames[i + 1].inertia)

        for s in range(TWEEN_STEPS):
            t = s / float(TWEEN_STEPS)  # 0..(TWEEN_STEPS-1)/TWEEN_STEPS
            centers_xy = lerp(c0, c1, t)
            inertia = (1.0 - t) * in0 + t * in1

            im = Image.new("RGB", (CANVAS_W, CANVAS_H), BG)
            dr = ImageDraw.Draw(im)

            dr.rounded_rectangle(
                [inner_x, inner_y, inner_x + inner_w, inner_y + inner_h],
                radius=16, fill=CARD, outline=STROKE, width=2
            )

            dr.text((inner_x + 22, inner_y + 16), "K-means", fill=TEXT)
            dr.text(
                (inner_x + 120, inner_y + 20),
                f"k={K} | iteração={i+1}/{len(frames)} | seed={seed}",
                fill=MUTED
            )

            # barra inertia
            bar_x, bar_y, bar_w, bar_h = inner_x + 22, inner_y + 50, LEFT_W - 44, 12
            dr.rounded_rectangle([bar_x, bar_y, bar_x + bar_w, bar_y + bar_h], radius=8, fill=BAR_BG, outline=STROKE)
            prog = 1.0 - (inertia - inertiaN) / (inertia0 - inertiaN + 1e-9)
            prog = max(0.0, min(1.0, prog))
            fill_w = int(bar_w * prog)
            dr.rounded_rectangle([bar_x, bar_y, bar_x + fill_w, bar_y + bar_h], radius=8, fill=BAR_FILL)

            # legenda
            lx, ly = inner_x + 22, inner_y + 86
            for j in range(K):
                dr.rectangle([lx, ly + j * 22 - 10, lx + 12, ly + j * 22 + 2], fill=COLORS[j])
                dr.text((lx + 18, ly + j * 22 - 12), f"cluster {j}", fill=MUTED)

            # plot bg
            dr.rounded_rectangle(
                [plot_x0 - 12, plot_y0 - 12, plot_x0 + plot_w + 12, plot_y0 + plot_h + 12],
                radius=14, fill=(11, 18, 32), outline=STROKE
            )

            # pontos com cross-fade de cor se trocar de cluster
            r = POINT_R
            for p_idx, (px, py) in enumerate(Xv):
                c_old = COLORS[int(l0[p_idx])]
                c_new = COLORS[int(l1[p_idx])]
                c = c_old if c_old == c_new else lerp_color(c_old, c_new, t)
                dr.ellipse([px - r, py - r, px + r, py + r], fill=c)

            # centróides suavizados
            for j in range(K):
                cx, cy = float(centers_xy[j, 0]), float(centers_xy[j, 1])
                dr.ellipse(
                    [cx - CENTER_R, cy - CENTER_R, cx + CENTER_R, cy + CENTER_R],
                    fill=COLORS[j], outline=TEXT, width=2
                )

            dr.text((inner_x + 22, inner_y + inner_h - 26), "gerado automaticamente (GIF)", fill=(100, 116, 139))
            images.append(im)

    # último frame + hold
    last_centers = centers_v[-1]
    last_labels = frames[-1].labels
    last_inertia = float(frames[-1].inertia)

    def render_last() -> Image.Image:
        im = Image.new("RGB", (CANVAS_W, CANVAS_H), BG)
        dr = ImageDraw.Draw(im)
        dr.rounded_rectangle([inner_x, inner_y, inner_x + inner_w, inner_y + inner_h], radius=16, fill=CARD, outline=STROKE, width=2)
        dr.text((inner_x + 22, inner_y + 16), "K-means", fill=TEXT)
        dr.text((inner_x + 120, inner_y + 20), f"k={K} | final | seed={seed}", fill=MUTED)

        bar_x, bar_y, bar_w, bar_h = inner_x + 22, inner_y + 50, LEFT_W - 44, 12
        dr.rounded_rectangle([bar_x, bar_y, bar_x + bar_w, bar_y + bar_h], radius=8, fill=BAR_BG, outline=STROKE)
        prog = 1.0 - (last_inertia - inertiaN) / (inertia0 - inertiaN + 1e-9)
        prog = max(0.0, min(1.0, prog))
        dr.rounded_rectangle([bar_x, bar_y, bar_x + int(bar_w * prog), bar_y + bar_h], radius=8, fill=BAR_FILL)

        lx, ly = inner_x + 22, inner_y + 86
        for j in range(K):
            dr.rectangle([lx, ly + j * 22 - 10, lx + 12, ly + j * 22 + 2], fill=COLORS[j])
            dr.text((lx + 18, ly + j * 22 - 12), f"cluster {j}", fill=MUTED)

        dr.rounded_rectangle([plot_x0 - 12, plot_y0 - 12, plot_x0 + plot_w + 12, plot_y0 + plot_h + 12],
                             radius=14, fill=(11, 18, 32), outline=STROKE)

        r = POINT_R
        for p_idx, (px, py) in enumerate(Xv):
            c = COLORS[int(last_labels[p_idx])]
            dr.ellipse([px - r, py - r, px + r, py + r], fill=c)

        for j in range(K):
            cx, cy = float(last_centers[j, 0]), float(last_centers[j, 1])
            dr.ellipse([cx - CENTER_R, cy - CENTER_R, cx + CENTER_R, cy + CENTER_R], fill=COLORS[j], outline=TEXT, width=2)

        dr.text((inner_x + 22, inner_y + inner_h - 26), "gerado automaticamente (GIF)", fill=(100, 116, 139))
        return im

    final_img = render_last()
    images.append(final_img)
    for _ in range(HOLD_LAST):
        images.append(final_img)

    images[0].save(
        out,
        save_all=True,
        append_images=images[1:],
        duration=FRAME_MS,
        loop=0,
        optimize=False
    )
    print(f"[ok] wrote {out}")


if __name__ == "__main__":
    main()