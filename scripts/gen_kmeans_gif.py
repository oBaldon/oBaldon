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
CANVAS_W, CANVAS_H = 980, 360  # "grande"
K = 4
N_POINTS = 260

ITERS = 24           # iterações reais do k-means
TWEEN_STEPS = 6      # frames intermediários entre iterações (fluidez)
HOLD_LAST = 14       # frames repetidos ao final (para "ver" o resultado)

FRAME_MS = 55        # ~18 fps (mais fluido sem ficar pesado)
POINT_R = 3
CENTER_R = 10

# Layout: painel esquerdo + plot à direita
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
    # Mistura de gaussianas (esteticamente consistente)
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
        d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)  # (n,k)
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
    # SVG cresce para baixo; aqui é raster, mas mantemos eixo Y invertido para visual semelhante
    Yn[:, 1] = y0 + h - pad - Yn[:, 1]
    return Yn


def lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return (1.0 - t) * a + t * b


def main() -> None:
    Path("assets").mkdir(parents=True, exist_ok=True)
    out = Path("assets/kmeans.gif")

    seed = seed_from_today()
    rng = np.random.default_rng(seed)

    # Layout do card
    inner_x, inner_y = CARD_PAD, CARD_PAD
    inner_w, inner_h = CANVAS_W - 2 * CARD_PAD, CANVAS_H - 2 * CARD_PAD

    plot_x0 = inner_x + LEFT_W + GAP
    plot_y0 = inner_y + 22
    plot_w = inner_w - LEFT_W - GAP - 18
    plot_h = inner_h - 44

    X = gen_points(rng, N_POINTS)
    frames = kmeans_frames(X, K, ITERS, rng)

    # Normaliza pontos para o retângulo do plot (corrige "desenhar no lugar errado")
    Xv = normalize_to_rect(X, plot_x0, plot_y0, plot_w, plot_h, pad=24.0)

    # Normaliza centróides por frame usando o mesmo mapeamento: normaliza concatenado e fatiar
    centers_all = np.vstack([f.centers for f in frames])
    concat = np.vstack([X, centers_all])
    concat_v = normalize_to_rect(concat, plot_x0, plot_y0, plot_w, plot_h, pad=24.0)
    centers_v = concat_v[X.shape[0]:].reshape(len(frames), K, 2)

    # Interpolação (tween) para suavidade
    interp_centers: List[np.ndarray] = []
    interp_labels: List[np.ndarray] = []
    interp_inertia: List[float] = []

    for i in range(len(frames) - 1):
        c0 = centers_v[i]
        c1 = centers_v[i + 1]
        for s in range(TWEEN_STEPS):
            t = s / float(TWEEN_STEPS)  # 0..(TWEEN_STEPS-1)/TWEEN_STEPS
            interp_centers.append(lerp(c0, c1, t))
            interp_labels.append(frames[i].labels)
            interp_inertia.append(frames[i].inertia)

    # último estado + hold
    interp_centers.append(centers_v[-1])
    interp_labels.append(frames[-1].labels)
    interp_inertia.append(frames[-1].inertia)

    for _ in range(HOLD_LAST):
        interp_centers.append(centers_v[-1])
        interp_labels.append(frames[-1].labels)
        interp_inertia.append(frames[-1].inertia)

    inertia0 = float(frames[0].inertia)
    inertiaN = float(frames[-1].inertia)

    images: List[Image.Image] = []

    for idx in range(len(interp_centers)):
        labels = interp_labels[idx]
        inertia = float(interp_inertia[idx])
        centers_xy = interp_centers[idx]

        im = Image.new("RGB", (CANVAS_W, CANVAS_H), BG)
        dr = ImageDraw.Draw(im)

        # Card externo
        dr.rounded_rectangle(
            [inner_x, inner_y, inner_x + inner_w, inner_y + inner_h],
            radius=16, fill=CARD, outline=STROKE, width=2
        )

        # Título
        dr.text((inner_x + 22, inner_y + 16), "K-means", fill=TEXT)
        dr.text(
            (inner_x + 120, inner_y + 20),
            f"k={K} | frames={len(interp_centers)} | seed={seed}",
            fill=MUTED
        )

        # Barra de "inertia" (progresso)
        bar_x, bar_y, bar_w, bar_h = inner_x + 22, inner_y + 50, LEFT_W - 44, 12
        dr.rounded_rectangle([bar_x, bar_y, bar_x + bar_w, bar_y + bar_h], radius=8, fill=BAR_BG, outline=STROKE)
        prog = 1.0 - (inertia - inertiaN) / (inertia0 - inertiaN + 1e-9)
        prog = max(0.0, min(1.0, prog))
        fill_w = int(bar_w * prog)
        dr.rounded_rectangle([bar_x, bar_y, bar_x + fill_w, bar_y + bar_h], radius=8, fill=BAR_FILL)

        # Legenda
        lx, ly = inner_x + 22, inner_y + 86
        for j in range(K):
            dr.rectangle([lx, ly + j * 22 - 10, lx + 12, ly + j * 22 + 2], fill=COLORS[j])
            dr.text((lx + 18, ly + j * 22 - 12), f"cluster {j}", fill=MUTED)

        # Plot background
        dr.rounded_rectangle(
            [plot_x0 - 12, plot_y0 - 12, plot_x0 + plot_w + 12, plot_y0 + plot_h + 12],
            radius=14, fill=(11, 18, 32), outline=STROKE
        )

        # Pontos
        r = POINT_R
        for i, (px, py) in enumerate(Xv):
            c = COLORS[int(labels[i])]
            dr.ellipse([px - r, py - r, px + r, py + r], fill=c)

        # Centróides (suavizados)
        for j in range(K):
            cx, cy = float(centers_xy[j, 0]), float(centers_xy[j, 1])
            dr.ellipse([cx - CENTER_R, cy - CENTER_R, cx + CENTER_R, cy + CENTER_R], fill=COLORS[j], outline=TEXT, width=2)

        # Rodapé
        dr.text((inner_x + 22, inner_y + inner_h - 26), "gerado automaticamente (GIF)", fill=(100, 116, 139))

        images.append(im)

    # Salvar GIF
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