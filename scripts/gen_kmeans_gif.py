#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# =========================
# Configuração (ajuste aqui)
# =========================
CANVAS_W, CANVAS_H = 980, 360

# Fonte (garante UTF-8/acentos no GitHub Actions em ubuntu-latest)
FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)

K = 4
N_POINTS = 260

EPOCHS = 26               # aparece como "iteração 1/26 ... 26/26"
STEPS_PER_EPOCH = 14      # micro-passos por época (controle de duração)
BATCH_SIZE = 28           # tamanho do mini-batch (pequeno = movimento suave)

# Frames
TWEEN_PER_STEP = 2        # frames intermediários entre dois micro-estados (>=1)
FRAME_MS = 55
HOLD_LAST = 14

# Layout
CARD_PAD = 18
LEFT_W = 320              # reduzido para sobrar mais espaço ao plot (corrige “metade direita vazia”)
GAP = 18

# Header/spacing (corrige encavalamento de título/legenda)
HEADER_H = 44

# Visual
BG = (11, 18, 32)
CARD = (15, 23, 42)
STROKE = (31, 41, 55)
TEXT = (229, 231, 235)
MUTED = (148, 163, 184)
BAR_BG = (17, 24, 39)
BAR_FILL = (34, 197, 94)

POINT_R = 3
CENTER_R = 10

COLORS = [
    (37, 99, 235),   # azul
    (22, 163, 74),   # verde
    (245, 158, 11),  # amarelo
    (239, 68, 68),   # vermelho
]


@dataclass(frozen=True)
class VizState:
    centers: np.ndarray  # (k,2) em coordenadas "do dado"
    labels: np.ndarray   # (n,) labels (0..k-1) para visualização
    inertia: float       # métrica para barra (aprox, mas consistente)


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


def assign_labels_and_inertia(X: np.ndarray, centers: np.ndarray) -> Tuple[np.ndarray, float]:
    d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)  # (n,k)
    labels = d2.argmin(axis=1)
    inertia = float(np.take_along_axis(d2, labels[:, None], axis=1).sum())
    return labels, inertia


def normalize_to_rect(
    X: np.ndarray, x0: float, y0: float, w: float, h: float, pad: float = 20.0
) -> np.ndarray:
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


def fit_text(dr: ImageDraw.ImageDraw, text: str, max_w: int) -> str:
    """
    Trunca com '…' para caber em max_w pixels.
    """
    if dr.textlength(text, font=FONT) <= max_w:
        return text

    ell = "…"
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi) // 2
        cand = text[:mid] + ell
        if dr.textlength(cand, font=FONT) <= max_w:
            lo = mid + 1
        else:
            hi = mid
    return text[:max(0, lo - 1)] + ell


def mini_batch_kmeans_states(
    X: np.ndarray,
    k: int,
    epochs: int,
    steps_per_epoch: int,
    batch_size: int,
    rng: np.random.Generator
) -> List[VizState]:
    n = X.shape[0]
    centers = X[rng.choice(n, size=k, replace=False)].copy()

    counts = np.ones(k, dtype=np.int64)

    labels, inertia = assign_labels_and_inertia(X, centers)
    states: List[VizState] = [VizState(centers=centers.copy(), labels=labels.copy(), inertia=float(inertia))]

    for _epoch in range(epochs):
        for _step in range(steps_per_epoch):
            batch_idx = rng.choice(n, size=min(batch_size, n), replace=False)
            Xb = X[batch_idx]

            d2b = ((Xb[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            lb = d2b.argmin(axis=1)

            for xi, cj in zip(Xb, lb, strict=False):
                counts[cj] += 1
                eta = 1.0 / counts[cj]
                centers[cj] = (1.0 - eta) * centers[cj] + eta * xi

            labels, inertia = assign_labels_and_inertia(X, centers)
            states.append(VizState(centers=centers.copy(), labels=labels.copy(), inertia=float(inertia)))

    return states


def main() -> None:
    Path("assets").mkdir(parents=True, exist_ok=True)
    out = Path("assets/kmeans.gif")

    seed = seed_from_today()
    rng = np.random.default_rng(seed)

    # Layout do card
    inner_x, inner_y = CARD_PAD, CARD_PAD
    inner_w, inner_h = CANVAS_W - 2 * CARD_PAD, CANVAS_H - 2 * CARD_PAD

    # Ajuste: reservar header e evitar que legenda encoste no topo
    plot_x0 = inner_x + LEFT_W + GAP
    plot_y0 = inner_y + HEADER_H
    plot_w = inner_w - LEFT_W - GAP - 18
    plot_h = inner_h - (HEADER_H + 22)

    # Dados e estados
    X = gen_points(rng, N_POINTS)
    states = mini_batch_kmeans_states(
        X=X,
        k=K,
        epochs=EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        batch_size=BATCH_SIZE,
        rng=rng
    )

    # Normalização para o plot (agora com área maior e centralização correta)
    Xv = normalize_to_rect(X, plot_x0, plot_y0, plot_w, plot_h, pad=24.0)

    centers_all = np.vstack([st.centers for st in states])
    concat = np.vstack([X, centers_all])
    concat_v = normalize_to_rect(concat, plot_x0, plot_y0, plot_w, plot_h, pad=24.0)
    centers_v_all = concat_v[X.shape[0]:].reshape(len(states), K, 2)

    inertia0 = float(states[0].inertia)
    inertiaN = float(states[-1].inertia)

    images: List[Image.Image] = []

    # estados = 1 + EPOCHS*STEPS_PER_EPOCH
    def state_to_epoch_step(s_idx: int) -> Tuple[int, int]:
        if s_idx == 0:
            return (0, 0)
        s = s_idx - 1
        epoch = s // STEPS_PER_EPOCH + 1
        step = s % STEPS_PER_EPOCH + 1
        return (epoch, step)

    def draw_frame(
        centers_xy: np.ndarray,
        labels_from: np.ndarray,
        labels_to: np.ndarray,
        blend_t: float,
        inertia: float,
        epoch_idx: int
    ) -> Image.Image:
        im = Image.new("RGB", (CANVAS_W, CANVAS_H), BG)
        dr = ImageDraw.Draw(im)

        dr.rounded_rectangle(
            [inner_x, inner_y, inner_x + inner_w, inner_y + inner_h],
            radius=16, fill=CARD, outline=STROKE, width=2
        )

        # Header (fixo, sem sobrepor)
        title = "K-means (mini-batch)"
        dr.text((inner_x + 22, inner_y + 14), title, fill=TEXT, font=FONT)

        meta = f"k={K} | iteração={epoch_idx}/{EPOCHS} | seed={seed}"
        meta_x = inner_x + 22 + int(dr.textlength(title, font=FONT)) + 14
        meta_y = inner_y + 16
        meta_max_w = int((inner_x + inner_w) - 22 - meta_x)
        meta = fit_text(dr, meta, meta_max_w)
        dr.text((meta_x, meta_y), meta, fill=MUTED, font=FONT)

        # Barra inertia
        bar_x, bar_y, bar_w, bar_h = inner_x + 22, inner_y + 44, LEFT_W - 44, 12
        dr.rounded_rectangle([bar_x, bar_y, bar_x + bar_w, bar_y + bar_h], radius=8, fill=BAR_BG, outline=STROKE)
        prog = 1.0 - (inertia - inertiaN) / (inertia0 - inertiaN + 1e-9)
        prog = max(0.0, min(1.0, prog))
        dr.rounded_rectangle([bar_x, bar_y, bar_x + int(bar_w * prog), bar_y + bar_h], radius=8, fill=BAR_FILL)

        # Legenda (empurrada para baixo para não encavalar no header)
        lx, ly = inner_x + 22, inner_y + 74
        for j in range(K):
            dr.rectangle([lx, ly + j * 22 - 10, lx + 12, ly + j * 22 + 2], fill=COLORS[j])
            dr.text((lx + 18, ly + j * 22 - 12), f"cluster {j}", fill=MUTED, font=FONT)

        # Plot BG (agora ocupa quase todo lado direito e não “vaza”)
        dr.rounded_rectangle(
            [plot_x0 - 12, plot_y0 - 12, plot_x0 + plot_w + 12, plot_y0 + plot_h + 12],
            radius=14, fill=(11, 18, 32), outline=STROKE
        )

        # Pontos
        r = POINT_R
        for p_idx, (px, py) in enumerate(Xv):
            c0 = COLORS[int(labels_from[p_idx])]
            c1 = COLORS[int(labels_to[p_idx])]
            c = c0 if c0 == c1 else lerp_color(c0, c1, blend_t)
            dr.ellipse([px - r, py - r, px + r, py + r], fill=c)

        # Centróides
        for j in range(K):
            cx, cy = float(centers_xy[j, 0]), float(centers_xy[j, 1])
            dr.ellipse(
                [cx - CENTER_R, cy - CENTER_R, cx + CENTER_R, cy + CENTER_R],
                fill=COLORS[j], outline=TEXT, width=2
            )

        return im

    for i in range(len(states) - 1):
        st0 = states[i]
        st1 = states[i + 1]
        c0 = centers_v_all[i]
        c1 = centers_v_all[i + 1]

        epoch, _step = state_to_epoch_step(i + 1)
        nsub = max(1, TWEEN_PER_STEP)

        for s in range(nsub):
            t = s / float(nsub)  # 0..(nsub-1)/nsub
            centers_xy = lerp(c0, c1, t)
            inertia = (1.0 - t) * float(st0.inertia) + t * float(st1.inertia)
            images.append(
                draw_frame(
                    centers_xy=centers_xy,
                    labels_from=st0.labels,
                    labels_to=st1.labels,
                    blend_t=t,
                    inertia=inertia,
                    epoch_idx=epoch
                )
            )

    # Hold final
    if images:
        last = images[-1]
        for _ in range(HOLD_LAST):
            images.append(last)

        images[0].save(
            out,
            save_all=True,
            append_images=images[1:],
            duration=FRAME_MS,
            loop=0,
            optimize=False
        )

    print(f"[ok] wrote {out} | frames={len(images)}")


if __name__ == "__main__":
    main()