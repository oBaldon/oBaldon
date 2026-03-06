#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# =========================
# Configuração
# =========================
CANVAS_W, CANVAS_H = 980, 360
FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)

DATA_N = 420

GRID_W = 12
GRID_H = 8

ITERATIONS = 220
SNAP_EVERY = 2          # mais snapshots
TWEEN_PER_SNAP = 4      # desacelera entre estados
HOLD_FIRST = 10
HOLD_LAST = 16

LR0 = 0.28
SIGMA0 = max(GRID_W, GRID_H) / 2.2

FRAME_MS = 55

CARD_PAD = 18
LEFT_W = 360
GAP = 18

BG = (11, 18, 32)
CARD = (15, 23, 42)
STROKE = (31, 41, 55)
TEXT = (229, 231, 235)
MUTED = (148, 163, 184)
BAR_BG = (17, 24, 39)
BAR_FILL = (34, 197, 94)

C_DATA = (100, 160, 255)
C_GRID = (56, 189, 248)
C_BMU = (245, 158, 11)
C_SAMPLE = (34, 197, 94)


@dataclass(frozen=True)
class Snap:
    weights: np.ndarray         # (GRID_H, GRID_W, 2)
    sample: np.ndarray          # (2,)
    bmu: Tuple[int, int]
    it: int


def seed_from_today() -> int:
    d = date.today()
    return (d.year * 10000 + d.month * 100 + d.day) % (2**32 - 1)


def fit_text(dr: ImageDraw.ImageDraw, text: str, max_w: int) -> str:
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


def wrap_text(dr: ImageDraw.ImageDraw, text: str, max_w: int) -> List[str]:
    words = text.split()
    if not words:
        return [""]
    lines: List[str] = []
    cur = words[0]
    for w in words[1:]:
        cand = f"{cur} {w}"
        if dr.textlength(cand, font=FONT) <= max_w:
            cur = cand
        else:
            if dr.textlength(cur, font=FONT) > max_w:
                cur = fit_text(dr, cur, max_w)
            lines.append(cur)
            cur = w
    if dr.textlength(cur, font=FONT) > max_w:
        cur = fit_text(dr, cur, max_w)
    lines.append(cur)
    return lines


# =========================
# Dataset
# =========================
def gen_dataset(rng: np.random.Generator, n: int) -> np.ndarray:
    """
    Três grupos 2D, mas com distribuição que ocupa melhor o plano.
    """
    n1 = n // 3
    n2 = n // 3
    n3 = n - n1 - n2

    c1 = rng.normal([-2.4, 1.6], [0.55, 0.45], size=(n1, 2))
    c2 = rng.normal([2.0, 1.1], [0.65, 0.50], size=(n2, 2))
    c3 = rng.normal([0.0, -2.0], [0.85, 0.60], size=(n3, 2))

    X = np.vstack([c1, c2, c3]).astype(float)
    return X


# =========================
# Normalização centralizada
# =========================
def normalize_to_rect(
    X: np.ndarray,
    x0: float,
    y0: float,
    w: float,
    h: float,
    pad: float = 24.0,
) -> Tuple[np.ndarray, Tuple[float, float, float, float, float, float, float]]:
    xmin, ymin = float(X[:, 0].min()), float(X[:, 1].min())
    xmax, ymax = float(X[:, 0].max()), float(X[:, 1].max())

    sx = (w - 2 * pad) / (xmax - xmin + 1e-9)
    sy = (h - 2 * pad) / (ymax - ymin + 1e-9)
    s = min(sx, sy)

    W = (xmax - xmin) * s
    H = (ymax - ymin) * s

    ox = x0 + (w - W) / 2.0
    oy = y0 + (h - H) / 2.0

    Yn = (X - np.array([xmin, ymin])) * s
    Yn[:, 0] = ox + Yn[:, 0]
    Yn[:, 1] = oy + (H - Yn[:, 1])

    return Yn, (xmin, ymin, xmax, ymax, s, ox, oy)


def to_plot(pt: np.ndarray, tf: Tuple[float, float, float, float, float, float, float]) -> Tuple[float, float]:
    xmin, ymin, xmax, ymax, s, ox, oy = tf
    H = (ymax - ymin) * s
    x = ox + (pt[0] - xmin) * s
    y = oy + (H - (pt[1] - ymin) * s)
    return float(x), float(y)


# =========================
# SOM
# =========================
def init_grid_weights(X: np.ndarray) -> np.ndarray:
    """
    Inicializa a grade como um retângulo regular cobrindo o bbox dos dados.
    Isso evita o 'novelo' inicial e torna a deformação compreensível.
    """
    xmin, ymin = X.min(axis=0)
    xmax, ymax = X.max(axis=0)

    gx = np.linspace(xmin, xmax, GRID_W)
    gy = np.linspace(ymin, ymax, GRID_H)

    W = np.zeros((GRID_H, GRID_W, 2), dtype=float)
    for i in range(GRID_H):
        for j in range(GRID_W):
            W[i, j, 0] = gx[j]
            W[i, j, 1] = gy[i]
    return W


def som_train(X: np.ndarray, rng: np.random.Generator) -> List[Snap]:
    weights = init_grid_weights(X)
    snaps: List[Snap] = []

    for it in range(ITERATIONS):
        x = X[rng.integers(0, X.shape[0])]

        dist = np.linalg.norm(weights - x, axis=2)
        bmu_idx = np.unravel_index(np.argmin(dist), dist.shape)

        lr = LR0 * np.exp(-it / ITERATIONS)
        sigma = SIGMA0 * np.exp(-it / ITERATIONS)

        for i in range(GRID_H):
            for j in range(GRID_W):
                d2 = (i - bmu_idx[0]) ** 2 + (j - bmu_idx[1]) ** 2
                h = np.exp(-d2 / (2.0 * sigma * sigma))
                weights[i, j] += lr * h * (x - weights[i, j])

        if it % SNAP_EVERY == 0:
            snaps.append(Snap(weights.copy(), x.copy(), bmu_idx, it))

    return snaps


def lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return (1.0 - t) * a + t * b


# =========================
# Render
# =========================
def main() -> None:
    Path("assets").mkdir(parents=True, exist_ok=True)
    out = Path("assets/som.gif")

    seed = seed_from_today()
    rng = np.random.default_rng(seed)

    X = gen_dataset(rng, DATA_N)

    inner_x, inner_y = CARD_PAD, CARD_PAD
    inner_w, inner_h = CANVAS_W - 2 * CARD_PAD, CANVAS_H - 2 * CARD_PAD

    plot_x0 = inner_x + LEFT_W + GAP
    plot_y0 = inner_y + 22
    plot_w = inner_w - LEFT_W - GAP - 18
    plot_h = inner_h - 44

    Xv, tf = normalize_to_rect(X, plot_x0, plot_y0, plot_w, plot_h, pad=24.0)

    snaps = som_train(X, rng)

    # barra baseada no avanço da iteração
    images: List[Image.Image] = []
    left_text_max_w = int(LEFT_W - 44)

    def draw_state(weights: np.ndarray, sample: np.ndarray, bmu: Tuple[int, int], it: int) -> Image.Image:
        im = Image.new("RGB", (CANVAS_W, CANVAS_H), BG)
        dr = ImageDraw.Draw(im)

        dr.rounded_rectangle(
            [inner_x, inner_y, inner_x + inner_w, inner_y + inner_h],
            radius=16, fill=CARD, outline=STROKE, width=2
        )

        dr.text((inner_x + 22, inner_y + 16), "Self Organizing Map (Kohonen)", fill=TEXT, font=FONT)

        meta = f"grid={GRID_W}x{GRID_H} | iteration={it}/{ITERATIONS} | seed={seed}"
        dr.text((inner_x + 22, inner_y + 36), fit_text(dr, meta, left_text_max_w), fill=MUTED, font=FONT)

        # barra de progresso
        bar_x, bar_y, bar_w, bar_h = inner_x + 22, inner_y + 56, LEFT_W - 44, 12
        dr.rounded_rectangle([bar_x, bar_y, bar_x + bar_w, bar_y + bar_h],
                             radius=8, fill=BAR_BG, outline=STROKE)
        prog = max(0.0, min(1.0, it / max(1, ITERATIONS)))
        dr.rounded_rectangle([bar_x, bar_y, bar_x + int(bar_w * prog), bar_y + bar_h],
                             radius=8, fill=BAR_FILL)

        y_cursor = inner_y + 78
        desc = "visualização: grade de neurônios se adapta à distribuição dos dados"
        for ln in wrap_text(dr, desc, left_text_max_w):
            dr.text((inner_x + 22, y_cursor), ln, fill=(120, 135, 155), font=FONT)
            y_cursor += 18
        y_cursor += 6

        dr.text((inner_x + 22, y_cursor), "linha azul: malha SOM", fill=(120, 135, 155), font=FONT)
        y_cursor += 18
        dr.text((inner_x + 22, y_cursor), "ponto verde: amostra atual", fill=(120, 135, 155), font=FONT)
        y_cursor += 18
        dr.text((inner_x + 22, y_cursor), "ponto laranja: BMU", fill=(120, 135, 155), font=FONT)

        # plot bg
        dr.rounded_rectangle(
            [plot_x0 - 12, plot_y0 - 12, plot_x0 + plot_w + 12, plot_y0 + plot_h + 12],
            radius=14, fill=(11, 18, 32), outline=STROKE
        )

        # dados
        for px, py in Xv:
            dr.ellipse([px - 2, py - 2, px + 2, py + 2], fill=C_DATA)

        # sample atual
        sx, sy = to_plot(sample, tf)
        dr.ellipse([sx - 4, sy - 4, sx + 4, sy + 4], fill=C_SAMPLE)

        # grade SOM
        for i in range(GRID_H):
            for j in range(GRID_W):
                x, y = to_plot(weights[i, j], tf)

                if j + 1 < GRID_W:
                    x2, y2 = to_plot(weights[i, j + 1], tf)
                    dr.line([x, y, x2, y2], fill=C_GRID, width=2)

                if i + 1 < GRID_H:
                    x2, y2 = to_plot(weights[i + 1, j], tf)
                    dr.line([x, y, x2, y2], fill=C_GRID, width=2)

        # BMU
        bi, bj = bmu
        bx, by = to_plot(weights[bi, bj], tf)
        dr.ellipse([bx - 6, by - 6, bx + 6, by + 6], fill=C_BMU)

        footer = f"BMU=({bi},{bj})"
        dr.text((inner_x + 22, inner_y + inner_h - 26), fit_text(dr, footer, int(inner_w - 44)),
                fill=(100, 116, 139), font=FONT)

        return im

    # segura o primeiro estado
    if snaps:
        first = draw_state(snaps[0].weights, snaps[0].sample, snaps[0].bmu, snaps[0].it)
        images.append(first)
        for _ in range(HOLD_FIRST):
            images.append(first)

    # tween entre snapshots
    for i in range(len(snaps) - 1):
        a = snaps[i]
        b = snaps[i + 1]

        for k in range(TWEEN_PER_SNAP):
            t = k / float(TWEEN_PER_SNAP)

            w = lerp(a.weights, b.weights, t)
            s = lerp(a.sample, b.sample, t)

            # BMU e it ficam do estado base para estabilidade visual
            images.append(draw_state(w, s, a.bmu, a.it))

    # último estado
    if snaps:
        last = draw_state(snaps[-1].weights, snaps[-1].sample, snaps[-1].bmu, snaps[-1].it)
        images.append(last)
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

    print(f"[ok] wrote {out} | frames={len(images)} | seed={seed}")


if __name__ == "__main__":
    main()