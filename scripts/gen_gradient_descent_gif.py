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

EPOCHS = 26
STEPS_PER_EPOCH = 18

# GD com momentum (clássico)
LR = 0.045
BETA = 0.92  # momentum

# Domínio theta = [x,y]
RANGE = 3.4
GRID_W = 240
GRID_H = 170
CONTOUR_LEVELS = 11

# Campo vetorial (setas)
ARROW_STEP = 18
ARROW_LEN = 10

# GIF
FRAME_MS = 55
HOLD_LAST = 14
TRAIL = 110

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

C_CONTOUR = (65, 86, 125)
C_FIELD = (80, 105, 150)
C_PATH = (56, 189, 248)
C_DOT = (229, 231, 235)
C_DOT_START = (245, 158, 11)
C_DOT_END = (34, 197, 94)


@dataclass(frozen=True)
class Snap:
    theta: np.ndarray
    loss: float
    epoch: int
    step_in_epoch: int
    step_total_epoch: int


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
# Quadrática convexa:
# f(theta)=1/2 theta^T A theta + c^T theta
# grad = A theta + c
# =========================
def make_quadratic(rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gera A SPD com condicionamento alto (vale estreito => trajetória mais longa/visível).
    """
    ang = float(rng.uniform(0.2, 1.2))
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]], dtype=float)

    # condicionamento alto: l2 >> l1
    l1 = float(rng.uniform(0.6, 1.2))
    l2 = float(rng.uniform(10.0, 18.0))
    D = np.diag([l1, l2])

    A = (R.T @ D @ R).astype(float)
    c = rng.normal(0.0, 1.1, size=(2,)).astype(float)
    return A, c


def loss_quad(theta: np.ndarray, A: np.ndarray, c: np.ndarray) -> float:
    return float(0.5 * theta.T @ A @ theta + c.T @ theta)


def grad_quad(theta: np.ndarray, A: np.ndarray, c: np.ndarray) -> np.ndarray:
    return (A @ theta + c).astype(float)


def collect_snaps(rng: np.random.Generator, A: np.ndarray, c: np.ndarray) -> List[Snap]:
    # start bem distante para caminho longo
    theta = rng.normal(0.0, 3.0, size=(2,)).astype(float)
    v = np.zeros((2,), dtype=float)

    snaps: List[Snap] = []
    for ep in range(1, EPOCHS + 1):
        for st in range(1, STEPS_PER_EPOCH + 1):
            g = grad_quad(theta, A, c)
            v = BETA * v + (1.0 - BETA) * g
            theta = theta - LR * v

            # mantém no domínio visual (sem “sumir”)
            theta = np.clip(theta, -RANGE * 1.25, RANGE * 1.25)

            snaps.append(
                Snap(
                    theta=theta.copy(),
                    loss=loss_quad(theta, A, c),
                    epoch=ep,
                    step_in_epoch=st,
                    step_total_epoch=STEPS_PER_EPOCH,
                )
            )
    return snaps


def main() -> None:
    Path("assets").mkdir(parents=True, exist_ok=True)
    out = Path("assets/gradient_descent.gif")

    seed = seed_from_today()
    rng = np.random.default_rng(seed)

    A, c = make_quadratic(rng)
    snaps = collect_snaps(rng, A, c)

    losses = np.array([s.loss for s in snaps], dtype=float)
    loss0 = float(losses[0])
    best_seen = float(losses.min())

    # Layout
    inner_x, inner_y = CARD_PAD, CARD_PAD
    inner_w, inner_h = CANVAS_W - 2 * CARD_PAD, CANVAS_H - 2 * CARD_PAD

    plot_x0 = inner_x + LEFT_W + GAP
    plot_y0 = inner_y + 22
    plot_w = inner_w - LEFT_W - GAP - 18
    plot_h = inner_h - 44

    px0, py0 = plot_x0 + 24, plot_y0 + 24
    pw, ph = plot_w - 48, plot_h - 48

    def to_px(theta: np.ndarray) -> Tuple[float, float]:
        x = (theta[0] + RANGE) / (2.0 * RANGE)
        y = (theta[1] + RANGE) / (2.0 * RANGE)
        px = px0 + x * pw
        py = py0 + (1.0 - y) * ph
        return float(px), float(py)

    # Campo de loss para heat + contornos
    xs = np.linspace(-RANGE, RANGE, GRID_W)
    ys = np.linspace(-RANGE, RANGE, GRID_H)
    field = np.zeros((GRID_H, GRID_W), dtype=float)
    for j, yy in enumerate(ys):
        for i, xx in enumerate(xs):
            field[j, i] = loss_quad(np.array([xx, yy], dtype=float), A, c)

    fmin, fmax = float(field.min()), float(field.max())
    if fmax - fmin < 1e-12:
        fmax = fmin + 1.0

    # log para contraste
    fld = np.log(field - fmin + 1e-6)
    fld_min, fld_max = float(fld.min()), float(fld.max())
    if fld_max - fld_min < 1e-12:
        fld_max = fld_min + 1.0
    fldn = (fld - fld_min) / (fld_max - fld_min)

    hm = np.zeros((GRID_H, GRID_W, 3), dtype=np.uint8)
    inten = (1.0 - fldn)
    base = (12 + inten * 28).astype(np.uint8)
    hm[..., 0] = base
    hm[..., 1] = (base + 7).clip(0, 255)
    hm[..., 2] = (base + 16).clip(0, 255)
    heat_img = Image.fromarray(hm, mode="RGB").resize((int(pw), int(ph)), resample=Image.Resampling.BILINEAR)

    levels = np.linspace(float(fld.min()), float(fld.max()), CONTOUR_LEVELS)

    left_text_max_w = int(LEFT_W - 44)

    images: List[Image.Image] = []
    path: List[Tuple[float, float]] = []

    # “best running” observado (para mostrar melhoria real durante o GIF)
    running_best = float("inf")

    for s in snaps:
        theta = s.theta
        loss = s.loss
        running_best = min(running_best, loss)

        im = Image.new("RGB", (CANVAS_W, CANVAS_H), BG)
        dr = ImageDraw.Draw(im)

        dr.rounded_rectangle(
            [inner_x, inner_y, inner_x + inner_w, inner_y + inner_h],
            radius=16, fill=CARD, outline=STROKE, width=2
        )

        dr.text((inner_x + 22, inner_y + 16), "Gradient Descent (Momentum Field)", fill=TEXT, font=FONT)
        meta = f"epoch={s.epoch} step={s.step_in_epoch}/{s.step_total_epoch} | seed={seed}"
        dr.text((inner_x + 22, inner_y + 36), fit_text(dr, meta, left_text_max_w), fill=MUTED, font=FONT)

        # Barra pelo running_best (monótona)
        bar_x, bar_y, bar_w, bar_h = inner_x + 22, inner_y + 56, LEFT_W - 44, 12
        dr.rounded_rectangle([bar_x, bar_y, bar_x + bar_w, bar_y + bar_h],
                             radius=8, fill=BAR_BG, outline=STROKE)

        denom = (loss0 - best_seen) if abs(loss0 - best_seen) > 1e-12 else 1.0
        prog = (loss0 - running_best) / denom
        prog = max(0.0, min(1.0, prog))
        dr.rounded_rectangle([bar_x, bar_y, bar_x + int(bar_w * prog), bar_y + bar_h],
                             radius=8, fill=BAR_FILL)

        y_cursor = inner_y + 78
        dr.text((inner_x + 22, y_cursor),
                fit_text(dr, f"loss: {loss:.4f}   best: {running_best:.4f}", left_text_max_w),
                fill=MUTED, font=FONT)
        y_cursor += 20
        dr.text((inner_x + 22, y_cursor),
                fit_text(dr, f"lr={LR:.3f}   beta={BETA:.2f}   steps/epoch={STEPS_PER_EPOCH}", left_text_max_w),
                fill=MUTED, font=FONT)
        y_cursor += 22

        exp = "visualização: campo vetorial do gradiente ∇f(x,y) com momentum (trajetória mais longa)"
        for ln in wrap_text(dr, exp, left_text_max_w):
            dr.text((inner_x + 22, y_cursor), ln, fill=(120, 135, 155), font=FONT)
            y_cursor += 18

        # Plot BG + heatmap
        dr.rounded_rectangle(
            [plot_x0 - 12, plot_y0 - 12, plot_x0 + plot_w + 12, plot_y0 + plot_h + 12],
            radius=14, fill=(11, 18, 32), outline=STROKE
        )
        im.paste(heat_img, (int(px0), int(py0)))

        # Contornos leves
        for lv in levels:
            for gy in range(GRID_H - 1):
                for gx in range(GRID_W - 1):
                    v00 = fld[gy, gx]
                    v10 = fld[gy, gx + 1]
                    v01 = fld[gy + 1, gx]
                    v11 = fld[gy + 1, gx + 1]
                    mn = min(v00, v10, v01, v11)
                    mx = max(v00, v10, v01, v11)
                    if mn <= lv <= mx:
                        x1 = px0 + (gx / (GRID_W - 1)) * pw
                        y1 = py0 + (gy / (GRID_H - 1)) * ph
                        dr.point((x1, y1), fill=C_CONTOUR)

        # Campo vetorial (-grad) com leve normalização por magnitude (mais visível perto do mínimo)
        for ay in range(int(py0 + 10), int(py0 + ph - 10), ARROW_STEP):
            for ax in range(int(px0 + 10), int(px0 + pw - 10), ARROW_STEP):
                tx = ((ax - px0) / pw) * (2.0 * RANGE) - RANGE
                ty = (1.0 - (ay - py0) / ph) * (2.0 * RANGE) - RANGE
                th = np.array([tx, ty], dtype=float)

                g = grad_quad(th, A, c)
                ng = float(np.linalg.norm(g)) + 1e-9
                dx, dy = (-g / ng)

                # aumenta um pouco o comprimento quando o gradiente é pequeno (apenas visual)
                # isso preserva direção e melhora legibilidade perto do mínimo.
                scale = 0.7 + 0.9 * (1.0 / (1.0 + ng))
                ex = ax + dx * (ARROW_LEN * scale)
                ey = ay - dy * (ARROW_LEN * scale)

                dr.line([ax, ay, ex, ey], fill=C_FIELD, width=1)
                dr.point((ex, ey), fill=C_FIELD)

        # Trajetória
        cur = to_px(theta)
        path.append(cur)
        trail = path[-TRAIL:]

        if len(trail) >= 2:
            for k in range(1, len(trail)):
                a = k / (len(trail) - 1)
                col = (
                    int(C_PATH[0] * a + 20 * (1 - a)),
                    int(C_PATH[1] * a + 30 * (1 - a)),
                    int(C_PATH[2] * a + 40 * (1 - a)),
                )
                dr.line([trail[k - 1], trail[k]], fill=col, width=3)

        sx, sy = path[0]
        dr.ellipse([sx - 5, sy - 5, sx + 5, sy + 5], fill=C_DOT_START)

        cx, cy = cur
        dr.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=C_DOT)

        dr.text((plot_x0 + 30, plot_y0 + 10), "y", fill=MUTED, font=FONT)
        dr.text((plot_x0 + plot_w - 36, plot_y0 + plot_h - 26), "x", fill=MUTED, font=FONT)

        footer = f"theta=[{theta[0]: .2f}, {theta[1]: .2f}]"
        dr.text((inner_x + 22, inner_y + inner_h - 26), fit_text(dr, footer, int(inner_w - 44)),
                fill=(100, 116, 139), font=FONT)

        images.append(im)

    # final marker + hold
    if images and path:
        im = images[-1].copy()
        dr = ImageDraw.Draw(im)
        ex, ey = path[-1]
        dr.ellipse([ex - 5, ey - 5, ex + 5, ey + 5], fill=C_DOT_END)
        images.append(im)
        for _ in range(HOLD_LAST):
            images.append(im)

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