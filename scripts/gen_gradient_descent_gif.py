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

# Regressão linear (MSE) em 2D, b FIXO para casar com o campo (visualização)
N = 240
NOISE_Y = 0.70
B_FIXED = 0.0

EPOCHS = 26
STEPS_PER_EPOCH = 18   # mais passos => movimento perceptível
LR = 0.06              # menor LR => trajetória mais longa/visível

# Campo/contornos em (w0,w1)
GRID_W = 220
GRID_H = 160
W_RANGE = 3.2
CONTOUR_LEVELS = 10

# GIF
TARGET_TOTAL_FRAMES = 320
MIN_TWEEN = 2
MAX_TWEEN = 8
FRAME_MS = 55
HOLD_LAST = 14

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

# Contour / path
C_CONTOUR = (65, 86, 125)
C_PATH = (56, 189, 248)
C_DOT = (229, 231, 235)
C_DOT_START = (245, 158, 11)
C_DOT_END = (34, 197, 94)


@dataclass(frozen=True)
class Snap:
    w: np.ndarray  # (2,)
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


def gen_regression_data(rng: np.random.Generator, n: int, noise_y: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    y = X w* + noise, com b=0 para o campo e para o treino.
    """
    w_star = rng.normal(0.0, 1.0, size=(2,))
    w_star = w_star / (np.linalg.norm(w_star) + 1e-12)

    X = rng.normal(0.0, 1.2, size=(n, 2))
    y = X @ w_star + rng.normal(0.0, noise_y, size=(n,))
    return X.astype(float), y.astype(float), w_star.astype(float)


def mse_loss(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float) -> float:
    e = (X @ w + b) - y
    return float(np.mean(e * e))


def gd_step_w_only(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float, lr: float) -> np.ndarray:
    """
    Atualiza apenas w (b fixo), para a trajetória casar com o campo 2D (w0,w1).
    grad_w = 2/n * X^T (Xw+b - y)
    """
    n = X.shape[0]
    e = (X @ w + b) - y
    grad_w = (2.0 / n) * (X.T @ e)
    return w - lr * grad_w


def collect_snaps(X: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> List[Snap]:
    # inicialização mais distante para trajetória visível
    w = rng.normal(0.0, 2.4, size=(2,))
    snaps: List[Snap] = []

    for ep in range(1, EPOCHS + 1):
        for st in range(1, STEPS_PER_EPOCH + 1):
            w = gd_step_w_only(X, y, w, B_FIXED, lr=LR)
            loss = mse_loss(X, y, w, B_FIXED)
            snaps.append(Snap(w=w.copy(), loss=float(loss), epoch=ep, step_in_epoch=st, step_total_epoch=STEPS_PER_EPOCH))
    return snaps


def allocate_tweens(weights: np.ndarray, target: int, min_tween: int, max_tween: int) -> List[int]:
    if weights.size == 0:
        return []
    base = max(1, target // weights.size)
    scaled = weights / (weights.mean() + 1e-12)

    tw = np.rint(base * scaled).astype(int)
    tw = np.clip(tw, min_tween, max_tween)

    def total(tw_: np.ndarray) -> int:
        return int(tw_.sum() + 1)

    cur = total(tw)
    if cur > target:
        order = np.argsort(weights)
        i = 0
        while cur > target and i < order.size * 250:
            j = order[i % order.size]
            if tw[j] > min_tween:
                tw[j] -= 1
                cur -= 1
            i += 1
    elif cur < target:
        order = np.argsort(-weights)
        i = 0
        while cur < target and i < order.size * 250:
            j = order[i % order.size]
            if tw[j] < max_tween:
                tw[j] += 1
                cur += 1
            i += 1
    return tw.tolist()


def main() -> None:
    Path("assets").mkdir(parents=True, exist_ok=True)
    out = Path("assets/gradient_descent.gif")

    seed = seed_from_today()
    rng = np.random.default_rng(seed)

    # Layout
    inner_x, inner_y = CARD_PAD, CARD_PAD
    inner_w, inner_h = CANVAS_W - 2 * CARD_PAD, CANVAS_H - 2 * CARD_PAD

    plot_x0 = inner_x + LEFT_W + GAP
    plot_y0 = inner_y + 22
    plot_w = inner_w - LEFT_W - GAP - 18
    plot_h = inner_h - 44

    # Dados
    X, y, w_star = gen_regression_data(rng, N, NOISE_Y)

    # Snaps
    snaps = collect_snaps(X, y, rng)
    losses = np.array([s.loss for s in snaps], dtype=float)
    loss0 = float(losses[0])
    lossN = float(losses.min())

    # Campo MSE em (w0,w1) com b fixo
    w0s = np.linspace(-W_RANGE, W_RANGE, GRID_W)
    w1s = np.linspace(-W_RANGE, W_RANGE, GRID_H)
    field = np.zeros((GRID_H, GRID_W), dtype=float)
    for j, w1 in enumerate(w1s):
        for i, w0 in enumerate(w0s):
            field[j, i] = mse_loss(X, y, np.array([w0, w1], dtype=float), B_FIXED)

    fmin, fmax = float(field.min()), float(field.max())
    if fmax - fmin < 1e-12:
        fmax = fmin + 1.0

    # Pacing por variação real de w (se delta w ~ 0, tween menor)
    dw = np.array([np.linalg.norm(snaps[i + 1].w - snaps[i].w) for i in range(len(snaps) - 1)], dtype=float)
    weights = dw + 1e-6
    tweens = allocate_tweens(weights, TARGET_TOTAL_FRAMES, MIN_TWEEN, MAX_TWEEN)

    left_text_max_w = int(LEFT_W - 44)

    # Mapeamento w->pixel
    px0, py0 = plot_x0 + 24, plot_y0 + 24
    pw, ph = plot_w - 48, plot_h - 48

    def w_to_px(w: np.ndarray) -> Tuple[float, float]:
        x = (w[0] + W_RANGE) / (2.0 * W_RANGE)
        yv = (w[1] + W_RANGE) / (2.0 * W_RANGE)
        px = px0 + x * pw
        py = py0 + (1.0 - yv) * ph
        return float(px), float(py)

    # Heatmap contínuo (grayscale azulado)
    # Normaliza log para realçar contornos e evitar “parado”
    fld = np.log(field - fmin + 1e-6)
    fld_min, fld_max = float(fld.min()), float(fld.max())
    if fld_max - fld_min < 1e-12:
        fld_max = fld_min + 1.0
    fldn = (fld - fld_min) / (fld_max - fld_min)

    hm = np.zeros((GRID_H, GRID_W, 3), dtype=np.uint8)
    # intensidade invertida: menor loss => mais claro
    inten = (1.0 - fldn)
    base = (12 + inten * 26).astype(np.uint8)
    hm[..., 0] = base
    hm[..., 1] = (base + 6).clip(0, 255)
    hm[..., 2] = (base + 14).clip(0, 255)
    heat_img = Image.fromarray(hm, mode="RGB").resize((int(pw), int(ph)), resample=Image.Resampling.BILINEAR)

    # Contornos (níveis no domínio log)
    levels = np.linspace(float(fld.min()), float(fld.max()), CONTOUR_LEVELS)

    images: List[Image.Image] = []
    path_pts: List[Tuple[float, float]] = []
    TRAIL = 80  # rastro

    for i in range(len(snaps) - 1):
        s0 = snaps[i]
        s1 = snaps[i + 1]
        nsub = tweens[i] if i < len(tweens) else MIN_TWEEN

        for sf in range(nsub):
            t = sf / float(nsub)
            w = (1.0 - t) * s0.w + t * s1.w
            loss = float((1.0 - t) * s0.loss + t * s1.loss)

            im = Image.new("RGB", (CANVAS_W, CANVAS_H), BG)
            dr = ImageDraw.Draw(im)

            dr.rounded_rectangle(
                [inner_x, inner_y, inner_x + inner_w, inner_y + inner_h],
                radius=16, fill=CARD, outline=STROKE, width=2
            )

            dr.text((inner_x + 22, inner_y + 16), "Gradient Descent (MSE Contours)", fill=TEXT, font=FONT)
            meta = f"epoch={s0.epoch} step={s0.step_in_epoch}/{s0.step_total_epoch} | seed={seed}"
            dr.text((inner_x + 22, inner_y + 36), fit_text(dr, meta, left_text_max_w), fill=MUTED, font=FONT)

            # Barra (loss ↓ => progresso ↑)
            bar_x, bar_y, bar_w, bar_h = inner_x + 22, inner_y + 56, LEFT_W - 44, 12
            dr.rounded_rectangle([bar_x, bar_y, bar_x + bar_w, bar_y + bar_h],
                                 radius=8, fill=BAR_BG, outline=STROKE)
            denom = (loss0 - lossN) if abs(loss0 - lossN) > 1e-12 else 1.0
            prog = (loss0 - loss) / denom
            prog = max(0.0, min(1.0, prog))
            dr.rounded_rectangle([bar_x, bar_y, bar_x + int(bar_w * prog), bar_y + bar_h],
                                 radius=8, fill=BAR_FILL)

            y_cursor = inner_y + 78
            dr.text((inner_x + 22, y_cursor), fit_text(dr, f"loss: {loss:.4f}   best: {lossN:.4f}", left_text_max_w),
                    fill=MUTED, font=FONT)
            y_cursor += 20
            dr.text((inner_x + 22, y_cursor), fit_text(dr, f"lr={LR:.3f}   steps/epoch={STEPS_PER_EPOCH}   N={N}", left_text_max_w),
                    fill=MUTED, font=FONT)
            y_cursor += 22

            explain = "visualização: MSE(w0,w1) com b fixo; ponto = parâmetros (w0,w1)"
            for ln in wrap_text(dr, explain, left_text_max_w):
                dr.text((inner_x + 22, y_cursor), ln, fill=(120, 135, 155), font=FONT)
                y_cursor += 18

            # Plot BG + heatmap
            dr.rounded_rectangle(
                [plot_x0 - 12, plot_y0 - 12, plot_x0 + plot_w + 12, plot_y0 + plot_h + 12],
                radius=14, fill=(11, 18, 32), outline=STROKE
            )
            im.paste(heat_img, (int(px0), int(py0)))

            # Contornos leves por “cruzamento” (no grid original, domínio log)
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
                            dr.rectangle([x1, y1, x1 + 1, y1 + 1], outline=C_CONTOUR)

            # Trajetória com rastro (fade)
            cur_pt = w_to_px(w)
            path_pts.append(cur_pt)
            trail = path_pts[-TRAIL:]

            if len(trail) >= 2:
                # desenha segmentos com “fade” via mistura de cores (sem alpha)
                for k in range(1, len(trail)):
                    a = (k / (len(trail) - 1))
                    col = (
                        int(C_PATH[0] * a + 20 * (1 - a)),
                        int(C_PATH[1] * a + 30 * (1 - a)),
                        int(C_PATH[2] * a + 40 * (1 - a)),
                    )
                    dr.line([trail[k - 1], trail[k]], fill=col, width=3)

            # marcadores
            sx, sy = path_pts[0]
            dr.ellipse([sx - 5, sy - 5, sx + 5, sy + 5], fill=C_DOT_START)

            cx, cy = cur_pt
            dr.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=C_DOT)

            # eixos
            dr.text((plot_x0 + 30, plot_y0 + 10), "w1", fill=MUTED, font=FONT)
            dr.text((plot_x0 + plot_w - 52, plot_y0 + plot_h - 26), "w0", fill=MUTED, font=FONT)

            footer = f"w=[{w[0]: .2f}, {w[1]: .2f}]  b={B_FIXED: .2f}"
            dr.text((inner_x + 22, inner_y + inner_h - 26), fit_text(dr, footer, int(inner_w - 44)),
                    fill=(100, 116, 139), font=FONT)

            images.append(im)

    # marca final e hold
    if images and path_pts:
        im = images[-1].copy()
        dr = ImageDraw.Draw(im)
        ex, ey = path_pts[-1]
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