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

N_TOTAL = 260

EPOCHS = 28
STEPS_PER_EPOCH = 10
LR = 0.028
C = 1.4
REG = 1.0

TARGET_TOTAL_FRAMES = 280
MIN_TWEEN = 2
MAX_TWEEN = 7
FRAME_MS = 55
HOLD_LAST = 14

# Começa o GIF depois de um pequeno aquecimento
WARMUP_EPOCHS = 0

POINT_R = 2
LINE_W = 3

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

C_NEG = (239, 68, 68)
C_POS = (37, 99, 235)
C_HYPER = (56, 189, 248)
C_MARGIN = (245, 158, 11)
C_SUPPORT = (34, 197, 94)


@dataclass(frozen=True)
class Snap:
    w: np.ndarray
    b: float
    obj: float
    best_obj: float
    acc: float
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
# Dataset revisado
# =========================
def gen_dataset(rng: np.random.Generator, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Dataset quase separável, com poucos pontos próximos da margem
    e poucos outliers, melhor para visualização de SVM.
    """
    n_pos = n // 2
    n_neg = n - n_pos

    X_pos = np.column_stack([
        rng.normal(1.25, 0.70, size=n_pos),
        rng.normal(0.95, 0.62, size=n_pos),
    ])
    X_neg = np.column_stack([
        rng.normal(-1.20, 0.70, size=n_neg),
        rng.normal(-0.90, 0.62, size=n_neg),
    ])

    X = np.vstack([X_pos, X_neg]).astype(float)
    y = np.hstack([np.ones(n_pos, dtype=int), -np.ones(n_neg, dtype=int)])

    ang = float(rng.uniform(-0.35, 0.35))
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]], dtype=float)
    X = X @ R.T

    bridge_n = max(8, n // 18)
    bridge = np.column_stack([
        rng.normal(0.0, 0.35, size=bridge_n),
        rng.normal(0.0, 0.65, size=bridge_n),
    ])
    bridge_y = rng.choice([-1, 1], size=bridge_n, replace=True)

    X = np.vstack([X, bridge]).astype(float)
    y = np.hstack([y, bridge_y]).astype(int)

    out_n = 6
    out = rng.normal(0.0, 2.2, size=(out_n, 2))
    out_y = rng.choice([-1, 1], size=out_n, replace=True)
    X = np.vstack([X, out]).astype(float)
    y = np.hstack([y, out_y]).astype(int)

    idx = rng.permutation(X.shape[0])
    return X[idx], y[idx]


# =========================
# Geometria do plot
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


def to_plot(
    pt: np.ndarray,
    tf: Tuple[float, float, float, float, float, float, float],
) -> Tuple[float, float]:
    xmin, ymin, xmax, ymax, s, ox, oy = tf
    H = (ymax - ymin) * s
    x = ox + (pt[0] - xmin) * s
    y = oy + (H - (pt[1] - ymin) * s)
    return float(x), float(y)


def line_points_for_value(
    w: np.ndarray,
    b: float,
    value: float,
    rect: Tuple[float, float, float, float],
) -> Tuple[np.ndarray, np.ndarray]:
    xmin, ymin, xmax, ymax = rect
    w0, w1 = float(w[0]), float(w[1])
    eps = 1e-9
    c = b - value

    if abs(w1) > eps:
        y_at_xmin = -(w0 * xmin + c) / w1
        y_at_xmax = -(w0 * xmax + c) / w1
        return np.array([xmin, y_at_xmin]), np.array([xmax, y_at_xmax])

    if abs(w0) > eps:
        x0 = -c / w0
        return np.array([x0, ymin]), np.array([x0, ymax])

    return np.array([xmin, ymin]), np.array([xmax, ymax])


def lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return (1.0 - t) * a + t * b


# =========================
# SVM linear aproximada
# =========================
def decision_values(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    return X @ w + b


def predict(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    return np.where(decision_values(X, w, b) >= 0.0, 1, -1).astype(int)


def svm_objective(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float, c_penalty: float, reg: float) -> float:
    margin = 1.0 - y * decision_values(X, w, b)
    hinge = np.maximum(0.0, margin)
    return float(0.5 * reg * np.dot(w, w) + c_penalty * np.mean(hinge))


def accuracy(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float) -> float:
    return float(np.mean(predict(X, w, b) == y))


def grad_step(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float, lr: float, c_penalty: float, reg: float) -> Tuple[np.ndarray, float]:
    dec = decision_values(X, w, b)
    active = (1.0 - y * dec) > 0.0

    grad_w = reg * w
    grad_b = 0.0

    if np.any(active):
        ya = y[active][:, None]
        Xa = X[active]
        grad_w += c_penalty * np.mean(-ya * Xa, axis=0)
        grad_b += c_penalty * float(np.mean(-y[active]))

    w2 = w - lr * grad_w
    b2 = b - lr * grad_b
    return w2.astype(float), float(b2)


def collect_snaps(X: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> List[Snap]:
    w = rng.normal(0.0, 1.2, size=(2,)).astype(float)
    b = float(rng.normal(0.0, 0.6))

    snaps: List[Snap] = []
    best_obj = float("inf")

    for ep in range(1, EPOCHS + 1):
        for st in range(1, STEPS_PER_EPOCH + 1):
            w, b = grad_step(X, y, w, b, lr=LR, c_penalty=C, reg=REG)
            obj = svm_objective(X, y, w, b, C, REG)
            acc = accuracy(X, y, w, b)

            if obj < best_obj:
                best_obj = obj

            snaps.append(
                Snap(
                    w=w.copy(),
                    b=b,
                    obj=obj,
                    best_obj=best_obj,
                    acc=acc,
                    epoch=ep,
                    step_in_epoch=st,
                    step_total_epoch=STEPS_PER_EPOCH,
                )
            )
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
        while cur > target and i < order.size * 200:
            j = order[i % order.size]
            if tw[j] > min_tween:
                tw[j] -= 1
                cur -= 1
            i += 1
    elif cur < target:
        order = np.argsort(-weights)
        i = 0
        while cur < target and i < order.size * 200:
            j = order[i % order.size]
            if tw[j] < max_tween:
                tw[j] += 1
                cur += 1
            i += 1
    return tw.tolist()


# =========================
# Main
# =========================
def main() -> None:
    Path("assets").mkdir(parents=True, exist_ok=True)
    out = Path("assets/svm.gif")

    seed = seed_from_today()
    rng = np.random.default_rng(seed)

    X, y = gen_dataset(rng, N_TOTAL)

    inner_x, inner_y = CARD_PAD, CARD_PAD
    inner_w, inner_h = CANVAS_W - 2 * CARD_PAD, CANVAS_H - 2 * CARD_PAD

    plot_x0 = inner_x + LEFT_W + GAP
    plot_y0 = inner_y + 22
    plot_w = inner_w - LEFT_W - GAP - 18
    plot_h = inner_h - 44

    Xv, tf = normalize_to_rect(X, plot_x0, plot_y0, plot_w, plot_h, pad=24.0)
    xmin, ymin, xmax, ymax, _s, _ox, _oy = tf
    rect = (xmin, ymin, xmax, ymax)

    snaps_all = collect_snaps(X, y, rng)

    snaps = [s for s in snaps_all if s.epoch >= WARMUP_EPOCHS]
    if len(snaps) < 2:
        snaps = snaps_all

    objs = np.array([s.best_obj for s in snaps], dtype=float)
    obj0 = float(objs[0])
    objN = float(objs.min())

    weights = []
    for i in range(len(snaps) - 1):
        dw = float(np.linalg.norm(snaps[i + 1].w - snaps[i].w))
        dobj = float(abs(snaps[i + 1].obj - snaps[i].obj))
        weights.append(dw + 0.8 * dobj + 1e-6)
    weights = np.array(weights, dtype=float)
    tweens = allocate_tweens(weights, TARGET_TOTAL_FRAMES, MIN_TWEEN, MAX_TWEEN)

    left_text_max_w = int(LEFT_W - 44)
    images: List[Image.Image] = []

    for i in range(len(snaps) - 1):
        a = snaps[i]
        b = snaps[i + 1]
        nsub = tweens[i] if i < len(tweens) else MIN_TWEEN

        for k in range(nsub):
            t = k / float(nsub)

            w = lerp(a.w, b.w, t)
            bb = float((1.0 - t) * a.b + t * b.b)
            obj = float((1.0 - t) * a.obj + t * b.obj)
            best_obj = float(min(a.best_obj, b.best_obj))
            acc = float((1.0 - t) * a.acc + t * b.acc)

            im = Image.new("RGB", (CANVAS_W, CANVAS_H), BG)
            dr = ImageDraw.Draw(im)

            dr.rounded_rectangle(
                [inner_x, inner_y, inner_x + inner_w, inner_y + inner_h],
                radius=16, fill=CARD, outline=STROKE, width=2
            )

            dr.text((inner_x + 22, inner_y + 16), "SVM Linear (Maximum Margin)", fill=TEXT, font=FONT)

            meta = f"epoch={a.epoch} step={a.step_in_epoch}/{a.step_total_epoch} | seed={seed}"
            dr.text((inner_x + 22, inner_y + 36), fit_text(dr, meta, left_text_max_w), fill=MUTED, font=FONT)

            bar_x, bar_y, bar_w, bar_h = inner_x + 22, inner_y + 56, LEFT_W - 44, 12
            dr.rounded_rectangle([bar_x, bar_y, bar_x + bar_w, bar_y + bar_h],
                                 radius=8, fill=BAR_BG, outline=STROKE)
            denom = (obj0 - objN) if abs(obj0 - objN) > 1e-12 else 1.0
            prog = (obj0 - best_obj) / denom
            prog = max(0.0, min(1.0, prog))
            dr.rounded_rectangle([bar_x, bar_y, bar_x + int(bar_w * prog), bar_y + bar_h],
                                 radius=8, fill=BAR_FILL)

            y_cursor = inner_y + 78
            dr.text((inner_x + 22, y_cursor),
                    fit_text(dr, f"objective: {obj:.4f}   best: {best_obj:.4f}   ac: {acc*100:5.1f}%", left_text_max_w),
                    fill=MUTED, font=FONT)
            y_cursor += 20

            dr.text((inner_x + 22, y_cursor),
                    fit_text(dr, f"C={C:.2f}   lr={LR:.3f}   reg={REG:.2f}", left_text_max_w),
                    fill=MUTED, font=FONT)
            y_cursor += 22

            desc = "visualização: hiperplano separador, margens ±1 e vetores de suporte aproximados"
            for ln in wrap_text(dr, desc, left_text_max_w):
                dr.text((inner_x + 22, y_cursor), ln, fill=(120, 135, 155), font=FONT)
                y_cursor += 18
            y_cursor += 6

            dr.text((inner_x + 22, y_cursor), "linha azul: hiperplano", fill=(120, 135, 155), font=FONT)
            y_cursor += 18
            dr.text((inner_x + 22, y_cursor), "linhas amarelas: margens", fill=(120, 135, 155), font=FONT)
            y_cursor += 18
            dr.text((inner_x + 22, y_cursor), "anel verde: support vectors", fill=(120, 135, 155), font=FONT)

            # plot
            dr.rounded_rectangle(
                [plot_x0 - 12, plot_y0 - 12, plot_x0 + plot_w + 12, plot_y0 + plot_h + 12],
                radius=14, fill=(11, 18, 32), outline=STROKE
            )

            dec = decision_values(X, w, bb)
            margin_val = y * dec

            # critério restritivo para support vectors
            sv_mask = np.abs(margin_val - 1.0) <= 0.07

            # hiperplano
            p1, p2 = line_points_for_value(w, bb, 0.0, rect)
            ax, ay = to_plot(p1, tf)
            bx, by = to_plot(p2, tf)
            dr.line([ax, ay, bx, by], fill=C_HYPER, width=LINE_W)

            # margens
            if np.linalg.norm(w) > 1e-9:
                m1a, m1b = line_points_for_value(w, bb, 1.0, rect)
                m2a, m2b = line_points_for_value(w, bb, -1.0, rect)

                x1, y1 = to_plot(m1a, tf)
                x2, y2 = to_plot(m1b, tf)
                dr.line([x1, y1, x2, y2], fill=C_MARGIN, width=2)

                x3, y3 = to_plot(m2a, tf)
                x4, y4 = to_plot(m2b, tf)
                dr.line([x3, y3, x4, y4], fill=C_MARGIN, width=2)

            # pontos por cima das linhas
            r = POINT_R
            for idx, (px, py) in enumerate(Xv):
                col = C_POS if y[idx] == 1 else C_NEG
                dr.ellipse([px - r, py - r, px + r, py + r], fill=col)

                if sv_mask[idx]:
                    dr.ellipse([px - (r + 2.5), py - (r + 2.5), px + (r + 2.5), py + (r + 2.5)],
                               outline=C_SUPPORT, width=1)

            footer = f"w=[{w[0]: .2f}, {w[1]: .2f}]  b={bb: .2f}"
            dr.text((inner_x + 22, inner_y + inner_h - 26), fit_text(dr, footer, int(inner_w - 44)),
                    fill=(100, 116, 139), font=FONT)

            images.append(im)

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

    print(f"[ok] wrote {out} | frames={len(images)} | seed={seed}")


if __name__ == "__main__":
    main()