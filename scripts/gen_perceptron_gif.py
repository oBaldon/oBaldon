#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw


# =========================
# Configuração
# =========================
CANVAS_W, CANVAS_H = 980, 360

N_TOTAL = 360          # total de pontos (balanceado)
NOISE = 0.95
SEP = 1.35             # separação moderada
HARD_FRAC = 0.35       # fração de pontos perto da fronteira, mas com rótulo consistente
MARGIN = 0.22          # quão perto da fronteira os "hard points" ficam (menor = mais difícil, ainda separável)

EPOCHS = 22
LR = 0.07

SNAP_EVERY_UPDATES = 10

# GIF pacing
TARGET_TOTAL_FRAMES = 320
MIN_TWEEN = 3
MAX_TWEEN = 16
FRAME_MS = 55
HOLD_LAST = 12

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

# Classes
C_NEG = (239, 68, 68)     # -1
C_POS = (37, 99, 235)     # +1
C_LINE = (34, 197, 94)    # fronteira


@dataclass(frozen=True)
class Snap:
    w: np.ndarray
    b: float
    acc: float
    step: int
    epoch: int


def seed_from_today() -> int:
    d = date.today()
    return (d.year * 10000 + d.month * 100 + d.day) % (2**32 - 1)


def sign01(a: np.ndarray) -> np.ndarray:
    y = np.where(a >= 0.0, 1, -1)
    return y.astype(int)


def gen_dataset_separable(rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gera dataset separável com:
    - dois clusters gaussianos moderadamente separados
    - pontos "difíceis" próximos à fronteira, com rótulo consistente via hiperplano "professor"
    """
    # hiperplano professor (fixo e simples): w* x + b = 0
    w_star = np.array([1.0, -0.7], dtype=float)
    b_star = 0.10

    n_hard = int(N_TOTAL * HARD_FRAC)
    n_easy = N_TOTAL - n_hard
    n_easy_half = n_easy // 2

    # easy points: dois gaussianos separados
    m0 = np.array([-SEP, -0.7])
    m1 = np.array([ SEP,  0.8])
    cov = np.array([[NOISE, 0.18], [0.18, NOISE]])

    X0 = rng.multivariate_normal(m0, cov, size=n_easy_half)
    X1 = rng.multivariate_normal(m1, cov, size=n_easy - n_easy_half)

    X_easy = np.vstack([X0, X1])
    y_easy = np.hstack([-np.ones(X0.shape[0], dtype=int), np.ones(X1.shape[0], dtype=int)])

    # hard points: amostras perto da fronteira do professor, mas com rótulo coerente
    hard = []
    hard_y = []
    tries = 0
    while len(hard) < n_hard and tries < n_hard * 80:
        tries += 1
        x = rng.normal(loc=0.0, scale=NOISE * 0.9, size=(2,))
        # força proximidade da fronteira: |w*x + b| pequeno
        score = float(np.dot(w_star, x) + b_star)
        if abs(score) <= MARGIN:
            # empurra levemente para um lado aleatório mantendo separável
            side = rng.choice([-1.0, 1.0])
            # deslocamento pequeno na direção normal do hiperplano
            nrm = w_star / (np.linalg.norm(w_star) + 1e-9)
            x2 = x + side * (MARGIN * 1.25) * nrm
            y2 = 1 if (np.dot(w_star, x2) + b_star) >= 0 else -1
            hard.append(x2)
            hard_y.append(y2)

    X_hard = np.array(hard, dtype=float)
    y_hard = np.array(hard_y, dtype=int)

    X = np.vstack([X_easy, X_hard]).astype(float)
    y = np.hstack([y_easy, y_hard]).astype(int)

    # embaralha
    idx = rng.permutation(X.shape[0])
    return X[idx], y[idx]


def accuracy(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float) -> float:
    pred = sign01(X @ w + b)
    return float((pred == y).mean())


def perceptron_snaps(X: np.ndarray, y: np.ndarray, epochs: int, lr: float, rng: np.random.Generator) -> List[Snap]:
    # inicialização deliberadamente fraca (próxima de zero)
    w = rng.normal(0.0, 0.06, size=(2,)).astype(float)
    b = float(rng.normal(0.0, 0.06))

    snaps: List[Snap] = []
    step = 0
    updates = 0

    snaps.append(Snap(w=w.copy(), b=b, acc=accuracy(X, y, w, b), step=step, epoch=0))

    for ep in range(1, epochs + 1):
        order = rng.permutation(X.shape[0])
        for i in order:
            step += 1
            xi = X[i]
            yi = int(y[i])
            a = float(np.dot(w, xi) + b)
            yhat = 1 if a >= 0 else -1

            if yhat != yi:
                w = w + lr * yi * xi
                b = b + lr * yi
                updates += 1

                if updates % SNAP_EVERY_UPDATES == 0:
                    snaps.append(Snap(w=w.copy(), b=b, acc=accuracy(X, y, w, b), step=step, epoch=ep))

        snaps.append(Snap(w=w.copy(), b=b, acc=accuracy(X, y, w, b), step=step, epoch=ep))

        # early stop se chegou em 100% (mas ainda mantém alguns snaps finais via hold do gif)
        if snaps[-1].acc >= 0.999:
            break

    # remove duplicados (w,b iguais)
    cleaned = [snaps[0]]
    for s in snaps[1:]:
        p = cleaned[-1]
        if np.linalg.norm(s.w - p.w) > 1e-10 or abs(s.b - p.b) > 1e-10:
            cleaned.append(s)
    return cleaned


def normalize_to_rect(X: np.ndarray, x0: float, y0: float, w: float, h: float, pad: float = 24.0) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    xmin, ymin = float(X[:, 0].min()), float(X[:, 1].min())
    xmax, ymax = float(X[:, 0].max()), float(X[:, 1].max())

    sx = (w - 2 * pad) / (xmax - xmin + 1e-9)
    sy = (h - 2 * pad) / (ymax - ymin + 1e-9)
    s = min(sx, sy)

    Yn = (X - np.array([xmin, ymin])) * s
    Yn[:, 0] = x0 + pad + Yn[:, 0]
    Yn[:, 1] = y0 + h - pad - Yn[:, 1]
    return Yn, (xmin, ymin, xmax, ymax)


def to_plot(pt: np.ndarray, rect: Tuple[float, float, float, float], plot: Tuple[float, float, float, float], pad: float = 24.0) -> Tuple[float, float]:
    xmin, ymin, xmax, ymax = rect
    x0, y0, w, h = plot

    sx = (w - 2 * pad) / (xmax - xmin + 1e-9)
    sy = (h - 2 * pad) / (ymax - ymin + 1e-9)
    s = min(sx, sy)

    x = x0 + pad + (pt[0] - xmin) * s
    y = y0 + h - pad - (pt[1] - ymin) * s
    return float(x), float(y)


def lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return (1.0 - t) * a + t * b


def allocate_tweens(weights: np.ndarray, target_total_frames: int, min_tween: int, max_tween: int) -> List[int]:
    if weights.size == 0:
        return []
    base = max(1, target_total_frames // weights.size)
    scaled = weights / (weights.mean() + 1e-12)
    tw = np.rint(base * scaled).astype(int)
    tw = np.clip(tw, min_tween, max_tween)

    def total(tw_: np.ndarray) -> int:
        return int(tw_.sum() + 1)

    cur = total(tw)
    if cur > target_total_frames:
        order = np.argsort(weights)
        i = 0
        while cur > target_total_frames and i < order.size * 100:
            j = order[i % order.size]
            if tw[j] > min_tween:
                tw[j] -= 1
                cur -= 1
            i += 1
    elif cur < target_total_frames:
        order = np.argsort(-weights)
        i = 0
        while cur < target_total_frames and i < order.size * 100:
            j = order[i % order.size]
            if tw[j] < max_tween:
                tw[j] += 1
                cur += 1
            i += 1
    return tw.tolist()


def decision_line_points(w: np.ndarray, b: float, rect: Tuple[float, float, float, float]) -> Tuple[np.ndarray, np.ndarray]:
    xmin, ymin, xmax, ymax = rect
    w0, w1 = float(w[0]), float(w[1])
    eps = 1e-9
    if abs(w1) > eps:
        y0 = -(w0 * xmin + b) / w1
        y1 = -(w0 * xmax + b) / w1
        return np.array([xmin, y0], dtype=float), np.array([xmax, y1], dtype=float)
    if abs(w0) > eps:
        x = -b / w0
        return np.array([x, ymin], dtype=float), np.array([x, ymax], dtype=float)
    return np.array([xmin, ymin], dtype=float), np.array([xmax, ymax], dtype=float)


def main() -> None:
    Path("assets").mkdir(parents=True, exist_ok=True)
    out = Path("assets/perceptron.gif")

    seed = seed_from_today()
    rng = np.random.default_rng(seed)

    inner_x, inner_y = CARD_PAD, CARD_PAD
    inner_w, inner_h = CANVAS_W - 2 * CARD_PAD, CANVAS_H - 2 * CARD_PAD

    plot_x0 = inner_x + LEFT_W + GAP
    plot_y0 = inner_y + 22
    plot_w = inner_w - LEFT_W - GAP - 18
    plot_h = inner_h - 44
    plot = (plot_x0, plot_y0, plot_w, plot_h)

    X, y = gen_dataset_separable(rng)
    Xv, rect = normalize_to_rect(X, plot_x0, plot_y0, plot_w, plot_h, pad=24.0)

    snaps = perceptron_snaps(X, y, EPOCHS, LR, rng)

    weights = []
    for i in range(len(snaps) - 1):
        dw = float(np.linalg.norm(snaps[i + 1].w - snaps[i].w))
        db = float(abs(snaps[i + 1].b - snaps[i].b))
        weights.append(dw + 0.35 * db + 1e-6)
    weights = np.array(weights, dtype=float)
    tweens = allocate_tweens(weights, TARGET_TOTAL_FRAMES, MIN_TWEEN, MAX_TWEEN)

    images: List[Image.Image] = []

    for i in range(len(snaps) - 1):
        s0, s1 = snaps[i], snaps[i + 1]
        nsub = tweens[i] if i < len(tweens) else MIN_TWEEN

        for sf in range(nsub):
            t = sf / float(nsub)
            w = lerp(s0.w, s1.w, t)
            b = float((1.0 - t) * s0.b + t * s1.b)
            acc = float((1.0 - t) * s0.acc + t * s1.acc)

            im = Image.new("RGB", (CANVAS_W, CANVAS_H), BG)
            dr = ImageDraw.Draw(im)

            dr.rounded_rectangle([inner_x, inner_y, inner_x + inner_w, inner_y + inner_h],
                                 radius=16, fill=CARD, outline=STROKE, width=2)

            dr.text((inner_x + 22, inner_y + 16), "Perceptron", fill=TEXT)
            dr.text((inner_x + 140, inner_y + 20),
                    f"epoch={s0.epoch} | step={s0.step} | seed={seed}", fill=MUTED)

            # barra acurácia (agora deve subir e estabilizar em 100%)
            bar_x, bar_y, bar_w, bar_h = inner_x + 22, inner_y + 50, LEFT_W - 44, 12
            dr.rounded_rectangle([bar_x, bar_y, bar_x + bar_w, bar_y + bar_h],
                                 radius=8, fill=BAR_BG, outline=STROKE)
            prog = max(0.0, min(1.0, acc))
            dr.rounded_rectangle([bar_x, bar_y, bar_x + int(bar_w * prog), bar_y + bar_h],
                                 radius=8, fill=BAR_FILL)
            dr.text((inner_x + 22, inner_y + 70), f"accuracy: {acc*100:5.1f}%", fill=MUTED)

            # legenda
            lx, ly = inner_x + 22, inner_y + 105
            dr.rectangle([lx, ly - 10, lx + 12, ly + 2], fill=C_POS)
            dr.text((lx + 18, ly - 12), "y=+1", fill=MUTED)
            dr.rectangle([lx, ly + 18 - 10, lx + 12, ly + 18 + 2], fill=C_NEG)
            dr.text((lx + 18, ly + 18 - 12), "y=-1", fill=MUTED)

            # plot
            dr.rounded_rectangle([plot_x0 - 12, plot_y0 - 12, plot_x0 + plot_w + 12, plot_y0 + plot_h + 12],
                                 radius=14, fill=(11, 18, 32), outline=STROKE)

            r = 3
            for p_idx, (px, py) in enumerate(Xv):
                dr.ellipse([px - r, py - r, px + r, py + r], fill=(C_POS if y[p_idx] == 1 else C_NEG))

            pA, pB = decision_line_points(w, b, rect)
            ax, ay = to_plot(pA, rect, plot, pad=24.0)
            bx, by = to_plot(pB, rect, plot, pad=24.0)
            dr.line([ax, ay, bx, by], fill=C_LINE, width=3)

            dr.text((inner_x + 22, inner_y + inner_h - 26),
                    f"w=[{w[0]: .2f}, {w[1]: .2f}]  b={b: .2f}", fill=(100, 116, 139))

            images.append(im)

    # final + hold
    last = snaps[-1]
    final = images[-1]
    for _ in range(HOLD_LAST):
        images.append(final)

    images[0].save(out, save_all=True, append_images=images[1:], duration=FRAME_MS, loop=0, optimize=False)
    print(f"[ok] wrote {out} | acc_final={last.acc:.3f} | snaps={len(snaps)} | frames={len(images)}")


if __name__ == "__main__":
    main()