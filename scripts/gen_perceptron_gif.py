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

N_PER_CLASS = 140
NOISE = 0.55

EPOCHS = 18
LR = 0.12

# GIF pacing
TARGET_TOTAL_FRAMES = 300
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
C_NEG = (239, 68, 68)     # -1 (vermelho)
C_POS = (37, 99, 235)     # +1 (azul)
C_LINE = (34, 197, 94)    # fronteira (verde)


@dataclass(frozen=True)
class Snap:
    w: np.ndarray  # (2,)
    b: float
    acc: float
    epoch: int


def seed_from_today() -> int:
    d = date.today()
    return (d.year * 10000 + d.month * 100 + d.day) % (2**32 - 1)


def gen_separable_2d(rng: np.random.Generator, n_per_class: int, noise: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gera dois clusters gaussianos bem separados (separável linearmente na prática).
    y em {-1, +1}
    """
    m0 = np.array([-2.0, -1.2])
    m1 = np.array([ 2.0,  1.4])

    cov = np.array([[noise, 0.10], [0.10, noise]])

    X0 = rng.multivariate_normal(m0, cov, size=n_per_class)
    X1 = rng.multivariate_normal(m1, cov, size=n_per_class)

    X = np.vstack([X0, X1]).astype(float)
    y = np.hstack([-np.ones(n_per_class, dtype=int), np.ones(n_per_class, dtype=int)])

    # embaralhar
    idx = rng.permutation(X.shape[0])
    return X[idx], y[idx]


def perceptron_train_snaps(X: np.ndarray, y: np.ndarray, epochs: int, lr: float, rng: np.random.Generator) -> List[Snap]:
    """
    Perceptron online: atualiza a cada amostra.
    Captura snapshot por época (e também snapshots extras quando houve muitas correções).
    """
    w = rng.normal(0.0, 0.7, size=(2,)).astype(float)
    b = float(rng.normal(0.0, 0.4))

    snaps: List[Snap] = []

    def accuracy(w_: np.ndarray, b_: float) -> float:
        pred = np.sign(X @ w_ + b_)
        pred[pred == 0] = 1
        return float((pred == y).mean())

    for ep in range(1, epochs + 1):
        # embaralha por época
        order = rng.permutation(X.shape[0])

        updates = 0
        for i in order:
            xi = X[i]
            yi = y[i]
            a = float(np.dot(w, xi) + b)
            yhat = 1 if a >= 0 else -1
            if yhat != yi:
                # atualização perceptron
                w = w + lr * yi * xi
                b = b + lr * yi
                updates += 1

        acc = accuracy(w, b)
        snaps.append(Snap(w=w.copy(), b=b, acc=acc, epoch=ep))

        # snapshot extra se teve muita correção (deixa o movimento mais “contínuo” nos estágios difíceis)
        if updates > X.shape[0] * 0.12 and ep < epochs:
            w2 = w + rng.normal(0, 0.02, size=(2,))  # micro-variação p/ evitar empate visual
            b2 = b + float(rng.normal(0, 0.02))
            acc2 = accuracy(w2, b2)
            snaps.append(Snap(w=w2.copy(), b=b2, acc=acc2, epoch=ep))

    return snaps


def normalize_to_rect(X: np.ndarray, x0: float, y0: float, w: float, h: float, pad: float = 20.0) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    """
    Normaliza para retângulo e retorna também (xmin,ymin,xmax,ymax) do espaço original
    para desenhar a fronteira em coordenadas do espaço original com conversão.
    """
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
    """
    Converte ponto do espaço original para pixel no plot (usando mesma escala do normalize_to_rect).
    """
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
        order = np.argsort(weights)  # reduzir nos menores
        i = 0
        while cur > target_total_frames and i < order.size * 80:
            j = order[i % order.size]
            if tw[j] > min_tween:
                tw[j] -= 1
                cur -= 1
            i += 1
    elif cur < target_total_frames:
        order = np.argsort(-weights)  # aumentar nos maiores
        i = 0
        while cur < target_total_frames and i < order.size * 80:
            j = order[i % order.size]
            if tw[j] < max_tween:
                tw[j] += 1
                cur += 1
            i += 1

    return tw.tolist()


def decision_line_points(w: np.ndarray, b: float, rect: Tuple[float, float, float, float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fronteira: w0*x + w1*y + b = 0.
    Retorna dois pontos (x,y) no espaço original, usando extremos do retângulo.
    """
    xmin, ymin, xmax, ymax = rect
    w0, w1 = float(w[0]), float(w[1])

    # Caso quase vertical/horizontal: escolhe interseções robustas
    eps = 1e-9
    pts = []

    # Interseção com x=xmin e x=xmax: y = -(w0*x + b)/w1
    if abs(w1) > eps:
        y_at_xmin = -(w0 * xmin + b) / w1
        y_at_xmax = -(w0 * xmax + b) / w1
        pts.append(np.array([xmin, y_at_xmin], dtype=float))
        pts.append(np.array([xmax, y_at_xmax], dtype=float))
        return pts[0], pts[1]

    # w1 ~ 0 => linha vertical: x = -b/w0
    if abs(w0) > eps:
        x0 = -b / w0
        return np.array([x0, ymin], dtype=float), np.array([x0, ymax], dtype=float)

    # w ~ 0: retorna diagonal qualquer
    return np.array([xmin, ymin], dtype=float), np.array([xmax, ymax], dtype=float)


def main() -> None:
    Path("assets").mkdir(parents=True, exist_ok=True)
    out = Path("assets/perceptron.gif")

    seed = seed_from_today()
    rng = np.random.default_rng(seed)

    # Layout do card
    inner_x, inner_y = CARD_PAD, CARD_PAD
    inner_w, inner_h = CANVAS_W - 2 * CARD_PAD, CANVAS_H - 2 * CARD_PAD

    plot_x0 = inner_x + LEFT_W + GAP
    plot_y0 = inner_y + 22
    plot_w = inner_w - LEFT_W - GAP - 18
    plot_h = inner_h - 44
    plot = (plot_x0, plot_y0, plot_w, plot_h)

    # Dataset
    X, y = gen_separable_2d(rng, N_PER_CLASS, NOISE)

    # Snaps do treinamento
    snaps = perceptron_train_snaps(X, y, EPOCHS, LR, rng)

    # Normalização dos pontos para plot
    Xv, rect = normalize_to_rect(X, plot_x0, plot_y0, plot_w, plot_h, pad=24.0)

    # Pesos para pacing adaptativo (mudança em w,b)
    weights = []
    for i in range(len(snaps) - 1):
        dw = float(np.linalg.norm(snaps[i + 1].w - snaps[i].w))
        db = float(abs(snaps[i + 1].b - snaps[i].b))
        weights.append(dw + 0.25 * db + 1e-6)
    weights = np.array(weights, dtype=float)
    tweens = allocate_tweens(weights, TARGET_TOTAL_FRAMES, MIN_TWEEN, MAX_TWEEN)

    images: List[Image.Image] = []

    # Render
    for i in range(len(snaps) - 1):
        s0 = snaps[i]
        s1 = snaps[i + 1]
        nsub = tweens[i] if i < len(tweens) else MIN_TWEEN

        for sf in range(nsub):
            t = sf / float(nsub)  # 0..(nsub-1)/nsub
            w = lerp(s0.w, s1.w, t)
            b = float((1.0 - t) * s0.b + t * s1.b)
            acc = float((1.0 - t) * s0.acc + t * s1.acc)

            im = Image.new("RGB", (CANVAS_W, CANVAS_H), BG)
            dr = ImageDraw.Draw(im)

            # Card
            dr.rounded_rectangle(
                [inner_x, inner_y, inner_x + inner_w, inner_y + inner_h],
                radius=16, fill=CARD, outline=STROKE, width=2
            )

            # Cabeçalho
            dr.text((inner_x + 22, inner_y + 16), "Perceptron", fill=TEXT)
            dr.text(
                (inner_x + 140, inner_y + 20),
                f"2D linear classifier | epoch={s0.epoch}/{EPOCHS} | subframe={sf+1}/{nsub} | seed={seed}",
                fill=MUTED
            )

            # Barra de acurácia
            bar_x, bar_y, bar_w, bar_h = inner_x + 22, inner_y + 50, LEFT_W - 44, 12
            dr.rounded_rectangle([bar_x, bar_y, bar_x + bar_w, bar_y + bar_h], radius=8, fill=BAR_BG, outline=STROKE)
            prog = max(0.0, min(1.0, acc))
            dr.rounded_rectangle([bar_x, bar_y, bar_x + int(bar_w * prog), bar_y + bar_h], radius=8, fill=BAR_FILL)
            dr.text((inner_x + 22, inner_y + 70), f"accuracy: {acc*100:5.1f}%", fill=MUTED)

            # Legenda classes
            lx, ly = inner_x + 22, inner_y + 105
            dr.rectangle([lx, ly - 10, lx + 12, ly + 2], fill=C_POS)
            dr.text((lx + 18, ly - 12), "y=+1", fill=MUTED)
            dr.rectangle([lx, ly + 18 - 10, lx + 12, ly + 18 + 2], fill=C_NEG)
            dr.text((lx + 18, ly + 18 - 12), "y=-1", fill=MUTED)

            # Plot BG
            dr.rounded_rectangle(
                [plot_x0 - 12, plot_y0 - 12, plot_x0 + plot_w + 12, plot_y0 + plot_h + 12],
                radius=14, fill=(11, 18, 32), outline=STROKE
            )

            # Pontos
            r = 3
            for p_idx, (px, py) in enumerate(Xv):
                col = C_POS if y[p_idx] == 1 else C_NEG
                dr.ellipse([px - r, py - r, px + r, py + r], fill=col)

            # Fronteira de decisão
            pA, pB = decision_line_points(w, b, rect)
            ax, ay = to_plot(pA, rect, plot, pad=24.0)
            bx, by = to_plot(pB, rect, plot, pad=24.0)
            dr.line([ax, ay, bx, by], fill=C_LINE, width=3)

            # Rodapé com pesos (curto)
            dr.text(
                (inner_x + 22, inner_y + inner_h - 26),
                f"w=[{w[0]: .2f}, {w[1]: .2f}]  b={b: .2f}",
                fill=(100, 116, 139)
            )

            images.append(im)

    # Frame final + hold
    last = snaps[-1]
    def render_final() -> Image.Image:
        im = Image.new("RGB", (CANVAS_W, CANVAS_H), BG)
        dr = ImageDraw.Draw(im)

        dr.rounded_rectangle(
            [inner_x, inner_y, inner_x + inner_w, inner_y + inner_h],
            radius=16, fill=CARD, outline=STROKE, width=2
        )

        dr.text((inner_x + 22, inner_y + 16), "Perceptron", fill=TEXT)
        dr.text((inner_x + 140, inner_y + 20), f"final | epoch={EPOCHS}/{EPOCHS} | seed={seed}", fill=MUTED)

        bar_x, bar_y, bar_w, bar_h = inner_x + 22, inner_y + 50, LEFT_W - 44, 12
        dr.rounded_rectangle([bar_x, bar_y, bar_x + bar_w, bar_y + bar_h], radius=8, fill=BAR_BG, outline=STROKE)
        prog = max(0.0, min(1.0, last.acc))
        dr.rounded_rectangle([bar_x, bar_y, bar_x + int(bar_w * prog), bar_y + bar_h], radius=8, fill=BAR_FILL)
        dr.text((inner_x + 22, inner_y + 70), f"accuracy: {last.acc*100:5.1f}%", fill=MUTED)

        dr.rounded_rectangle(
            [plot_x0 - 12, plot_y0 - 12, plot_x0 + plot_w + 12, plot_y0 + plot_h + 12],
            radius=14, fill=(11, 18, 32), outline=STROKE
        )

        r = 3
        for p_idx, (px, py) in enumerate(Xv):
            col = C_POS if y[p_idx] == 1 else C_NEG
            dr.ellipse([px - r, py - r, px + r, py + r], fill=col)

        pA, pB = decision_line_points(last.w, last.b, rect)
        ax, ay = to_plot(pA, rect, plot, pad=24.0)
        bx, by = to_plot(pB, rect, plot, pad=24.0)
        dr.line([ax, ay, bx, by], fill=C_LINE, width=3)

        dr.text((inner_x + 22, inner_y + inner_h - 26),
                f"w=[{last.w[0]: .2f}, {last.w[1]: .2f}]  b={last.b: .2f}",
                fill=(100, 116, 139))
        return im

    final_img = render_final()
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
    print(f"[ok] wrote {out} | frames={len(images)}")


if __name__ == "__main__":
    main()