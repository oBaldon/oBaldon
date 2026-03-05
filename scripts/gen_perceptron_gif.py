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

N_TOTAL = 420
NOISE_X = 1.00           # ruído nas features (não quebra separabilidade por si só)
MARGIN = 0.30            # margem mínima do professor (maior => mais separável)
LABEL_FLIP_P = 0.04      # prob. de inverter rótulo (0 => separável; >0 => não separável)

EPOCHS = 26
LR = 0.05

# Snapshots
SNAP_EVERY_UPDATES = 12
SNAP_EVERY_EPOCH = 1

# GIF pacing
TARGET_TOTAL_FRAMES = 320
MIN_TWEEN = 3
MAX_TWEEN = 16
FRAME_MS = 55
HOLD_LAST = 12

POINT_R = 3
LINE_W = 3

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
C_LINE = (34, 197, 94)    # linha "pocket" (best)
C_LINE_CUR = (56, 189, 248)  # linha atual (opcional, azul claro)


@dataclass(frozen=True)
class Snap:
    # estado atual (perceptron)
    w: np.ndarray
    b: float
    # melhor estado observado (pocket)
    w_best: np.ndarray
    b_best: float
    best_acc: float
    cur_acc: float
    updates: int
    epoch: int


def seed_from_today() -> int:
    d = date.today()
    return (d.year * 10000 + d.month * 100 + d.day) % (2**32 - 1)


def sign_pm1(a: np.ndarray) -> np.ndarray:
    return np.where(a >= 0.0, 1, -1).astype(int)


def accuracy(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float) -> float:
    pred = sign_pm1(X @ w + b)
    return float((pred == y).mean())


def gen_dataset_teacher(
    rng: np.random.Generator,
    n_total: int,
    noise_x: float,
    margin: float,
    label_flip_p: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Gera dados a partir de um hiperplano "professor" (w*, b*).
    - Garante margem mínima (se label_flip_p==0, separável).
    - Se label_flip_p>0, introduz não separabilidade (ruído de rótulo).
    Retorna X, y, w*, b*.
    """
    # professor (direção aleatória normalizada)
    w_star = rng.normal(0.0, 1.0, size=(2,))
    w_star = w_star / (np.linalg.norm(w_star) + 1e-12)
    b_star = float(rng.normal(0.0, 0.25))

    X = []
    y = []

    # Rejeição: força |w*x + b| >= margin
    tries = 0
    while len(X) < n_total and tries < n_total * 400:
        tries += 1
        x = rng.normal(0.0, noise_x, size=(2,))
        s = float(np.dot(w_star, x) + b_star)
        if abs(s) < margin:
            continue
        yi = 1 if s >= 0 else -1
        X.append(x)
        y.append(yi)

    X = np.array(X, dtype=float)
    y = np.array(y, dtype=int)

    # Ruído de rótulo (quebra separabilidade)
    if label_flip_p > 0:
        flip = rng.random(size=y.shape[0]) < label_flip_p
        y = y.copy()
        y[flip] *= -1

    # Balanceamento leve (opcional): embaralha
    idx = rng.permutation(X.shape[0])
    return X[idx], y[idx], w_star, b_star


def normalize_to_rect(
    X: np.ndarray, x0: float, y0: float, w: float, h: float, pad: float = 24.0
) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    xmin, ymin = float(X[:, 0].min()), float(X[:, 1].min())
    xmax, ymax = float(X[:, 0].max()), float(X[:, 1].max())

    sx = (w - 2 * pad) / (xmax - xmin + 1e-9)
    sy = (h - 2 * pad) / (ymax - ymin + 1e-9)
    s = min(sx, sy)

    Yn = (X - np.array([xmin, ymin])) * s
    Yn[:, 0] = x0 + pad + Yn[:, 0]
    Yn[:, 1] = y0 + h - pad - Yn[:, 1]
    return Yn, (xmin, ymin, xmax, ymax)


def to_plot(
    pt: np.ndarray,
    rect: Tuple[float, float, float, float],
    plot: Tuple[float, float, float, float],
    pad: float = 24.0,
) -> Tuple[float, float]:
    xmin, ymin, xmax, ymax = rect
    x0, y0, w, h = plot

    sx = (w - 2 * pad) / (xmax - xmin + 1e-9)
    sy = (h - 2 * pad) / (ymax - ymin + 1e-9)
    s = min(sx, sy)

    x = x0 + pad + (pt[0] - xmin) * s
    y = y0 + h - pad - (pt[1] - ymin) * s
    return float(x), float(y)


def decision_line_points(
    w: np.ndarray, b: float, rect: Tuple[float, float, float, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Linha: w0*x + w1*y + b = 0.
    Retorna dois pontos no espaço original (para converter para o plot).
    """
    xmin, ymin, xmax, ymax = rect
    w0, w1 = float(w[0]), float(w[1])
    eps = 1e-9

    if abs(w1) > eps:
        y_at_xmin = -(w0 * xmin + b) / w1
        y_at_xmax = -(w0 * xmax + b) / w1
        return np.array([xmin, y_at_xmin]), np.array([xmax, y_at_xmax])

    if abs(w0) > eps:
        x0 = -b / w0
        return np.array([x0, ymin]), np.array([x0, ymax])

    return np.array([xmin, ymin]), np.array([xmax, ymax])


def lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return (1.0 - t) * a + t * b


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
        while cur > target and i < order.size * 120:
            j = order[i % order.size]
            if tw[j] > min_tween:
                tw[j] -= 1
                cur -= 1
            i += 1
    elif cur < target:
        order = np.argsort(-weights)
        i = 0
        while cur < target and i < order.size * 120:
            j = order[i % order.size]
            if tw[j] < max_tween:
                tw[j] += 1
                cur += 1
            i += 1
    return tw.tolist()


def perceptron_with_pocket_snaps(
    X: np.ndarray, y: np.ndarray, epochs: int, lr: float, rng: np.random.Generator
) -> List[Snap]:
    """
    Perceptron online (fiel): atualiza em cada erro.
    Pocket: mantém (w_best,b_best) com melhor acurácia observada.
    Snapshots a cada SNAP_EVERY_UPDATES e ao final de cada época.
    """
    # inicialização neutra (mais fiel e previsível)
    w = np.zeros(2, dtype=float)
    b = 0.0

    w_best = w.copy()
    b_best = b
    best_acc = accuracy(X, y, w_best, b_best)

    snaps: List[Snap] = []
    updates = 0

    snaps.append(
        Snap(
            w=w.copy(),
            b=b,
            w_best=w_best.copy(),
            b_best=b_best,
            best_acc=best_acc,
            cur_acc=best_acc,
            updates=updates,
            epoch=0,
        )
    )

    for ep in range(1, epochs + 1):
        order = rng.permutation(X.shape[0])

        for i in order:
            xi = X[i]
            yi = int(y[i])

            a = float(np.dot(w, xi) + b)
            yhat = 1 if a >= 0 else -1

            if yhat != yi:
                # atualização perceptron (matematicamente padrão)
                w = w + lr * yi * xi
                b = b + lr * yi
                updates += 1

                cur_acc = accuracy(X, y, w, b)
                if cur_acc > best_acc + 1e-12:
                    best_acc = cur_acc
                    w_best = w.copy()
                    b_best = b

                if updates % SNAP_EVERY_UPDATES == 0:
                    snaps.append(
                        Snap(
                            w=w.copy(),
                            b=b,
                            w_best=w_best.copy(),
                            b_best=b_best,
                            best_acc=best_acc,
                            cur_acc=cur_acc,
                            updates=updates,
                            epoch=ep,
                        )
                    )

        if SNAP_EVERY_EPOCH:
            cur_acc = accuracy(X, y, w, b)
            if cur_acc > best_acc + 1e-12:
                best_acc = cur_acc
                w_best = w.copy()
                b_best = b

            snaps.append(
                Snap(
                    w=w.copy(),
                    b=b,
                    w_best=w_best.copy(),
                    b_best=b_best,
                    best_acc=best_acc,
                    cur_acc=cur_acc,
                    updates=updates,
                    epoch=ep,
                )
            )

        # Se separável e já atingiu 100% no pocket, pode parar cedo
        if best_acc >= 0.999:
            break

    # remove duplicados (pocket igual e w igual)
    cleaned = [snaps[0]]
    for s in snaps[1:]:
        p = cleaned[-1]
        if (
            np.linalg.norm(s.w_best - p.w_best) > 1e-10
            or abs(s.b_best - p.b_best) > 1e-10
            or abs(s.best_acc - p.best_acc) > 1e-12
        ):
            cleaned.append(s)
    return cleaned


def main() -> None:
    Path("assets").mkdir(parents=True, exist_ok=True)
    out = Path("assets/perceptron.gif")

    seed = seed_from_today()
    rng = np.random.default_rng(seed)

    # Layout
    inner_x, inner_y = CARD_PAD, CARD_PAD
    inner_w, inner_h = CANVAS_W - 2 * CARD_PAD, CANVAS_H - 2 * CARD_PAD

    plot_x0 = inner_x + LEFT_W + GAP
    plot_y0 = inner_y + 22
    plot_w = inner_w - LEFT_W - GAP - 18
    plot_h = inner_h - 44
    plot = (plot_x0, plot_y0, plot_w, plot_h)

    # Dataset
    X, y, w_star, b_star = gen_dataset_teacher(
        rng,
        n_total=N_TOTAL,
        noise_x=NOISE_X,
        margin=MARGIN,
        label_flip_p=LABEL_FLIP_P,
    )

    Xv, rect = normalize_to_rect(X, plot_x0, plot_y0, plot_w, plot_h, pad=24.0)

    # Treino (perceptron + pocket)
    snaps = perceptron_with_pocket_snaps(X, y, EPOCHS, LR, rng)

    # Pacing por mudança do pocket (o que o usuário vê)
    weights = []
    for i in range(len(snaps) - 1):
        s0, s1 = snaps[i], snaps[i + 1]
        dw = float(np.linalg.norm(s1.w_best - s0.w_best))
        db = float(abs(s1.b_best - s0.b_best))
        dacc = float(abs(s1.best_acc - s0.best_acc))
        weights.append(dw + 0.35 * db + 2.5 * dacc + 1e-6)
    weights = np.array(weights, dtype=float)
    tweens = allocate_tweens(weights, TARGET_TOTAL_FRAMES, MIN_TWEEN, MAX_TWEEN)

    images: List[Image.Image] = []

    for i in range(len(snaps) - 1):
        s0 = snaps[i]
        s1 = snaps[i + 1]
        nsub = tweens[i] if i < len(tweens) else MIN_TWEEN

        for sf in range(nsub):
            t = sf / float(nsub)

            # interpola pocket (linha exibida)
            w_best = lerp(s0.w_best, s1.w_best, t)
            b_best = float((1.0 - t) * s0.b_best + t * s1.b_best)
            best_acc = float((1.0 - t) * s0.best_acc + t * s1.best_acc)

            # interpola estado atual (opcional para desenhar linha atual)
            w_cur = lerp(s0.w, s1.w, t)
            b_cur = float((1.0 - t) * s0.b + t * s1.b)
            cur_acc = float((1.0 - t) * s0.cur_acc + t * s1.cur_acc)

            im = Image.new("RGB", (CANVAS_W, CANVAS_H), BG)
            dr = ImageDraw.Draw(im)

            # Card
            dr.rounded_rectangle(
                [inner_x, inner_y, inner_x + inner_w, inner_y + inner_h],
                radius=16, fill=CARD, outline=STROKE, width=2
            )

            dr.text((inner_x + 22, inner_y + 16), "Perceptron (Pocket)", fill=TEXT)
            dr.text(
                (inner_x + 210, inner_y + 20),
                f"epoch={s0.epoch} | updates={s0.updates} | seed={seed}",
                fill=MUTED
            )

            # Barra: best_acc (monótona) — evita “barinha maluca” em dados não separáveis
            bar_x, bar_y, bar_w, bar_h = inner_x + 22, inner_y + 50, LEFT_W - 44, 12
            dr.rounded_rectangle([bar_x, bar_y, bar_x + bar_w, bar_y + bar_h],
                                 radius=8, fill=BAR_BG, outline=STROKE)
            prog = max(0.0, min(1.0, best_acc))
            dr.rounded_rectangle([bar_x, bar_y, bar_x + int(bar_w * prog), bar_y + bar_h],
                                 radius=8, fill=BAR_FILL)
            dr.text((inner_x + 22, inner_y + 70),
                    f"best accuracy: {best_acc*100:5.1f}%   (current: {cur_acc*100:5.1f}%)",
                    fill=MUTED)

            # Indicadores de separabilidade (informativo)
            sep_txt = "separável" if LABEL_FLIP_P == 0.0 else "não separável (ruído de rótulo)"
            dr.text((inner_x + 22, inner_y + 92),
                    f"dataset: {sep_txt} | margin={MARGIN:.2f} | flip_p={LABEL_FLIP_P:.2f}",
                    fill=(120, 135, 155))

            # Legenda
            lx, ly = inner_x + 22, inner_y + 122
            dr.rectangle([lx, ly - 10, lx + 12, ly + 2], fill=C_POS)
            dr.text((lx + 18, ly - 12), "y=+1", fill=MUTED)
            dr.rectangle([lx, ly + 18 - 10, lx + 12, ly + 18 + 2], fill=C_NEG)
            dr.text((lx + 18, ly + 18 - 12), "y=-1", fill=MUTED)

            dr.text((inner_x + 22, inner_y + 168),
                    "linha verde: pocket (melhor até agora)\nlinha azul: estado atual",
                    fill=(120, 135, 155))

            # Plot BG
            dr.rounded_rectangle(
                [plot_x0 - 12, plot_y0 - 12, plot_x0 + plot_w + 12, plot_y0 + plot_h + 12],
                radius=14, fill=(11, 18, 32), outline=STROKE
            )

            # Pontos
            r = POINT_R
            for idx, (px, py) in enumerate(Xv):
                dr.ellipse([px - r, py - r, px + r, py + r], fill=(C_POS if y[idx] == 1 else C_NEG))

            # Linha pocket (verde)
            pA, pB = decision_line_points(w_best, b_best, rect)
            ax, ay = to_plot(pA, rect, plot, pad=24.0)
            bx, by = to_plot(pB, rect, plot, pad=24.0)
            dr.line([ax, ay, bx, by], fill=C_LINE, width=LINE_W)

            # Linha atual (azul claro)
            pA2, pB2 = decision_line_points(w_cur, b_cur, rect)
            ax2, ay2 = to_plot(pA2, rect, plot, pad=24.0)
            bx2, by2 = to_plot(pB2, rect, plot, pad=24.0)
            dr.line([ax2, ay2, bx2, by2], fill=C_LINE_CUR, width=2)

            # Rodapé
            dr.text(
                (inner_x + 22, inner_y + inner_h - 26),
                f"pocket w=[{w_best[0]: .2f}, {w_best[1]: .2f}]  b={b_best: .2f}",
                fill=(100, 116, 139)
            )

            images.append(im)

    # Hold final
    if images:
        final = images[-1]
        for _ in range(HOLD_LAST):
            images.append(final)

        images[0].save(out, save_all=True, append_images=images[1:], duration=FRAME_MS, loop=0, optimize=False)

    last = snaps[-1]
    print(f"[ok] wrote {out} | best_acc={last.best_acc:.3f} | cur_acc={last.cur_acc:.3f} | snaps={len(snaps)} | frames={len(images)}")


if __name__ == "__main__":
    main()