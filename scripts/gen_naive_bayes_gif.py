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

N_TOTAL = 360
STEPS = 26
TWEEN_PER_STEP = 4
FRAME_MS = 55
HOLD_FIRST = 8
HOLD_LAST = 18

CARD_PAD = 18
LEFT_W = 360
GAP = 18

GRID_W = 220
GRID_H = 160

BG = (11, 18, 32)
CARD = (15, 23, 42)
STROKE = (31, 41, 55)
TEXT = (229, 231, 235)
MUTED = (148, 163, 184)
BAR_BG = (17, 24, 39)
BAR_FILL = (34, 197, 94)

C_NEG = (239, 68, 68)
C_POS = (37, 99, 235)

# versões apagadas para amostras ainda não usadas
C_NEG_DIM = (120, 55, 55)
C_POS_DIM = (55, 90, 150)

C_BOUNDARY = (245, 158, 11)

# heatmap: vermelho escuro -> azul escuro
HM_NEG = np.array([72, 26, 34], dtype=float)
HM_POS = np.array([22, 46, 88], dtype=float)


@dataclass(frozen=True)
class GNBModel:
    prior_pos: float
    prior_neg: float
    mu_pos: np.ndarray
    mu_neg: np.ndarray
    var_pos: np.ndarray
    var_neg: np.ndarray
    acc: float


@dataclass(frozen=True)
class Snap:
    used_n: int
    model: GNBModel
    prob: np.ndarray  # (GRID_H, GRID_W)


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
def gen_dataset(rng: np.random.Generator, n: int) -> Tuple[np.ndarray, np.ndarray]:
    n_pos = n // 2
    n_neg = n - n_pos

    X_pos = np.column_stack([
        rng.normal(1.4, 0.85, size=n_pos),
        rng.normal(1.0, 0.55, size=n_pos),
    ])
    X_neg = np.column_stack([
        rng.normal(-1.2, 0.70, size=n_neg),
        rng.normal(-0.9, 0.95, size=n_neg),
    ])

    X = np.vstack([X_pos, X_neg]).astype(float)
    y = np.hstack([np.ones(n_pos, dtype=int), -np.ones(n_neg, dtype=int)])

    flip = rng.random(n) < 0.04
    y[flip] *= -1

    idx = rng.permutation(n)
    return X[idx], y[idx]


# =========================
# Gaussian Naive Bayes
# =========================
def log_gaussian_diag(X: np.ndarray, mu: np.ndarray, var: np.ndarray) -> np.ndarray:
    return -0.5 * np.sum(np.log(2.0 * np.pi * var) + ((X - mu) ** 2) / var, axis=1)


def predict_proba_pos(X: np.ndarray, model: GNBModel) -> np.ndarray:
    logp_pos = np.log(model.prior_pos + 1e-12) + log_gaussian_diag(X, model.mu_pos, model.var_pos)
    logp_neg = np.log(model.prior_neg + 1e-12) + log_gaussian_diag(X, model.mu_neg, model.var_neg)

    m = np.maximum(logp_pos, logp_neg)
    p_pos = np.exp(logp_pos - m)
    p_neg = np.exp(logp_neg - m)
    return p_pos / (p_pos + p_neg + 1e-12)


def predict_gnb(X: np.ndarray, model: GNBModel) -> np.ndarray:
    p = predict_proba_pos(X, model)
    return np.where(p >= 0.5, 1, -1).astype(int)


def fit_gnb(X: np.ndarray, y: np.ndarray, X_eval: np.ndarray, y_eval: np.ndarray) -> GNBModel:
    X_pos = X[y == 1]
    X_neg = X[y == -1]

    # fallback para subconjuntos pequenos/desbalanceados
    if X_pos.shape[0] < 2:
        X_pos = np.vstack([X_pos, X_pos + 1e-3]) if X_pos.shape[0] == 1 else np.array([[0.5, 0.5], [0.6, 0.6]])
    if X_neg.shape[0] < 2:
        X_neg = np.vstack([X_neg, X_neg + 1e-3]) if X_neg.shape[0] == 1 else np.array([[-0.5, -0.5], [-0.6, -0.6]])

    prior_pos = float(np.mean(y == 1))
    prior_neg = float(np.mean(y == -1))

    mu_pos = X_pos.mean(axis=0)
    mu_neg = X_neg.mean(axis=0)

    var_pos = X_pos.var(axis=0) + 1e-3
    var_neg = X_neg.var(axis=0) + 1e-3

    model = GNBModel(
        prior_pos=prior_pos,
        prior_neg=prior_neg,
        mu_pos=mu_pos,
        mu_neg=mu_neg,
        var_pos=var_pos,
        var_neg=var_neg,
        acc=0.0,
    )

    pred = predict_gnb(X_eval, model)
    acc = float(np.mean(pred == y_eval))

    return GNBModel(
        prior_pos=prior_pos,
        prior_neg=prior_neg,
        mu_pos=mu_pos,
        mu_neg=mu_neg,
        var_pos=var_pos,
        var_neg=var_neg,
        acc=acc,
    )


# =========================
# Plot geometry
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


def lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return (1.0 - t) * a + t * b


# =========================
# Probability field
# =========================
def make_probability_field(
    tf: Tuple[float, float, float, float, float, float, float],
    model: GNBModel,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xmin, ymin, xmax, ymax, s, ox, oy = tf
    xs = np.linspace(xmin, xmax, GRID_W)
    ys = np.linspace(ymin, ymax, GRID_H)
    XX, YY = np.meshgrid(xs, ys)
    P = np.column_stack([XX.ravel(), YY.ravel()])
    prob = predict_proba_pos(P, model).reshape(GRID_H, GRID_W)
    return xs, ys, prob


def probability_to_heat(prob: np.ndarray) -> Image.Image:
    rgb = HM_NEG[None, None, :] * (1.0 - prob[..., None]) + HM_POS[None, None, :] * prob[..., None]
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def draw_boundary(
    dr: ImageDraw.ImageDraw,
    prob: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    tf: Tuple[float, float, float, float, float, float, float],
):
    for gy in range(GRID_H - 1):
        for gx in range(GRID_W - 1):
            block = np.array([
                prob[gy, gx],
                prob[gy, gx + 1],
                prob[gy + 1, gx],
                prob[gy + 1, gx + 1],
            ])
            if block.min() <= 0.5 <= block.max():
                px, py = to_plot(np.array([xs[gx], ys[gy]], dtype=float), tf)
                dr.point((px, py), fill=C_BOUNDARY)


# =========================
# Snap generation
# =========================
def make_snaps(X: np.ndarray, y: np.ndarray, tf) -> List[Snap]:
    counts = np.linspace(16, X.shape[0], STEPS).astype(int)
    counts = np.unique(counts)

    snaps: List[Snap] = []
    for n in counts:
        model = fit_gnb(X[:n], y[:n], X, y)
        _, _, prob = make_probability_field(tf, model)
        snaps.append(Snap(used_n=n, model=model, prob=prob))
    return snaps


# =========================
# Main
# =========================
def main() -> None:
    Path("assets").mkdir(parents=True, exist_ok=True)
    out = Path("assets/naive_bayes.gif")

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
    snaps = make_snaps(X, y, tf)

    acc0 = float(snaps[0].model.acc)
    accN = float(max(s.model.acc for s in snaps))

    left_text_max_w = int(LEFT_W - 44)
    images: List[Image.Image] = []

    def render_frame(
        used_n: int,
        model: GNBModel,
        prob: np.ndarray,
        stage_t: float,
    ) -> Image.Image:
        im = Image.new("RGB", (CANVAS_W, CANVAS_H), BG)
        dr = ImageDraw.Draw(im)

        dr.rounded_rectangle(
            [inner_x, inner_y, inner_x + inner_w, inner_y + inner_h],
            radius=16, fill=CARD, outline=STROKE, width=2
        )

        dr.text((inner_x + 22, inner_y + 16), "Gaussian Naive Bayes", fill=TEXT, font=FONT)

        meta = f"amostras usadas={used_n}/{N_TOTAL} | seed={seed}"
        dr.text((inner_x + 22, inner_y + 36), fit_text(dr, meta, left_text_max_w), fill=MUTED, font=FONT)

        # barra de acurácia
        bar_x, bar_y, bar_w, bar_h = inner_x + 22, inner_y + 56, LEFT_W - 44, 12
        dr.rounded_rectangle([bar_x, bar_y, bar_x + bar_w, bar_y + bar_h],
                             radius=8, fill=BAR_BG, outline=STROKE)

        denom = (accN - acc0) if abs(accN - acc0) > 1e-12 else 1.0
        prog = (model.acc - acc0) / denom
        prog = max(0.0, min(1.0, prog))
        dr.rounded_rectangle([bar_x, bar_y, bar_x + int(bar_w * prog), bar_y + bar_h],
                             radius=8, fill=BAR_FILL)

        y_cursor = inner_y + 78
        dr.text((inner_x + 22, y_cursor),
                fit_text(dr, f"accuracy: {model.acc*100:5.1f}%", left_text_max_w),
                fill=MUTED, font=FONT)
        y_cursor += 20

        pri = f"prior(+1)={model.prior_pos:.2f}   prior(-1)={model.prior_neg:.2f}"
        dr.text((inner_x + 22, y_cursor), fit_text(dr, pri, left_text_max_w), fill=MUTED, font=FONT)
        y_cursor += 22

        desc = "visualização: posterior P(y=+1 | x) aprendida progressivamente com mais amostras"
        for ln in wrap_text(dr, desc, left_text_max_w):
            dr.text((inner_x + 22, y_cursor), ln, fill=(120, 135, 155), font=FONT)
            y_cursor += 18
        y_cursor += 6

        dr.text((inner_x + 22, y_cursor), "pontos fortes: amostras já usadas", fill=(120, 135, 155), font=FONT)
        y_cursor += 18
        dr.text((inner_x + 22, y_cursor), "pontos fracos: amostras ainda não usadas", fill=(120, 135, 155), font=FONT)
        y_cursor += 18
        dr.text((inner_x + 22, y_cursor), "linha laranja: P(y=+1|x)=0.5", fill=(120, 135, 155), font=FONT)

        # plot
        dr.rounded_rectangle(
            [plot_x0 - 12, plot_y0 - 12, plot_x0 + plot_w + 12, plot_y0 + plot_h + 12],
            radius=14, fill=(11, 18, 32), outline=STROKE
        )

        heat = probability_to_heat(prob).resize(
            (int(plot_w - 48), int(plot_h - 48)),
            resample=Image.Resampling.BILINEAR
        )
        im.paste(heat, (int(plot_x0 + 24), int(plot_y0 + 24)))

        xs = np.linspace(tf[0], tf[2], GRID_W)
        ys = np.linspace(tf[1], tf[3], GRID_H)
        draw_boundary(dr, prob, xs, ys, tf)

        # pontos: usados fortes, restantes fracos
        r_used = 2.7
        r_dim = 2.0
        for i, (px, py) in enumerate(Xv):
            used = i < used_n
            if y[i] == 1:
                col = C_POS if used else C_POS_DIM
            else:
                col = C_NEG if used else C_NEG_DIM
            rr = r_used if used else r_dim
            dr.ellipse([px - rr, py - rr, px + rr, py + rr], fill=col)

        footer = (
            f"mu(+)= [{model.mu_pos[0]: .2f}, {model.mu_pos[1]: .2f}]   "
            f"mu(-)= [{model.mu_neg[0]: .2f}, {model.mu_neg[1]: .2f}]"
        )
        dr.text((inner_x + 22, inner_y + inner_h - 26), fit_text(dr, footer, int(inner_w - 44)),
                fill=(100, 116, 139), font=FONT)

        return im

    # primeiro frame
    first = render_frame(snaps[0].used_n, snaps[0].model, snaps[0].prob, 0.0)
    images.append(first)
    for _ in range(HOLD_FIRST):
        images.append(first)

    # tween entre modelos sucessivos
    for i in range(len(snaps) - 1):
        a = snaps[i]
        b = snaps[i + 1]

        for k in range(TWEEN_PER_STEP):
            t = k / float(TWEEN_PER_STEP)

            used_n = int(round((1.0 - t) * a.used_n + t * b.used_n))
            prob = lerp(a.prob, b.prob, t)

            model = GNBModel(
                prior_pos=float((1.0 - t) * a.model.prior_pos + t * b.model.prior_pos),
                prior_neg=float((1.0 - t) * a.model.prior_neg + t * b.model.prior_neg),
                mu_pos=lerp(a.model.mu_pos, b.model.mu_pos, t),
                mu_neg=lerp(a.model.mu_neg, b.model.mu_neg, t),
                var_pos=lerp(a.model.var_pos, b.model.var_pos, t),
                var_neg=lerp(a.model.var_neg, b.model.var_neg, t),
                acc=float((1.0 - t) * a.model.acc + t * b.model.acc),
            )

            images.append(render_frame(used_n, model, prob, t))

    # último frame
    last = render_frame(snaps[-1].used_n, snaps[-1].model, snaps[-1].prob, 1.0)
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