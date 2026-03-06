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
FONT_SMALL = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)

TARGET_TOTAL_FRAMES = 276
MIN_TWEEN = 5
MAX_TWEEN = 12
FRAME_MS = 55
HOLD_FIRST = 10
HOLD_LAST = 18

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

POINT = (56, 189, 248)
POINT_FADE = (52, 95, 145)
MEAN_C = (239, 68, 68)
AXIS_C = (245, 158, 11)
PROJ_C = (34, 197, 94)
GRID = (38, 50, 66)


# =========================
# Dados / estados
# =========================
@dataclass(frozen=True)
class Snap:
    phase: str
    progress: float
    explained: float
    lambda1: float
    lambda2: float
    proj_spread: float
    recon_error: float
    note: str


def seed_from_today() -> int:
    d = date.today()
    return (d.year * 10000 + d.month * 100 + d.day) % (2**32 - 1)


def fit_text(dr: ImageDraw.ImageDraw, text: str, max_w: int, font=FONT) -> str:
    if dr.textlength(text, font=font) <= max_w:
        return text
    ell = "…"
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi) // 2
        cand = text[:mid] + ell
        if dr.textlength(cand, font=font) <= max_w:
            lo = mid + 1
        else:
            hi = mid
    return text[: max(0, lo - 1)] + ell


def wrap_text(dr: ImageDraw.ImageDraw, text: str, max_w: int, font=FONT_SMALL) -> List[str]:
    words = text.split()
    if not words:
        return [""]

    lines: List[str] = []
    cur = words[0]

    for w in words[1:]:
        cand = f"{cur} {w}"
        if dr.textlength(cand, font=font) <= max_w:
            cur = cand
        else:
            lines.append(cur)
            cur = w

    lines.append(cur)
    return lines


def lerp_scalar(a: float, b: float, t: float) -> float:
    return float((1.0 - t) * a + t * b)


def allocate_tweens(
    n_transitions: int,
    target_total_frames: int,
    hold_first: int,
    hold_last: int,
    min_tween: int,
    max_tween: int,
) -> List[int]:
    if n_transitions <= 0:
        return []

    target = max(n_transitions * min_tween, target_total_frames - hold_first - hold_last)
    base = target // n_transitions
    rem = target % n_transitions

    out = [base + (1 if i < rem else 0) for i in range(n_transitions)]
    out = [max(min_tween, min(max_tween, x)) for x in out]

    cur = sum(out)
    i = 0
    while cur < target:
        j = i % n_transitions
        if out[j] < max_tween:
            out[j] += 1
            cur += 1
        i += 1
        if i > 10000:
            break

    i = 0
    while cur > target:
        j = i % n_transitions
        if out[j] > min_tween:
            out[j] -= 1
            cur -= 1
        i += 1
        if i > 10000:
            break

    return out


# =========================
# Dataset e PCA
# =========================
def gen_data(rng: np.random.Generator, n: int = 220) -> np.ndarray:
    angle = float(rng.uniform(-0.95, -0.45))
    R = np.array(
        [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ],
        dtype=float,
    )

    cov = np.array([[3.8, 0.0], [0.0, 0.38]], dtype=float)
    X = rng.multivariate_normal([0.0, 0.0], cov, size=n)

    X[:, 1] += 0.18 * np.sin(0.9 * X[:, 0])
    X = X @ R.T
    X += rng.normal(0.0, 0.05, size=X.shape)
    return X.astype(float)


def compute_pca(X: np.ndarray):
    mu = X.mean(axis=0)
    Xc = X - mu
    cov = np.cov(Xc.T)

    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    v1 = eigvecs[:, 0]
    v2 = eigvecs[:, 1]

    if v1[0] < 0:
        v1 = -v1
    if np.linalg.det(np.column_stack([v1, v2])) < 0:
        v2 = -v2

    z = Xc @ v1
    proj = mu + np.outer(z, v1)
    recon_err = float(np.mean(np.sum((X - proj) ** 2, axis=1)))
    explained = float(eigvals[0] / (eigvals[0] + eigvals[1] + 1e-12))

    return mu, Xc, cov, eigvals, v1, v2, z, proj, explained, recon_err


def build_snaps(explained: float, eigvals: np.ndarray, z: np.ndarray, recon_error: float) -> List[Snap]:
    lam1 = float(eigvals[0])
    lam2 = float(eigvals[1])
    spread = float(np.std(z))

    phases = [
        (
            "data",
            0.0,
            "Dados centralizados: a nuvem já indica a direção dominante de variância.",
        ),
        (
            "mean",
            0.18,
            "A média define o centro a partir do qual a orientação é medida.",
        ),
        (
            "axis",
            0.48,
            "O primeiro autovetor aponta para a direção de máxima variância.",
        ),
        (
            "projection",
            0.78,
            "Os pontos são projetados ortogonalmente sobre a componente principal.",
        ),
        (
            "compressed",
            1.00,
            "A estrutura 2D é resumida em 1D ao longo do eixo principal.",
        ),
    ]

    out: List[Snap] = []
    for phase, p, note in phases:
        out.append(
            Snap(
                phase=phase,
                progress=p,
                explained=explained,
                lambda1=lam1,
                lambda2=lam2,
                proj_spread=spread,
                recon_error=recon_error,
                note=note,
            )
        )
    return out


# =========================
# Geometria
# =========================
def normalize_to_rect(
    X: np.ndarray,
    x0: float,
    y0: float,
    w: float,
    h: float,
    pad: float = 22.0,
):
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


def to_plot(pt: np.ndarray, tf) -> Tuple[float, float]:
    xmin, ymin, xmax, ymax, s, ox, oy = tf
    H = (ymax - ymin) * s
    x = ox + (pt[0] - xmin) * s
    y = oy + (H - (pt[1] - ymin) * s)
    return float(x), float(y)


# =========================
# Desenho
# =========================
def draw_sparkline(
    dr: ImageDraw.ImageDraw,
    values: List[float],
    x0: int,
    y0: int,
    w: int,
    h: int,
):
    dr.rounded_rectangle([x0, y0, x0 + w, y0 + h], radius=9, fill=BAR_BG, outline=STROKE)

    if len(values) < 2:
        return

    vmin = min(values)
    vmax = max(values)
    if abs(vmax - vmin) < 1e-9:
        vmax = vmin + 1.0

    def pt(i: int, val: float) -> Tuple[int, int]:
        x = x0 + 4 + int(round((w - 8) * i / max(1, len(values) - 1)))
        y = y0 + h - 4 - int(round((h - 8) * (val - vmin) / (vmax - vmin)))
        return x, y

    for i in range(1, len(values)):
        dr.line([pt(i - 1, values[i - 1]), pt(i, values[i])], fill=PROJ_C, width=2)


def main() -> None:
    Path("assets").mkdir(exist_ok=True)
    out = Path("assets/pca.gif")

    seed = seed_from_today()
    rng = np.random.default_rng(seed)

    X = gen_data(rng)
    mu, Xc, cov, eigvals, v1, v2, z, proj, explained, recon_error = compute_pca(X)
    snaps = build_snaps(explained, eigvals, z, recon_error)

    inner_x, inner_y = CARD_PAD, CARD_PAD
    inner_w, inner_h = CANVAS_W - 2 * CARD_PAD, CANVAS_H - 2 * CARD_PAD

    plot_x0 = inner_x + LEFT_W + GAP
    plot_y0 = inner_y + 54
    plot_w = inner_w - LEFT_W - GAP - 18
    plot_h = inner_h - 76

    all_pts = np.vstack([X, proj, mu[None, :]])
    all_v, tf = normalize_to_rect(all_pts, plot_x0, plot_y0, plot_w, plot_h, pad=24.0)
    Xv = all_v[: X.shape[0]]
    Projv = all_v[X.shape[0] : X.shape[0] + proj.shape[0]]
    Mv = all_v[-1]

    mu_plot = np.array(to_plot(mu, tf))
    axis_dir_plot = np.array(to_plot(mu + v1, tf)) - mu_plot
    axis_dir_plot = axis_dir_plot / (np.linalg.norm(axis_dir_plot) + 1e-12)

    axis_len = max(plot_w, plot_h) * 0.38

    transitions = len(snaps) - 1
    tweens = allocate_tweens(
        transitions,
        TARGET_TOTAL_FRAMES,
        HOLD_FIRST,
        HOLD_LAST,
        MIN_TWEEN,
        MAX_TWEEN,
    )

    def draw_metrics(dr: ImageDraw.ImageDraw, x: int, y: int, st: Snap) -> int:
        dy = 22
        lines = [
            ("dimensão", "2D → 1D"),
            ("amostras", f"{X.shape[0]}"),
            ("var. explicada", f"{100.0 * st.explained:.1f}%"),
            ("λ1 / λ2", f"{st.lambda1:.2f} / {st.lambda2:.2f}"),
            ("spread proj.", f"{st.proj_spread:.2f}"),
            ("erro 1D", f"{st.recon_error:.3f}"),
        ]
        for i, (k, v) in enumerate(lines):
            yy = y + i * dy
            dr.text((x, yy), k, fill=MUTED, font=FONT)
            dr.text((x + 142, yy), v, fill=TEXT, font=FONT)
        return y + len(lines) * dy

    def render_frame(st_a: Snap, st_b: Snap, t: float, idx_b: int) -> Image.Image:
        progress = lerp_scalar(st_a.progress, st_b.progress, t)

        mean_alpha = float(np.clip((progress - 0.08) / 0.12, 0.0, 1.0))
        axis_alpha = float(np.clip((progress - 0.18) / 0.20, 0.0, 1.0))
        proj_alpha = float(np.clip((progress - 0.42) / 0.25, 0.0, 1.0))
        comp_alpha = float(np.clip((progress - 0.72) / 0.22, 0.0, 1.0))

        explained_now = lerp_scalar(st_a.explained, st_b.explained, t)
        lam1_now = lerp_scalar(st_a.lambda1, st_b.lambda1, t)
        lam2_now = lerp_scalar(st_a.lambda2, st_b.lambda2, t)
        spread_now = lerp_scalar(st_a.proj_spread, st_b.proj_spread, t)
        err_now = lerp_scalar(st_a.recon_error, st_b.recon_error, t)

        im = Image.new("RGB", (CANVAS_W, CANVAS_H), BG)
        dr = ImageDraw.Draw(im)

        dr.rounded_rectangle(
            [inner_x, inner_y, inner_x + inner_w, inner_y + inner_h],
            radius=16,
            fill=CARD,
            outline=STROKE,
            width=2,
        )

        title = "PCA"
        dr.text((inner_x + 22, inner_y + 14), title, fill=TEXT, font=FONT)

        meta = f"principal component analysis | seed={seed}"
        meta_x = inner_x + 98
        meta = fit_text(dr, meta, inner_w - (meta_x - inner_x) - 22, font=FONT)
        dr.text((meta_x, inner_y + 16), meta, fill=MUTED, font=FONT)

        bar_x, bar_y, bar_w, bar_h = inner_x + 22, inner_y + 44, LEFT_W - 44, 12
        dr.rounded_rectangle([bar_x, bar_y, bar_x + bar_w, bar_y + bar_h], radius=8, fill=BAR_BG, outline=STROKE)
        fill_w = int(round(bar_w * progress))
        if fill_w > 0:
            dr.rounded_rectangle([bar_x, bar_y, bar_x + fill_w, bar_y + bar_h], radius=8, fill=BAR_FILL)

        left_x = inner_x + 22
        y = inner_y + 72
        y = draw_metrics(
            dr,
            left_x,
            y,
            Snap(
                phase=st_b.phase,
                progress=progress,
                explained=explained_now,
                lambda1=lam1_now,
                lambda2=lam2_now,
                proj_spread=spread_now,
                recon_error=err_now,
                note=st_b.note,
            ),
        )

        y += 8
        dr.text((left_x, y), "dinâmica", fill=MUTED, font=FONT)
        y += 22
        lines = wrap_text(dr, st_b.note, LEFT_W - 44, font=FONT_SMALL)[:2]
        for line in lines:
            line = fit_text(dr, line, LEFT_W - 44, font=FONT_SMALL)
            dr.text((left_x, y), line, fill=TEXT, font=FONT_SMALL)
            y += 15

        dr.rounded_rectangle(
            [plot_x0 - 12, plot_y0 - 12, plot_x0 + plot_w + 12, plot_y0 + plot_h + 12],
            radius=14,
            fill=(11, 18, 32),
            outline=STROKE,
        )

        for gx in range(plot_x0 + 16, plot_x0 + plot_w, 34):
            dr.line([(gx, plot_y0), (gx, plot_y0 + plot_h)], fill=GRID, width=1)
        for gy in range(plot_y0 + 16, plot_y0 + plot_h, 34):
            dr.line([(plot_x0, gy), (plot_x0 + plot_w, gy)], fill=GRID, width=1)

        for px, py in Xv:
            dr.ellipse([px - 2, py - 2, px + 2, py + 2], fill=POINT)

        if mean_alpha > 0:
            mx, my = float(Mv[0]), float(Mv[1])
            r = 6
            col = MEAN_C
            dr.ellipse([mx - r, my - r, mx + r, my + r], outline=col, width=2)
            dr.line([(mx - 7, my), (mx + 7, my)], fill=col, width=2)
            dr.line([(mx, my - 7), (mx, my + 7)], fill=col, width=2)

        if axis_alpha > 0:
            mxy = np.array([float(Mv[0]), float(Mv[1])], dtype=float)
            d = axis_dir_plot
            p1 = mxy - d * axis_len * axis_alpha
            p2 = mxy + d * axis_len * axis_alpha
            dr.line([tuple(p1), tuple(p2)], fill=AXIS_C, width=3)

        if proj_alpha > 0:
            n_lines = min(90, X.shape[0])
            idxs = np.linspace(0, X.shape[0] - 1, n_lines).astype(int)
            for i in idxs:
                x1, y1 = float(Xv[i, 0]), float(Xv[i, 1])
                x2, y2 = float(Projv[i, 0]), float(Projv[i, 1])

                xm = lerp_scalar(x1, x2, proj_alpha)
                ym = lerp_scalar(y1, y2, proj_alpha)

                dr.line([(x1, y1), (xm, ym)], fill=PROJ_C, width=1)
                if proj_alpha > 0.55:
                    dr.ellipse([xm - 2, ym - 2, xm + 2, ym + 2], fill=PROJ_C)

        if comp_alpha > 0:
            mxy = np.array([float(Mv[0]), float(Mv[1])], dtype=float)
            d = axis_dir_plot
            perp = np.array([-d[1], d[0]])
            band_offset = 52.0
            base = mxy + perp * band_offset

            z_norm = (z - z.mean()) / (np.std(z) + 1e-9)
            z_scale = min(plot_w, plot_h) * 0.18

            for zi, pv in zip(z_norm, Projv, strict=False):
                target = base + d * (zi * z_scale)
                start = np.array([float(pv[0]), float(pv[1])])
                cur = (1.0 - comp_alpha) * start + comp_alpha * target
                dr.ellipse([cur[0] - 2, cur[1] - 2, cur[0] + 2, cur[1] + 2], fill=PROJ_C)

            a = base - d * z_scale * 1.15
            b = base + d * z_scale * 1.15
            dr.line([tuple(a), tuple(b)], fill=(80, 92, 112), width=2)

        return im

    images: List[Image.Image] = []

    first = render_frame(snaps[0], snaps[0], 0.0, 0)
    images.extend([first.copy() for _ in range(HOLD_FIRST)])

    for i in range(len(snaps) - 1):
        n = tweens[i] if i < len(tweens) else MIN_TWEEN
        for s in range(n):
            t = s / float(max(1, n))
            images.append(render_frame(snaps[i], snaps[i + 1], t, i + 1))

    last = render_frame(snaps[-1], snaps[-1], 1.0, len(snaps) - 1)
    images.extend([last.copy() for _ in range(HOLD_LAST)])

    images[0].save(
        out,
        save_all=True,
        append_images=images[1:],
        duration=FRAME_MS,
        loop=0,
        optimize=False,
    )

    print(f"[ok] wrote {out} | frames={len(images)} | states={len(snaps)}")


if __name__ == "__main__":
    main()