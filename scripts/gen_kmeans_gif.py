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
FONT_SMALL = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)

K = 4
N_POINTS = 280
MAX_ITERS = 10

TARGET_TOTAL_FRAMES = 272
MIN_TWEEN = 5
MAX_TWEEN = 12
FRAME_MS = 55
HOLD_FIRST = 8
HOLD_LAST = 18

CARD_PAD = 18
LEFT_W = 360
GAP = 18

POINT_R = 3
CENTER_R = 11
TRAIL_MAX = 6
REGION_CELL = 10

BG = (11, 18, 32)
CARD = (15, 23, 42)
STROKE = (31, 41, 55)
TEXT = (229, 231, 235)
MUTED = (148, 163, 184)
BAR_BG = (17, 24, 39)
BAR_FILL = (34, 197, 94)
GRID = (38, 50, 66)
TRAIL = (148, 163, 184)
CENTER_OLD = (100, 116, 139)

COLORS = [
    (37, 99, 235),
    (22, 163, 74),
    (245, 158, 11),
    (239, 68, 68),
]

REGION_COLORS = [
    (18, 41, 86),
    (16, 61, 38),
    (88, 56, 12),
    (78, 21, 21),
]


@dataclass(frozen=True)
class Snap:
    centers: np.ndarray
    prev_centers: np.ndarray
    labels: np.ndarray
    prev_labels: np.ndarray
    inertia: float
    best_inertia: float
    changed_ratio: float
    mean_shift: float
    iteration: int
    phase: str
    explain: str
    counts: np.ndarray


def seed_from_today() -> int:
    d = date.today()
    return (d.year * 10000 + d.month * 100 + d.day) % (2**32 - 1)


# =========================
# Utilitários visuais
# =========================
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
    return text[: max(0, lo - 1)] + ell


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


def lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return (1.0 - t) * a + t * b


def lerp_scalar(a: float, b: float, t: float) -> float:
    return float((1.0 - t) * a + t * b)


def lerp_color(c0: Tuple[int, int, int], c1: Tuple[int, int, int], t: float) -> Tuple[int, int, int]:
    return (
        int(round((1.0 - t) * c0[0] + t * c1[0])),
        int(round((1.0 - t) * c0[1] + t * c1[1])),
        int(round((1.0 - t) * c0[2] + t * c1[2])),
    )


def blend(c: Tuple[int, int, int], bg: Tuple[int, int, int], alpha: float) -> Tuple[int, int, int]:
    return (
        int(round(alpha * c[0] + (1.0 - alpha) * bg[0])),
        int(round(alpha * c[1] + (1.0 - alpha) * bg[1])),
        int(round(alpha * c[2] + (1.0 - alpha) * bg[2])),
    )


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
# Dataset
# =========================
def gen_points(rng: np.random.Generator, n: int) -> np.ndarray:
    means = np.array(
        [
            [-3.6, -2.7],
            [3.5, 2.7],
            [-2.1, 3.1],
            [3.8, -2.3],
        ],
        dtype=float,
    )
    covs = [
        np.array([[1.05, 0.22], [0.22, 0.78]], dtype=float),
        np.array([[0.82, -0.20], [-0.20, 0.92]], dtype=float),
        np.array([[0.70, 0.14], [0.14, 0.66]], dtype=float),
        np.array([[0.92, -0.18], [-0.18, 0.72]], dtype=float),
    ]
    weights = np.array([0.27, 0.24, 0.25, 0.24], dtype=float)

    idx = rng.choice(len(means), size=n, replace=True, p=weights)
    X = np.vstack([rng.multivariate_normal(means[j], covs[j]) for j in idx]).astype(float)

    bridge_n = max(20, n // 16)
    bridge = np.column_stack(
        [
            rng.normal(0.1, 1.05, size=bridge_n),
            rng.normal(0.0, 0.75, size=bridge_n),
        ]
    )
    X = np.vstack([X, bridge]).astype(float)

    out_n = 10
    out = rng.normal(0.0, 2.9, size=(out_n, 2)).astype(float)
    X = np.vstack([X, out])

    ang = float(rng.uniform(-0.28, 0.32))
    R = np.array(
        [[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]],
        dtype=float,
    )
    X = X @ R.T
    X[:, 0] *= 1.12
    X[:, 1] *= 1.05
    return X


def kmeans_plus_plus_init(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = X.shape[0]
    centers = np.empty((k, 2), dtype=float)

    first = int(rng.integers(0, n))
    centers[0] = X[first]
    dist2 = ((X - centers[0]) ** 2).sum(axis=1)

    for j in range(1, k):
        probs = dist2 / (dist2.sum() + 1e-12)
        idx = int(rng.choice(n, p=probs))
        centers[j] = X[idx]
        d2_new = ((X - centers[j]) ** 2).sum(axis=1)
        dist2 = np.minimum(dist2, d2_new)

    jitter = rng.normal(0.0, 0.75, size=centers.shape)
    push = centers - X.mean(axis=0, keepdims=True)
    centers = centers + 0.28 * push + jitter
    return centers.astype(float)


def assign_labels_and_inertia(X: np.ndarray, centers: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    labels = d2.argmin(axis=1)
    chosen = np.take_along_axis(d2, labels[:, None], axis=1)[:, 0]
    inertia = float(chosen.sum())
    return labels.astype(int), d2, inertia


def recompute_centers(
    X: np.ndarray,
    labels: np.ndarray,
    old_centers: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    k = old_centers.shape[0]
    new_centers = old_centers.copy()
    for j in range(k):
        pts = X[labels == j]
        if pts.shape[0] == 0:
            new_centers[j] = X[int(rng.integers(0, X.shape[0]))]
        else:
            new_centers[j] = pts.mean(axis=0)
    return new_centers


def build_states(X: np.ndarray, k: int, max_iters: int, rng: np.random.Generator) -> List[Snap]:
    centers = kmeans_plus_plus_init(X, k, rng)
    labels0, _, inertia0 = assign_labels_and_inertia(X, centers)
    counts0 = np.bincount(labels0, minlength=k).astype(int)

    states: List[Snap] = [
        Snap(
            centers=centers.copy(),
            prev_centers=centers.copy(),
            labels=labels0.copy(),
            prev_labels=labels0.copy(),
            inertia=float(inertia0),
            best_inertia=float(inertia0),
            changed_ratio=1.0,
            mean_shift=0.0,
            iteration=0,
            phase="init",
            explain="Inicialização: centróides começam fora do ótimo para evidenciar a formação dos grupos.",
            counts=counts0,
        )
    ]

    best_inertia = float(inertia0)
    prev_labels = labels0.copy()
    prev_centers = centers.copy()

    for it in range(1, max_iters + 1):
        labels, _, inertia_assign = assign_labels_and_inertia(X, centers)
        counts_assign = np.bincount(labels, minlength=k).astype(int)
        changed = float(np.mean(labels != prev_labels)) if prev_labels.shape[0] else 0.0
        best_inertia = min(best_inertia, float(inertia_assign))

        states.append(
            Snap(
                centers=centers.copy(),
                prev_centers=prev_centers.copy(),
                labels=labels.copy(),
                prev_labels=prev_labels.copy(),
                inertia=float(inertia_assign),
                best_inertia=float(best_inertia),
                changed_ratio=changed,
                mean_shift=0.0,
                iteration=it,
                phase="assign",
                explain="Etapa de atribuição: cada amostra é colorida pelo centróide mais próximo.",
                counts=counts_assign,
            )
        )

        new_centers = recompute_centers(X, labels, centers, rng)
        shift = float(np.linalg.norm(new_centers - centers, axis=1).mean())
        labels_after, _, inertia_update = assign_labels_and_inertia(X, new_centers)
        counts_update = np.bincount(labels_after, minlength=k).astype(int)
        best_inertia = min(best_inertia, float(inertia_update))

        states.append(
            Snap(
                centers=new_centers.copy(),
                prev_centers=centers.copy(),
                labels=labels_after.copy(),
                prev_labels=labels.copy(),
                inertia=float(inertia_update),
                best_inertia=float(best_inertia),
                changed_ratio=float(np.mean(labels_after != labels)),
                mean_shift=shift,
                iteration=it,
                phase="update",
                explain="Etapa de atualização: cada centróide se desloca para a média do seu grupo.",
                counts=counts_update,
            )
        )

        prev_labels = labels_after.copy()
        prev_centers = centers.copy()
        centers = new_centers

        if shift < 0.025 and float(np.mean(labels_after != labels)) < 0.003:
            break

    final = states[-1]
    states.append(
        Snap(
            centers=final.centers.copy(),
            prev_centers=final.centers.copy(),
            labels=final.labels.copy(),
            prev_labels=final.labels.copy(),
            inertia=float(final.inertia),
            best_inertia=float(final.best_inertia),
            changed_ratio=0.0,
            mean_shift=0.0,
            iteration=final.iteration,
            phase="final",
            explain="Convergência: as atribuições estabilizam e os centróides praticamente deixam de se mover.",
            counts=final.counts.copy(),
        )
    )

    return states


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


def to_plot(pt: np.ndarray, tf: Tuple[float, float, float, float, float, float, float]) -> Tuple[float, float]:
    xmin, ymin, xmax, ymax, s, ox, oy = tf
    H = (ymax - ymin) * s
    x = ox + (pt[0] - xmin) * s
    y = oy + (H - (pt[1] - ymin) * s)
    return float(x), float(y)


def region_map(
    centers_xy: np.ndarray,
    plot_x0: int,
    plot_y0: int,
    plot_w: int,
    plot_h: int,
    cell: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.arange(plot_x0, plot_x0 + plot_w, cell) + cell * 0.5
    ys = np.arange(plot_y0, plot_y0 + plot_h, cell) + cell * 0.5
    GX, GY = np.meshgrid(xs, ys)
    G = np.stack([GX, GY], axis=-1).reshape(-1, 2)
    d2 = ((G[:, None, :] - centers_xy[None, :, :]) ** 2).sum(axis=2)
    lab = d2.argmin(axis=1).reshape(len(ys), len(xs))
    return xs, ys, lab


# =========================
# Renderização
# =========================
def main() -> None:
    Path("assets").mkdir(parents=True, exist_ok=True)
    out = Path("assets/kmeans.gif")

    seed = seed_from_today()
    rng = np.random.default_rng(seed)

    inner_x, inner_y = CARD_PAD, CARD_PAD
    inner_w, inner_h = CANVAS_W - 2 * CARD_PAD, CANVAS_H - 2 * CARD_PAD

    plot_x0 = inner_x + LEFT_W + GAP
    plot_y0 = inner_y + 54
    plot_w = inner_w - LEFT_W - GAP - 18
    plot_h = inner_h - 76

    X = gen_points(rng, N_POINTS)
    states = build_states(X, K, MAX_ITERS, rng)

    all_centers = np.vstack([st.centers for st in states])
    concat = np.vstack([X, all_centers])
    concat_v, tf = normalize_to_rect(concat, plot_x0, plot_y0, plot_w, plot_h, pad=22.0)
    Xv = concat_v[: X.shape[0]]
    centers_v = concat_v[X.shape[0] :].reshape(len(states), K, 2)

    inertia_init = float(states[0].inertia)
    inertia_best_final = float(min(st.best_inertia for st in states))

    transitions = len(states) - 1
    tweens = allocate_tweens(
        transitions,
        TARGET_TOTAL_FRAMES,
        HOLD_FIRST,
        HOLD_LAST,
        MIN_TWEEN,
        MAX_TWEEN,
    )

    region_cache: dict[Tuple[float, ...], Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def get_regions(centers_xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        key = tuple(np.round(centers_xy.reshape(-1), 1).tolist())
        got = region_cache.get(key)
        if got is None:
            got = region_map(centers_xy, plot_x0, plot_y0, plot_w, plot_h, REGION_CELL)
            region_cache[key] = got
        return got

    def draw_metrics_block(dr: ImageDraw.ImageDraw, x: int, y: int, st: Snap) -> int:
        dy = 22
        lines = [
            ("iteração", f"{st.iteration}"),
            (
                "fase",
                "atribuição"
                if st.phase == "assign"
                else "atualização"
                if st.phase == "update"
                else "inicialização"
                if st.phase == "init"
                else "convergência",
            ),
            ("inércia", f"{st.inertia:,.1f}".replace(",", "X").replace(".", ",").replace("X", ".")),
            ("melhor", f"{st.best_inertia:,.1f}".replace(",", "X").replace(".", ",").replace("X", ".")),
            ("mudanças", f"{100.0 * st.changed_ratio:.1f}%"),
            ("desloc. médio", f"{st.mean_shift:.3f}"),
            ("k / amostras", f"{K} / {X.shape[0]}"),
        ]
        for i, (k, v) in enumerate(lines):
            yy = y + i * dy
            dr.text((x, yy), k, fill=MUTED, font=FONT)
            dr.text((x + 132, yy), v, fill=TEXT, font=FONT)
        return y + len(lines) * dy

    def draw_cluster_counts(dr: ImageDraw.ImageDraw, x: int, y: int, counts: np.ndarray) -> None:
        dr.text((x, y), "tamanho dos grupos", fill=MUTED, font=FONT)
        yy = y + 22
        total = max(1, int(counts.sum()))
        for j in range(K):
            cnt = int(counts[j])
            pct = cnt / total
            dr.rounded_rectangle(
                [x, yy + j * 24, x + 132, yy + 14 + j * 24],
                radius=7,
                fill=BAR_BG,
                outline=STROKE,
            )
            fill_w = int(round(132 * pct))
            if fill_w > 0:
                dr.rounded_rectangle(
                    [x, yy + j * 24, x + fill_w, yy + 14 + j * 24],
                    radius=7,
                    fill=COLORS[j],
                )
            dr.text((x + 142, yy - 3 + j * 24), f"c{j}: {cnt}", fill=TEXT, font=FONT_SMALL)

    def draw_frame(
        st_a: Snap,
        st_b: Snap,
        ca: np.ndarray,
        cb: np.ndarray,
        t: float,
        idx_b: int,
    ) -> Image.Image:
        centers_xy = lerp(ca, cb, t)
        inertia = lerp_scalar(st_a.inertia, st_b.inertia, t)
        best_inertia = min(st_a.best_inertia, st_b.best_inertia)
        changed_ratio = lerp_scalar(st_a.changed_ratio, st_b.changed_ratio, t)
        mean_shift = lerp_scalar(st_a.mean_shift, st_b.mean_shift, t)

        im = Image.new("RGB", (CANVAS_W, CANVAS_H), BG)
        dr = ImageDraw.Draw(im)

        dr.rounded_rectangle(
            [inner_x, inner_y, inner_x + inner_w, inner_y + inner_h],
            radius=16,
            fill=CARD,
            outline=STROKE,
            width=2,
        )

        title = "K-Means"
        dr.text((inner_x + 22, inner_y + 14), title, fill=TEXT, font=FONT)

        meta = f"clusterização iterativa | batch Lloyd | seed={seed}"
        meta_x = inner_x + 22 + int(dr.textlength(title, font=FONT)) + 14
        meta = fit_text(dr, meta, inner_w - (meta_x - inner_x) - 22)
        dr.text((meta_x, inner_y + 16), meta, fill=MUTED, font=FONT)

        bar_x, bar_y, bar_w, bar_h = inner_x + 22, inner_y + 44, LEFT_W - 44, 12
        dr.rounded_rectangle(
            [bar_x, bar_y, bar_x + bar_w, bar_y + bar_h],
            radius=8,
            fill=BAR_BG,
            outline=STROKE,
        )
        prog = 1.0 - (best_inertia - inertia_best_final) / (inertia_init - inertia_best_final + 1e-9)
        prog = float(np.clip(prog, 0.0, 1.0))
        if prog > 0:
            dr.rounded_rectangle(
                [bar_x, bar_y, bar_x + int(bar_w * prog), bar_y + bar_h],
                radius=8,
                fill=BAR_FILL,
            )

        left_x = inner_x + 22
        y = inner_y + 72

        mixed = Snap(
            centers=st_b.centers,
            prev_centers=st_a.prev_centers,
            labels=st_b.labels,
            prev_labels=st_a.prev_labels,
            inertia=inertia,
            best_inertia=best_inertia,
            changed_ratio=changed_ratio,
            mean_shift=mean_shift,
            iteration=st_b.iteration,
            phase=st_b.phase,
            explain=st_b.explain,
            counts=st_b.counts,
        )

        y = draw_metrics_block(dr, left_x, y, mixed)
        y += 10

        text_w = LEFT_W - 44
        dr.text((left_x, y), "interpretação", fill=MUTED, font=FONT)
        y += 22
        for line in wrap_text(dr, st_b.explain, text_w):
            dr.text((left_x, y), line, fill=TEXT, font=FONT)
            y += 19

        y += 4
        draw_cluster_counts(dr, left_x, y, st_b.counts)

        dr.rounded_rectangle(
            [plot_x0 - 12, plot_y0 - 12, plot_x0 + plot_w + 12, plot_y0 + plot_h + 12],
            radius=14,
            fill=(11, 18, 32),
            outline=STROKE,
        )

        for gx in range(plot_x0 + 18, plot_x0 + plot_w, 36):
            dr.line([(gx, plot_y0), (gx, plot_y0 + plot_h)], fill=blend(GRID, BG, 0.55), width=1)
        for gy in range(plot_y0 + 18, plot_y0 + plot_h, 36):
            dr.line([(plot_x0, gy), (plot_x0 + plot_w, gy)], fill=blend(GRID, BG, 0.55), width=1)

        xs, ys, reg = get_regions(centers_xy)
        for iy, cy in enumerate(ys):
            for ix, cx in enumerate(xs):
                col = REGION_COLORS[int(reg[iy, ix])]
                col = blend(col, BG, 0.43)
                x1 = int(round(cx - REGION_CELL * 0.5))
                y1 = int(round(cy - REGION_CELL * 0.5))
                x2 = int(round(cx + REGION_CELL * 0.5))
                y2 = int(round(cy + REGION_CELL * 0.5))
                dr.rectangle([x1, y1, x2, y2], fill=col)

        show_lines = (st_b.phase in {"assign", "update"}) and changed_ratio > 0.02
        if show_lines:
            idx_sorted = np.argsort(((X - st_b.centers[st_b.labels]) ** 2).sum(axis=1))[::-1]
            line_count = min(28, X.shape[0])
            for idx in idx_sorted[:line_count]:
                if int(st_a.labels[idx]) == int(st_b.labels[idx]) and st_b.phase == "assign":
                    continue
                px, py = float(Xv[idx, 0]), float(Xv[idx, 1])
                cj = int(st_b.labels[idx])
                cx, cy = float(centers_xy[cj, 0]), float(centers_xy[cj, 1])
                dr.line([(px, py), (cx, cy)], fill=blend(COLORS[cj], BG, 0.45), width=1)

        for i, (px, py) in enumerate(Xv):
            old_c = COLORS[int(st_a.labels[i])]
            new_c = COLORS[int(st_b.labels[i])]
            c = old_c if old_c == new_c else lerp_color(old_c, new_c, min(1.0, t * 1.1))
            dr.ellipse([px - POINT_R, py - POINT_R, px + POINT_R, py + POINT_R], fill=c)

        start_idx = max(0, idx_b - TRAIL_MAX)
        hist = [st.centers for st in states[start_idx : idx_b + 1]]
        if hist:
            hist_arr = np.stack(hist, axis=0)
            hist_v = np.empty_like(hist_arr)
            for a in range(hist_arr.shape[0]):
                for j in range(K):
                    hist_v[a, j] = np.array(to_plot(hist_arr[a, j], tf))
            for j in range(K):
                for a in range(1, hist_v.shape[0]):
                    alpha = 0.20 + 0.60 * (a / max(1, hist_v.shape[0] - 1))
                    dr.line(
                        [tuple(hist_v[a - 1, j]), tuple(hist_v[a, j])],
                        fill=blend(COLORS[j], BG, alpha),
                        width=2,
                    )

        for j in range(K):
            ocx, ocy = to_plot(st_b.prev_centers[j], tf)
            ncx, ncy = float(centers_xy[j, 0]), float(centers_xy[j, 1])
            dr.line([(ocx, ocy), (ncx, ncy)], fill=blend(TRAIL, BG, 0.55), width=2)
            dr.ellipse([ocx - 4, ocy - 4, ocx + 4, ocy + 4], outline=CENTER_OLD, width=2)

        for j in range(K):
            cx, cy = float(centers_xy[j, 0]), float(centers_xy[j, 1])
            dr.ellipse(
                [cx - CENTER_R, cy - CENTER_R, cx + CENTER_R, cy + CENTER_R],
                fill=COLORS[j],
                outline=TEXT,
                width=2,
            )
            dr.line([(cx - 6, cy), (cx + 6, cy)], fill=TEXT, width=2)
            dr.line([(cx, cy - 6), (cx, cy + 6)], fill=TEXT, width=2)

        legend_lines = [
            "fundo: região do centróide mais próximo   pontos: amostras atribuídas",
            "alvos: centróides   trilhas: movimento dos centróides",
        ]

        leg_y = plot_y0 + plot_h - 30
        for i, line in enumerate(legend_lines):
            line = fit_text(dr, line, plot_w - 12)
            dr.text(
                (plot_x0 + 8, leg_y + i * 14),
                line,
                fill=MUTED,
                font=FONT_SMALL,
            )

        flat_centers = "; ".join([f"c{j}=({st_b.centers[j,0]:.2f},{st_b.centers[j,1]:.2f})" for j in range(K)])
        footer = fit_text(dr, f"centros: {flat_centers}", plot_w - 8)
        dr.text((plot_x0 + 6, plot_y0 - 28), footer, fill=MUTED, font=FONT_SMALL)

        return im

    images: List[Image.Image] = []

    if states:
        first = draw_frame(states[0], states[0], centers_v[0], centers_v[0], 0.0, 0)
        images.extend([first.copy() for _ in range(HOLD_FIRST)])

    for i in range(len(states) - 1):
        st_a = states[i]
        st_b = states[i + 1]
        ca = centers_v[i]
        cb = centers_v[i + 1]
        nsub = tweens[i] if i < len(tweens) else MIN_TWEEN

        for s in range(nsub):
            t = s / float(max(1, nsub))
            images.append(draw_frame(st_a, st_b, ca, cb, t, i + 1))

    if images:
        last = draw_frame(states[-1], states[-1], centers_v[-1], centers_v[-1], 1.0, len(states) - 1)
        images.extend([last.copy() for _ in range(HOLD_LAST)])

        images[0].save(
            out,
            save_all=True,
            append_images=images[1:],
            duration=FRAME_MS,
            loop=0,
            optimize=False,
        )

    print(f"[ok] wrote {out} | frames={len(images)} | states={len(states)}")


if __name__ == "__main__":
    main()