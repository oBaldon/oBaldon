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

TARGET_TOTAL_FRAMES = 360
MIN_TWEEN = 6
MAX_TWEEN = 14
FRAME_MS = 65
HOLD_FIRST = 14
HOLD_LAST = 24

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
GRID = (38, 50, 66)

POINT = (56, 189, 248)
EDGE = (80, 92, 112)
EDGE_ACTIVE = (34, 197, 94)
CENTER = (245, 158, 11)

CLUSTER_COLORS = [
    (56, 189, 248),
    (34, 197, 94),
    (245, 158, 11),
    (239, 68, 68),
]


# =========================
# Estado
# =========================
@dataclass(frozen=True)
class Snap:
    Y: np.ndarray
    prev_Y: np.ndarray
    iteration: int
    neighbor_loss: float
    best_neighbor_loss: float
    spread: float
    best_spread: float
    trust_like: float
    note: str


# =========================
# Util
# =========================
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


def lerp_array(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return (1.0 - t) * a + t * b


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
# Dados em alta dimensão
# =========================
def gen_high_dim_data(rng: np.random.Generator, n_per_cluster: int = 40, d: int = 10):
    centers = np.array(
        [
            [-2.6, -1.6, 1.3, 0.0, 0.8, 0.0, -1.0, 0.0, 0.4, 0.0],
            [2.3, 1.7, -0.8, 0.0, -0.6, 0.0, 1.0, 0.0, -0.5, 0.0],
            [-1.8, 2.4, 0.0, 1.2, 0.0, -0.8, 0.0, 0.6, 0.0, -0.5],
            [2.4, -2.0, 0.0, -1.0, 0.0, 0.9, 0.0, -0.7, 0.0, 0.6],
        ],
        dtype=float,
    )
    centers = centers[:, :d]

    X_parts = []
    labels = []
    for i, c in enumerate(centers):
        cov_scale = 0.55 + 0.08 * i
        pts = rng.normal(0.0, cov_scale, size=(n_per_cluster, d)) + c
        pts[:, 2:] += 0.22 * np.sin(pts[:, :1] * (0.8 + 0.15 * i))
        X_parts.append(pts)
        labels.extend([i] * n_per_cluster)

    X = np.vstack(X_parts).astype(float)
    labels = np.array(labels, dtype=int)

    # leve embaralhamento para não agrupar por ordem
    perm = rng.permutation(X.shape[0])
    X = X[perm]
    labels = labels[perm]

    return X, labels


def pca_2d(X: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(axis=0, keepdims=True)
    cov = np.cov(Xc.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1][:2]
    V = eigvecs[:, idx]
    Y = Xc @ V
    return Y.astype(float)


def pairwise_sq_dists(X: np.ndarray) -> np.ndarray:
    x2 = np.sum(X * X, axis=1, keepdims=True)
    D = x2 + x2.T - 2.0 * (X @ X.T)
    return np.maximum(D, 0.0)


def knn_graph(X: np.ndarray, k: int = 8):
    D = pairwise_sq_dists(X)
    np.fill_diagonal(D, np.inf)
    idx = np.argsort(D, axis=1)[:, :k]
    return D, idx


def trust_like_score(knn_hd: np.ndarray, knn_2d: np.ndarray, k: int) -> float:
    score = np.mean([len(set(knn_hd[i]) & set(knn_2d[i])) / k for i in range(knn_hd.shape[0])])
    return float(score)


def build_embedding_snaps(rng: np.random.Generator):
    X, labels = gen_high_dim_data(rng, n_per_cluster=38, d=10)
    n = X.shape[0]
    k = 8

    D_hd, knn_hd = knn_graph(X, k=k)

    # inicialização compacta e um pouco desorganizada
    Y0 = pca_2d(X)
    Y0 *= 0.18
    Y0 += rng.normal(0.0, 0.10, size=Y0.shape)

    Y = Y0.copy()
    prev_Y = Y.copy()

    D_init = pairwise_sq_dists(Y)
    np.fill_diagonal(D_init, np.inf)
    knn_2d_init = np.argsort(D_init, axis=1)[:, :k]

    neighbor_loss0 = float(np.mean(np.take_along_axis(D_init, knn_hd, axis=1)))
    spread0 = float(np.mean(np.sqrt(np.sum((Y - Y.mean(axis=0)) ** 2, axis=1))))
    trust0 = trust_like_score(knn_hd, knn_2d_init, k)

    snaps: List[Snap] = [
        Snap(
            Y=Y.copy(),
            prev_Y=prev_Y.copy(),
            iteration=0,
            neighbor_loss=neighbor_loss0,
            best_neighbor_loss=neighbor_loss0,
            spread=spread0,
            best_spread=spread0,
            trust_like=trust0,
            note="A projeção começa compacta e desorganizada antes de preservar a vizinhança local.",
        )
    ]

    best_neighbor_loss = neighbor_loss0
    best_spread = spread0

    steps = 34
    lr = 0.070
    repulsion_w = 0.010
    center_pull = 0.008
    momentum = 0.86
    vel = np.zeros_like(Y)

    # arestas simétricas para desenhar / atrair
    edges = set()
    for i in range(n):
        for j in knn_hd[i]:
            a, b = sorted((int(i), int(j)))
            edges.add((a, b))
    edge_list = sorted(edges)

    for it in range(1, steps + 1):
        grad = np.zeros_like(Y)

        # atração entre vizinhos
        for i, j in edge_list:
            diff = Y[i] - Y[j]
            grad[i] += diff
            grad[j] -= diff

        # repulsão global suave
        for i in range(n):
            diff = Y[i] - Y
            d2 = np.sum(diff * diff, axis=1, keepdims=True) + 0.08
            inv = 1.0 / d2
            inv[i] = 0.0
            grad[i] += repulsion_w * np.sum(diff * inv, axis=0)

        # recentrar
        grad += center_pull * Y

        vel = momentum * vel - lr * grad
        Y = Y + vel

        # pequeno exagero inicial tipo "early separation"
        if it <= 8:
            Y *= 1.035

        # métricas
        D_2d = pairwise_sq_dists(Y)
        np.fill_diagonal(D_2d, np.inf)
        knn_2d = np.argsort(D_2d, axis=1)[:, :k]

        neighbor_loss = float(np.mean(np.take_along_axis(D_2d, knn_hd, axis=1)))
        spread = float(np.mean(np.sqrt(np.sum((Y - Y.mean(axis=0)) ** 2, axis=1))))
        trust = trust_like_score(knn_hd, knn_2d, k)

        best_neighbor_loss = min(best_neighbor_loss, neighbor_loss)
        best_spread = max(best_spread, spread)

        if it < 10:
            note = "Vizinhos próximos no espaço original passam a se aproximar também na projeção 2D."
        elif it < 22:
            note = "Os grupos se separam gradualmente enquanto a estrutura local é preservada."
        else:
            note = "A projeção estabiliza em 2D destacando a organização local dos dados."

        snaps.append(
            Snap(
                Y=Y.copy(),
                prev_Y=prev_Y.copy(),
                iteration=it,
                neighbor_loss=neighbor_loss,
                best_neighbor_loss=best_neighbor_loss,
                spread=spread,
                best_spread=best_spread,
                trust_like=trust,
                note=note,
            )
        )
        prev_Y = Y.copy()

    snaps.append(
        Snap(
            Y=snaps[-1].Y.copy(),
            prev_Y=snaps[-1].Y.copy(),
            iteration=snaps[-1].iteration,
            neighbor_loss=snaps[-1].neighbor_loss,
            best_neighbor_loss=snaps[-1].best_neighbor_loss,
            spread=snaps[-1].spread,
            best_spread=snaps[-1].best_spread,
            trust_like=snaps[-1].trust_like,
            note="A estrutura final em 2D evidencia grupos e continuidade local entre amostras semelhantes.",
        )
    )

    return X, labels, knn_hd, edge_list, snaps


# =========================
# Geometria do plot
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

    return Yn


# =========================
# Render
# =========================
def main() -> None:
    Path("assets").mkdir(exist_ok=True)
    out = Path("assets/tsne_umap.gif")

    seed = seed_from_today()
    rng = np.random.default_rng(seed)

    X, labels, knn_hd, edge_list, snaps = build_embedding_snaps(rng)

    all_Y = np.vstack([s.Y for s in snaps])
    transitions = len(snaps) - 1
    tweens = allocate_tweens(
        transitions,
        TARGET_TOTAL_FRAMES,
        HOLD_FIRST,
        HOLD_LAST,
        MIN_TWEEN,
        MAX_TWEEN,
    )

    inner_x, inner_y = CARD_PAD, CARD_PAD
    inner_w, inner_h = CANVAS_W - 2 * CARD_PAD, CANVAS_H - 2 * CARD_PAD

    plot_x0 = inner_x + LEFT_W + GAP
    plot_y0 = inner_y + 54
    plot_w = inner_w - LEFT_W - GAP - 18
    plot_h = inner_h - 76

    loss0 = float(snaps[0].neighbor_loss)
    loss_best = float(min(s.best_neighbor_loss for s in snaps))

    def draw_metrics(dr: ImageDraw.ImageDraw, x: int, y: int, st: Snap) -> int:
        dy = 22
        lines = [
            ("dimensão", "10D → 2D"),
            ("amostras", f"{X.shape[0]}"),
            ("iteração", f"{st.iteration}"),
            ("viz. loss", f"{st.neighbor_loss:.3f}"),
            ("melhor loss", f"{st.best_neighbor_loss:.3f}"),
            ("spread 2D", f"{st.spread:.3f}"),
            ("trust-like", f"{100.0 * st.trust_like:.1f}%"),
        ]
        for i, (k, v) in enumerate(lines):
            yy = y + i * dy
            dr.text((x, yy), k, fill=MUTED, font=FONT)
            dr.text((x + 138, yy), v, fill=TEXT, font=FONT)
        return y + len(lines) * dy

    def render_frame(st_a: Snap, st_b: Snap, t: float) -> Image.Image:
        Y = lerp_array(st_a.Y, st_b.Y, t)
        Yv = normalize_to_rect(Y, plot_x0, plot_y0, plot_w, plot_h, pad=24.0)

        neighbor_loss = lerp_scalar(st_a.neighbor_loss, st_b.neighbor_loss, t)
        best_neighbor_loss = min(st_a.best_neighbor_loss, st_b.best_neighbor_loss)
        spread = lerp_scalar(st_a.spread, st_b.spread, t)
        best_spread = max(st_a.best_spread, st_b.best_spread)
        trust_like = lerp_scalar(st_a.trust_like, st_b.trust_like, t)

        im = Image.new("RGB", (CANVAS_W, CANVAS_H), BG)
        dr = ImageDraw.Draw(im)

        dr.rounded_rectangle(
            [inner_x, inner_y, inner_x + inner_w, inner_y + inner_h],
            radius=16,
            fill=CARD,
            outline=STROKE,
            width=2,
        )

        title = "t-SNE / UMAP"
        dr.text((inner_x + 22, inner_y + 14), title, fill=TEXT, font=FONT)

        meta = f"projeção 2D animada | seed={seed}"
        meta_x = inner_x + 160
        meta = fit_text(dr, meta, inner_w - (meta_x - inner_x) - 22, font=FONT)
        dr.text((meta_x, inner_y + 16), meta, fill=MUTED, font=FONT)

        # barra monotônica baseada em melhora da estrutura local
        bar_x, bar_y, bar_w, bar_h = inner_x + 22, inner_y + 44, LEFT_W - 44, 12
        dr.rounded_rectangle([bar_x, bar_y, bar_x + bar_w, bar_y + bar_h], radius=8, fill=BAR_BG, outline=STROKE)
        prog = 1.0 - (best_neighbor_loss - loss_best) / (loss0 - loss_best + 1e-9)
        prog = float(np.clip(prog, 0.0, 1.0))
        if prog > 0:
            dr.rounded_rectangle([bar_x, bar_y, bar_x + int(bar_w * prog), bar_y + bar_h], radius=8, fill=BAR_FILL)

        left_x = inner_x + 22
        y = inner_y + 72
        y = draw_metrics(
            dr,
            left_x,
            y,
            Snap(
                Y=Y,
                prev_Y=st_a.prev_Y,
                iteration=st_b.iteration,
                neighbor_loss=neighbor_loss,
                best_neighbor_loss=best_neighbor_loss,
                spread=spread,
                best_spread=best_spread,
                trust_like=trust_like,
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

        # desenhar uma amostra pequena de arestas de vizinhança para comunicar preservação local
        edge_alpha = float(np.clip((st_b.iteration / max(1, snaps[-1].iteration)) * 1.15, 0.15, 0.55))
        shown_edges = edge_list[::4]
        for i, j in shown_edges:
            p1 = tuple(Yv[i])
            p2 = tuple(Yv[j])
            col = EDGE_ACTIVE if labels[i] == labels[j] else EDGE
            if labels[i] != labels[j]:
                col = EDGE
            dr.line([p1, p2], fill=col, width=1)

        # pontos
        for idx, (px, py) in enumerate(Yv):
            c = CLUSTER_COLORS[int(labels[idx]) % len(CLUSTER_COLORS)]
            dr.ellipse([px - 3, py - 3, px + 3, py + 3], fill=c)

        # centros dos grupos para reforço visual
        for lab in np.unique(labels):
            pts = Yv[labels == lab]
            cx, cy = pts.mean(axis=0)
            dr.ellipse([cx - 7, cy - 7, cx + 7, cy + 7], outline=CENTER, width=2)
            dr.line([(cx - 6, cy), (cx + 6, cy)], fill=CENTER, width=2)
            dr.line([(cx, cy - 6), (cx, cy + 6)], fill=CENTER, width=2)

        return im

    images: List[Image.Image] = []

    first = render_frame(snaps[0], snaps[0], 0.0)
    images.extend([first.copy() for _ in range(HOLD_FIRST)])

    for i in range(len(snaps) - 1):
        nsub = tweens[i] if i < len(tweens) else MIN_TWEEN
        for s in range(nsub):
            t = s / float(max(1, nsub))
            images.append(render_frame(snaps[i], snaps[i + 1], t))

    last = render_frame(snaps[-1], snaps[-1], 1.0)
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