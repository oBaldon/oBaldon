#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# =========================
# Configuração
# =========================
CANVAS_W, CANVAS_H = 980, 360
FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)

N_TOTAL = 320
MAX_DEPTH = 4
MIN_SAMPLES_SPLIT = 18
MIN_SAMPLES_LEAF = 8

TWEEN_PER_SPLIT = 8
FRAME_MS = 55
HOLD_LAST = 16

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

# Fundo das regiões
RG_NEG = (55, 22, 28)
RG_POS = (18, 35, 62)

C_SPLIT = (245, 158, 11)
C_SPLIT_NEW = (34, 197, 94)


@dataclass(frozen=True)
class Region:
    x0: float
    x1: float
    y0: float
    y1: float
    pred: int


@dataclass(frozen=True)
class SplitViz:
    feature: int       # 0=x, 1=y
    threshold: float
    x0: float
    x1: float
    y0: float
    y1: float
    depth: int


@dataclass(frozen=True)
class Snap:
    regions: List[Region]
    splits: List[SplitViz]
    depth: int
    leaves: int
    acc: float


@dataclass
class Node:
    idx: np.ndarray
    x0: float
    x1: float
    y0: float
    y1: float
    depth: int
    pred: int
    feature: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["Node"] = None
    right: Optional["Node"] = None


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


def gen_dataset(rng: np.random.Generator, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Dataset 2D com padrão adequado para árvore:
    combinação de regras retangulares/eixo-alinhadas.
    """
    X = rng.uniform(-3.2, 3.2, size=(n, 2)).astype(float)
    x = X[:, 0]
    y = X[:, 1]

    # Regra em blocos, ideal para árvore axis-aligned
    y_cls = np.where(
        ((x < -0.7) & (y > 0.2)) |
        ((x > 1.1) & (y < -0.4)) |
        ((x > -0.2) & (x < 1.4) & (y > 1.2)),
        1, -1
    ).astype(int)

    # Ruído pequeno
    flip = rng.random(n) < 0.06
    y_cls[flip] *= -1
    return X, y_cls


def gini_of(y: np.ndarray) -> float:
    if y.size == 0:
        return 0.0
    p1 = np.mean(y == 1)
    p0 = 1.0 - p1
    return 1.0 - p1 * p1 - p0 * p0


def best_split(X: np.ndarray, y: np.ndarray, idx: np.ndarray) -> Tuple[Optional[int], Optional[float], float]:
    """
    Busca exaustiva simples em thresholds candidatos dos próprios pontos.
    """
    Xi = X[idx]
    yi = y[idx]
    n = yi.size
    if n < MIN_SAMPLES_SPLIT:
        return None, None, 0.0

    parent_g = gini_of(yi)
    best_gain = 0.0
    best_feat = None
    best_thr = None

    for feat in (0, 1):
        vals = Xi[:, feat]
        uniq = np.unique(vals)
        if uniq.size < 2:
            continue
        thr_cands = (uniq[:-1] + uniq[1:]) / 2.0

        for thr in thr_cands:
            left = vals <= thr
            right = ~left
            nl = int(left.sum())
            nr = int(right.sum())
            if nl < MIN_SAMPLES_LEAF or nr < MIN_SAMPLES_LEAF:
                continue

            gl = gini_of(yi[left])
            gr = gini_of(yi[right])
            child_g = (nl / n) * gl + (nr / n) * gr
            gain = parent_g - child_g

            if gain > best_gain:
                best_gain = gain
                best_feat = feat
                best_thr = float(thr)

    return best_feat, best_thr, float(best_gain)


def majority_label(y: np.ndarray) -> int:
    return 1 if np.sum(y == 1) >= np.sum(y == -1) else -1


def build_tree_and_snaps(X: np.ndarray, y: np.ndarray) -> List[Snap]:
    xmin, ymin = float(X[:, 0].min()), float(X[:, 1].min())
    xmax, ymax = float(X[:, 0].max()), float(X[:, 1].max())

    root_idx = np.arange(X.shape[0])
    root = Node(
        idx=root_idx,
        x0=xmin, x1=xmax, y0=ymin, y1=ymax,
        depth=0,
        pred=majority_label(y[root_idx]),
    )

    leaves: List[Node] = [root]
    splits_viz: List[SplitViz] = []
    snaps: List[Snap] = []

    def predict_from_leaves() -> np.ndarray:
        pred = np.empty(X.shape[0], dtype=int)
        for lf in leaves:
            pred[lf.idx] = lf.pred
        return pred

    def current_regions() -> List[Region]:
        return [
            Region(lf.x0, lf.x1, lf.y0, lf.y1, lf.pred)
            for lf in leaves
        ]

    def push_snap(depth: int) -> None:
        pred = predict_from_leaves()
        acc = float(np.mean(pred == y))
        snaps.append(
            Snap(
                regions=current_regions(),
                splits=splits_viz.copy(),
                depth=depth,
                leaves=len(leaves),
                acc=acc,
            )
        )

    push_snap(depth=0)

    while True:
        # escolhe melhor folha para split globalmente
        best_leaf_i = None
        best_feat = None
        best_thr = None
        best_gain = 0.0

        for i, lf in enumerate(leaves):
            if lf.depth >= MAX_DEPTH:
                continue
            feat, thr, gain = best_split(X, y, lf.idx)
            if feat is not None and gain > best_gain:
                best_gain = gain
                best_feat = feat
                best_thr = thr
                best_leaf_i = i

        if best_leaf_i is None:
            break

        lf = leaves[best_leaf_i]
        vals = X[lf.idx, best_feat]
        left_mask = vals <= best_thr
        right_mask = ~left_mask

        left_idx = lf.idx[left_mask]
        right_idx = lf.idx[right_mask]

        if left_idx.size < MIN_SAMPLES_LEAF or right_idx.size < MIN_SAMPLES_LEAF:
            break

        if best_feat == 0:
            left_box = (lf.x0, best_thr, lf.y0, lf.y1)
            right_box = (best_thr, lf.x1, lf.y0, lf.y1)
            split_v = SplitViz(
                feature=0, threshold=float(best_thr),
                x0=float(best_thr), x1=float(best_thr),
                y0=lf.y0, y1=lf.y1,
                depth=lf.depth + 1
            )
        else:
            left_box = (lf.x0, lf.x1, lf.y0, best_thr)
            right_box = (lf.x0, lf.x1, best_thr, lf.y1)
            split_v = SplitViz(
                feature=1, threshold=float(best_thr),
                x0=lf.x0, x1=lf.x1,
                y0=float(best_thr), y1=float(best_thr),
                depth=lf.depth + 1
            )

        left_node = Node(
            idx=left_idx,
            x0=left_box[0], x1=left_box[1], y0=left_box[2], y1=left_box[3],
            depth=lf.depth + 1,
            pred=majority_label(y[left_idx])
        )
        right_node = Node(
            idx=right_idx,
            x0=right_box[0], x1=right_box[1], y0=right_box[2], y1=right_box[3],
            depth=lf.depth + 1,
            pred=majority_label(y[right_idx])
        )

        # substitui a folha
        leaves.pop(best_leaf_i)
        leaves.append(left_node)
        leaves.append(right_node)
        splits_viz.append(split_v)

        push_snap(depth=max(nd.depth for nd in leaves))

    return snaps


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


def box_to_plot(
    x0: float, x1: float, y0: float, y1: float,
    tf: Tuple[float, float, float, float, float, float, float],
) -> Tuple[float, float, float, float]:
    p1 = np.array([x0, y0], dtype=float)
    p2 = np.array([x1, y1], dtype=float)
    ax, ay = to_plot(p1, tf)
    bx, by = to_plot(p2, tf)
    left = min(ax, bx)
    right = max(ax, bx)
    top = min(ay, by)
    bottom = max(ay, by)
    return left, top, right, bottom


def lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return (1.0 - t) * a + t * b


def main() -> None:
    Path("assets").mkdir(parents=True, exist_ok=True)
    out = Path("assets/decision_tree.gif")

    seed = seed_from_today()
    rng = np.random.default_rng(seed)

    inner_x, inner_y = CARD_PAD, CARD_PAD
    inner_w, inner_h = CANVAS_W - 2 * CARD_PAD, CANVAS_H - 2 * CARD_PAD

    plot_x0 = inner_x + LEFT_W + GAP
    plot_y0 = inner_y + 22
    plot_w = inner_w - LEFT_W - GAP - 18
    plot_h = inner_h - 44

    X, y = gen_dataset(rng, N_TOTAL)
    Xv, tf = normalize_to_rect(X, plot_x0, plot_y0, plot_w, plot_h, pad=24.0)

    snaps = build_tree_and_snaps(X, y)

    acc0 = float(snaps[0].acc)
    accN = float(max(s.acc for s in snaps))

    images: List[Image.Image] = []

    # segura o estado inicial um pouco
    HOLD_FIRST = 8

    def draw_snap(base: Snap, next_snap: Optional[Snap], split_t: float) -> Image.Image:
        im = Image.new("RGB", (CANVAS_W, CANVAS_H), BG)
        dr = ImageDraw.Draw(im)

        dr.rounded_rectangle(
            [inner_x, inner_y, inner_x + inner_w, inner_y + inner_h],
            radius=16, fill=CARD, outline=STROKE, width=2
        )

        dr.text((inner_x + 22, inner_y + 16), "Decision Tree (CART-style splits)", fill=TEXT, font=FONT)

        left_text_max_w = int(LEFT_W - 44)
        meta = f"depth={base.depth}/{MAX_DEPTH} | folhas={base.leaves} | seed={seed}"
        dr.text((inner_x + 22, inner_y + 36), fit_text(dr, meta, left_text_max_w), fill=MUTED, font=FONT)

        # barra acurácia
        bar_x, bar_y, bar_w, bar_h = inner_x + 22, inner_y + 56, LEFT_W - 44, 12
        dr.rounded_rectangle([bar_x, bar_y, bar_x + bar_w, bar_y + bar_h],
                             radius=8, fill=BAR_BG, outline=STROKE)
        denom = (accN - acc0) if abs(accN - acc0) > 1e-12 else 1.0
        prog = (base.acc - acc0) / denom
        prog = max(0.0, min(1.0, prog))
        dr.rounded_rectangle([bar_x, bar_y, bar_x + int(bar_w * prog), bar_y + bar_h],
                             radius=8, fill=BAR_FILL)

        y_cursor = inner_y + 78
        dr.text((inner_x + 22, y_cursor), fit_text(dr, f"accuracy: {base.acc*100:5.1f}%", left_text_max_w),
                fill=MUTED, font=FONT)
        y_cursor += 20

        desc = "visualização: regiões de decisão retangulares criadas por splits eixo-alinhados"
        for ln in wrap_text(dr, desc, left_text_max_w):
            dr.text((inner_x + 22, y_cursor), ln, fill=(120, 135, 155), font=FONT)
            y_cursor += 18
        y_cursor += 6

        lx, ly = inner_x + 22, y_cursor + 8
        dr.rectangle([lx, ly - 10, lx + 12, ly + 2], fill=C_POS)
        dr.text((lx + 18, ly - 12), "classe +1", fill=MUTED, font=FONT)
        dr.rectangle([lx, ly + 18 - 10, lx + 12, ly + 18 + 2], fill=C_NEG)
        dr.text((lx + 18, ly + 18 - 12), "classe -1", fill=MUTED, font=FONT)
        y_cursor = ly + 44

        dr.text((inner_x + 22, y_cursor), "linha laranja: splits existentes", fill=(120, 135, 155), font=FONT)
        y_cursor += 18
        dr.text((inner_x + 22, y_cursor), "linha verde: novo split", fill=(120, 135, 155), font=FONT)

        # Plot BG
        dr.rounded_rectangle(
            [plot_x0 - 12, plot_y0 - 12, plot_x0 + plot_w + 12, plot_y0 + plot_h + 12],
            radius=14, fill=(11, 18, 32), outline=STROKE
        )

        # Regiões de decisão
        for rg in base.regions:
            l, t, r, b = box_to_plot(rg.x0, rg.x1, rg.y0, rg.y1, tf)
            fill = RG_POS if rg.pred == 1 else RG_NEG
            dr.rectangle([l, t, r, b], fill=fill)

        # Splits já consolidados
        base_splits = base.splits if next_snap is not None else base.splits
        for sp in base_splits:
            if sp.feature == 0:
                x1, y1 = to_plot(np.array([sp.x0, sp.y0]), tf)
                x2, y2 = to_plot(np.array([sp.x1, sp.y1]), tf)
            else:
                x1, y1 = to_plot(np.array([sp.x0, sp.y0]), tf)
                x2, y2 = to_plot(np.array([sp.x1, sp.y1]), tf)
            dr.line([x1, y1, x2, y2], fill=C_SPLIT, width=2)

        # Novo split sendo animado
        if next_snap is not None and len(next_snap.splits) > len(base.splits):
            sp = next_snap.splits[-1]
            if sp.feature == 0:
                xa, ya = to_plot(np.array([sp.x0, sp.y0]), tf)
                xb, yb = to_plot(np.array([sp.x1, sp.y1]), tf)
                ym = ya + (yb - ya) * split_t
                dr.line([xa, ya, xb, ym], fill=C_SPLIT_NEW, width=3)
            else:
                xa, ya = to_plot(np.array([sp.x0, sp.y0]), tf)
                xb, yb = to_plot(np.array([sp.x1, sp.y1]), tf)
                xm = xa + (xb - xa) * split_t
                dr.line([xa, ya, xm, yb], fill=C_SPLIT_NEW, width=3)

        # Pontos
        r = 3
        for i, (px, py) in enumerate(Xv):
            col = C_POS if y[i] == 1 else C_NEG
            dr.ellipse([px - r, py - r, px + r, py + r], fill=col)

        footer = f"folhas={base.leaves} | profundidade={base.depth}"
        dr.text((inner_x + 22, inner_y + inner_h - 26), fit_text(dr, footer, int(inner_w - 44)),
                fill=(100, 116, 139), font=FONT)

        return im

    # estado inicial
    first = draw_snap(snaps[0], None, 0.0)
    images.append(first)
    for _ in range(HOLD_FIRST):
        images.append(first)

    # transições entre snapshots
    for i in range(len(snaps) - 1):
        a = snaps[i]
        b = snaps[i + 1]
        for t_i in range(TWEEN_PER_SPLIT):
            t = t_i / float(TWEEN_PER_SPLIT)
            images.append(draw_snap(a, b, t))

    # último estado
    last = draw_snap(snaps[-1], None, 1.0)
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