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

GRID_H = 8
GRID_W = 8
N = GRID_H * GRID_W

TARGET_TOTAL_FRAMES = 268
MIN_TWEEN = 3
MAX_TWEEN = 8
FRAME_MS = 55
HOLD_FIRST = 10
HOLD_LAST = 18

MAX_SWEEPS = 9
SNAP_EVERY_UPDATES = 4

CARD_PAD = 18
LEFT_W = 352
GAP = 18

BG = (11, 18, 32)
CARD = (15, 23, 42)
STROKE = (31, 41, 55)
TEXT = (229, 231, 235)
MUTED = (148, 163, 184)
BAR_BG = (17, 24, 39)
BAR_FILL = (34, 197, 94)

C_ON = (37, 99, 235)
C_OFF = (30, 41, 59)
C_TARGET = (34, 197, 94)
C_NOISY = (245, 158, 11)
C_CHANGED = (239, 68, 68)
C_GRID = (51, 65, 85)
C_SPARK = (56, 189, 248)


@dataclass(frozen=True)
class Snap:
    state: np.ndarray
    prev_state: np.ndarray
    changed_mask: np.ndarray
    sweep: int
    update_idx: int
    energy: float
    best_energy: float
    overlap: float
    best_overlap: float
    stable_ratio: float
    corrected_ratio: float


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
    return out


# =========================
# Padrões
# =========================
def arr(rows: List[str]) -> np.ndarray:
    return np.array([[1 if c == "#" else -1 for c in r] for r in rows], dtype=int)


def base_patterns() -> List[np.ndarray]:
    return [
        arr(
            [
                "..####..",
                ".######.",
                "##....##",
                "##....##",
                "##....##",
                "##....##",
                ".######.",
                "..####..",
            ]
        ),
        arr(
            [
                "##....##",
                ".##..##.",
                "..####..",
                "...##...",
                "...##...",
                "..####..",
                ".##..##.",
                "##....##",
            ]
        ),
        arr(
            [
                "...##...",
                "...##...",
                "...##...",
                "########",
                "########",
                "...##...",
                "...##...",
                "...##...",
            ]
        ),
    ]


def flatten(g: np.ndarray) -> np.ndarray:
    return g.reshape(-1).astype(int)


def gridify(v: np.ndarray) -> np.ndarray:
    return v.reshape(GRID_H, GRID_W)


def hebbian(patterns: List[np.ndarray]) -> np.ndarray:
    X = np.stack(patterns).astype(float)
    W = (X.T @ X) / float(N)
    np.fill_diagonal(W, 0.0)
    return W


def energy(s: np.ndarray, W: np.ndarray) -> float:
    return float(-0.5 * s @ W @ s)


def overlap(s: np.ndarray, t: np.ndarray) -> float:
    return float((s * t).mean())


def corrupt(target: np.ndarray, ratio: float, rng: np.random.Generator) -> np.ndarray:
    s = target.copy()
    n = max(1, int(round(len(s) * ratio)))
    idx = rng.choice(len(s), n, replace=False)
    s[idx] *= -1
    return s


# =========================
# Dinâmica
# =========================
def build_snaps(rng: np.random.Generator):
    patterns = [flatten(p) for p in base_patterns()]
    target_idx = int(rng.integers(0, len(patterns)))
    target = patterns[target_idx]
    W = hebbian(patterns)

    noisy = corrupt(target, 0.25, rng)
    state = noisy.copy()
    prev = state.copy()

    e0 = energy(state, W)
    o0 = overlap(state, target)

    snaps: List[Snap] = [
        Snap(
            state=state.copy(),
            prev_state=prev.copy(),
            changed_mask=np.zeros_like(state, dtype=bool),
            sweep=0,
            update_idx=0,
            energy=e0,
            best_energy=e0,
            overlap=o0,
            best_overlap=o0,
            stable_ratio=float((state == target).mean()),
            corrected_ratio=float(((state == target) & (noisy != target)).mean()),
        )
    ]

    best_e = e0
    best_o = o0
    updates = 0

    for sweep in range(1, MAX_SWEEPS + 1):
        order = rng.permutation(N)
        changed_since_snap = np.zeros(N, dtype=bool)
        any_change = False

        for i, j in enumerate(order, start=1):
            updates += 1
            old = int(state[j])
            h = float(W[j] @ state)
            new = 1 if h >= 0 else -1
            state[j] = new

            if new != old:
                changed_since_snap[j] = True
                any_change = True

            if (i % SNAP_EVERY_UPDATES == 0) or (i == N):
                e = energy(state, W)
                o = overlap(state, target)
                best_e = min(best_e, e)
                best_o = max(best_o, o)

                snaps.append(
                    Snap(
                        state=state.copy(),
                        prev_state=prev.copy(),
                        changed_mask=changed_since_snap.copy(),
                        sweep=sweep,
                        update_idx=updates,
                        energy=e,
                        best_energy=best_e,
                        overlap=o,
                        best_overlap=best_o,
                        stable_ratio=float((state == target).mean()),
                        corrected_ratio=float(((state == target) & (noisy != target)).mean()),
                    )
                )
                prev = state.copy()
                changed_since_snap[:] = False

        if not any_change:
            break

    snaps.append(
        Snap(
            state=snaps[-1].state.copy(),
            prev_state=snaps[-1].state.copy(),
            changed_mask=np.zeros_like(snaps[-1].state, dtype=bool),
            sweep=snaps[-1].sweep,
            update_idx=snaps[-1].update_idx,
            energy=snaps[-1].energy,
            best_energy=snaps[-1].best_energy,
            overlap=snaps[-1].overlap,
            best_overlap=snaps[-1].best_overlap,
            stable_ratio=snaps[-1].stable_ratio,
            corrected_ratio=snaps[-1].corrected_ratio,
        )
    )

    return snaps, noisy, target, target_idx


# =========================
# Desenho
# =========================
def draw_grid(
    dr: ImageDraw.ImageDraw,
    grid: np.ndarray,
    x0: int,
    y0: int,
    cell: int,
    on_color: Tuple[int, int, int],
    off_color: Tuple[int, int, int],
    border: Tuple[int, int, int],
    changed: np.ndarray | None = None,
    label: str | None = None,
) -> None:
    if label:
        dr.text((x0, y0 - 18), label, fill=MUTED, font=FONT_SMALL)

    changed2 = changed.reshape(GRID_H, GRID_W) if changed is not None else None

    for r in range(GRID_H):
        for c in range(GRID_W):
            x1 = x0 + c * cell
            y1 = y0 + r * cell
            x2 = x1 + cell - 1
            y2 = y1 + cell - 1

            fill = on_color if grid[r, c] > 0 else off_color
            dr.rounded_rectangle([x1, y1, x2, y2], radius=3, fill=fill, outline=border)

            if changed2 is not None and bool(changed2[r, c]):
                dr.rounded_rectangle(
                    [x1 + 1, y1 + 1, x2 - 1, y2 - 1],
                    radius=3,
                    outline=C_CHANGED,
                    width=2,
                )


def draw_spark(dr: ImageDraw.ImageDraw, values: List[float], x0: int, y0: int, w: int, h: int) -> None:
    dr.rounded_rectangle([x0, y0, x0 + w, y0 + h], radius=10, fill=BAR_BG, outline=STROKE)

    if len(values) < 2:
        return

    vmin = min(values)
    vmax = max(values)
    if abs(vmax - vmin) < 1e-9:
        vmax = vmin + 1.0

    def pt(i: int, v: float) -> Tuple[int, int]:
        x = x0 + int(round((w - 8) * i / max(1, len(values) - 1))) + 4
        y = y0 + h - 4 - int(round((h - 8) * (v - vmin) / (vmax - vmin)))
        return x, y

    for i in range(1, len(values)):
        dr.line([pt(i - 1, values[i - 1]), pt(i, values[i])], fill=C_SPARK, width=2)


# =========================
# Main
# =========================
def main() -> None:
    Path("assets").mkdir(exist_ok=True)
    out = Path("assets/hopfield.gif")

    seed = seed_from_today()
    rng = np.random.default_rng(seed)

    snaps, noisy, target, target_idx = build_snaps(rng)
    energies = [s.energy for s in snaps]

    o0 = float(snaps[0].overlap)
    obest = float(max(s.best_overlap for s in snaps))

    inner_x, inner_y = CARD_PAD, CARD_PAD
    inner_w, inner_h = CANVAS_W - 2 * CARD_PAD, CANVAS_H - 2 * CARD_PAD

    plot_x0 = inner_x + LEFT_W + GAP
    plot_y0 = inner_y + 54
    plot_w = inner_w - LEFT_W - GAP - 18
    plot_h = inner_h - 76

    big_cell = 22
    mini_cell = 10

    big_w = GRID_W * big_cell
    big_h = GRID_H * big_cell

    big_x = plot_x0 + 10
    big_y = plot_y0 + 20

    mini_x = big_x + big_w + 18
    mini_y = plot_y0 + 20
    mini_w = GRID_W * mini_cell

    target_x = mini_x + mini_w + 30

    spark_x = mini_x
    spark_y = plot_y0 + plot_h - 74
    spark_w = plot_x0 + plot_w - spark_x - 8
    spark_h = 60

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
            ("padrão alvo", f"{target_idx + 1}/3"),
            ("sweep", f"{st.sweep}"),
            ("updates", f"{st.update_idx}/{MAX_SWEEPS * N}"),
            ("energia", f"{st.energy:.2f}"),
            ("melhor energia", f"{st.best_energy:.2f}"),
            ("overlap", f"{st.overlap:.3f}"),
            ("melhor overlap", f"{st.best_overlap:.3f}"),
            ("bits corretos", f"{100.0 * st.stable_ratio:.1f}%"),
            ("ruído corrigido", f"{100.0 * st.corrected_ratio:.1f}%"),
        ]
        for i, (k, v) in enumerate(lines):
            yy = y + i * dy
            dr.text((x, yy), k, fill=MUTED, font=FONT)
            dr.text((x + 138, yy), v, fill=TEXT, font=FONT)
        return y + len(lines) * dy

    def frame(sa: Snap, sb: Snap, t: float, idx: int) -> Image.Image:
        state = sa.state if t < 0.5 else sb.state
        changed = sb.changed_mask if t > 0.30 else np.zeros_like(sb.changed_mask)

        e = lerp_scalar(sa.energy, sb.energy, t)
        o = lerp_scalar(sa.overlap, sb.overlap, t)
        be = min(sa.best_energy, sb.best_energy)
        bo = max(sa.best_overlap, lerp_scalar(sa.best_overlap, sb.best_overlap, t))
        stable = lerp_scalar(sa.stable_ratio, sb.stable_ratio, t)
        corrected = lerp_scalar(sa.corrected_ratio, sb.corrected_ratio, t)

        im = Image.new("RGB", (CANVAS_W, CANVAS_H), BG)
        dr = ImageDraw.Draw(im)

        dr.rounded_rectangle(
            [inner_x, inner_y, inner_x + inner_w, inner_y + inner_h],
            radius=16,
            fill=CARD,
            outline=STROKE,
            width=2,
        )

        title = "Hopfield Network"
        dr.text((inner_x + 22, inner_y + 14), title, fill=TEXT, font=FONT)

        meta = f"recuperação de padrões | grade={GRID_H}x{GRID_W} | seed={seed}"
        meta_x = inner_x + 200
        meta = fit_text(dr, meta, inner_w - (meta_x - inner_x) - 22, font=FONT)
        dr.text((meta_x, inner_y + 16), meta, fill=MUTED, font=FONT)

        bar_x, bar_y, bar_w, bar_h = inner_x + 22, inner_y + 44, LEFT_W - 44, 12
        dr.rounded_rectangle([bar_x, bar_y, bar_x + bar_w, bar_y + bar_h], radius=8, fill=BAR_BG, outline=STROKE)

        prog = (bo - o0) / (obest - o0 + 1e-9)
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
                state=state,
                prev_state=sa.prev_state,
                changed_mask=changed,
                sweep=sb.sweep,
                update_idx=sb.update_idx,
                energy=e,
                best_energy=be,
                overlap=o,
                best_overlap=bo,
                stable_ratio=stable,
                corrected_ratio=corrected,
            ),
        )

        # bloco curto no lugar da interpretação longa
        y += 8
        dr.text((left_x, y), "dinâmica", fill=MUTED, font=FONT)
        y += 22
        short_note = "A rede reduz energia e reconstrói o padrão."
        short_note = fit_text(dr, short_note, LEFT_W - 44, font=FONT_SMALL)
        dr.text((left_x, y), short_note, fill=TEXT, font=FONT_SMALL)

        dr.rounded_rectangle(
            [plot_x0 - 12, plot_y0 - 12, plot_x0 + plot_w + 12, plot_y0 + plot_h + 12],
            radius=14,
            fill=(11, 18, 32),
            outline=STROKE,
        )

        draw_grid(
            dr,
            gridify(state),
            big_x,
            big_y,
            big_cell,
            C_ON,
            C_OFF,
            C_GRID,
            changed,
            "estado atual",
        )

        draw_grid(
            dr,
            gridify(noisy),
            mini_x,
            mini_y,
            mini_cell,
            C_NOISY,
            C_OFF,
            C_GRID,
            None,
            "entrada",
        )

        draw_grid(
            dr,
            gridify(target),
            target_x,
            mini_y,
            mini_cell,
            C_TARGET,
            C_OFF,
            C_GRID,
            None,
            "alvo",
        )

        dr.text((spark_x, spark_y - 16), "energia do estado", fill=MUTED, font=FONT_SMALL)
        draw_spark(dr, energies[: idx + 1], spark_x, spark_y, spark_w, spark_h)

        return im

    images: List[Image.Image] = []

    first = frame(snaps[0], snaps[0], 0.0, 0)
    images.extend([first.copy() for _ in range(HOLD_FIRST)])

    for i in range(len(snaps) - 1):
        n = tweens[i] if i < len(tweens) else MIN_TWEEN
        for s in range(n):
            t = s / float(max(1, n))
            images.append(frame(snaps[i], snaps[i + 1], t, i + 1))

    last = frame(snaps[-1], snaps[-1], 1.0, len(snaps) - 1)
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