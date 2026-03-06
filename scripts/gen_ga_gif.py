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

RANGE = 5.12

# GA
POP = 120
GENERATIONS = 26
ELITE = 6
TOURNAMENT_K = 3
CROSS_RATE = 0.92
MUT_RATE = 0.22

SIGMA0 = 0.55
SIGMA_MIN = 0.10

# Campo
GRID_W = 240
GRID_H = 170
CONTOUR_LEVELS = 11

# GIF pacing (principal mudança)
TWEEN_PER_GEN = 20      # <<< aumenta duração/legibilidade (6..12 é bom)
HOLD_GEN0 = 10         # frames extras no começo
FRAME_MS = 55
HOLD_LAST = 14
TRAIL = 50             # rastro do best-so-far

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

C_CONTOUR = (65, 86, 125)
C_POP = (56, 189, 248)
C_BEST = (245, 158, 11)
C_BEST_SO_FAR = (34, 197, 94)


@dataclass(frozen=True)
class Snap:
    pop: np.ndarray
    fit: np.ndarray
    best: np.ndarray
    best_fit: float
    best_so_far: np.ndarray
    best_so_far_fit: float
    gen: int


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
# Rastrigin + fitness
# =========================
def rastrigin(xy: np.ndarray) -> np.ndarray:
    x = xy[..., 0]
    y = xy[..., 1]
    A = 10.0
    return 2 * A + (x * x - A * np.cos(2 * np.pi * x)) + (y * y - A * np.cos(2 * np.pi * y))


def fitness(xy: np.ndarray) -> np.ndarray:
    return (-rastrigin(xy)).astype(float)


def clip_domain(xy: np.ndarray) -> np.ndarray:
    return np.clip(xy, -RANGE, RANGE)


def tournament_select(rng: np.random.Generator, pop: np.ndarray, fit: np.ndarray, k: int) -> np.ndarray:
    idx = rng.integers(0, pop.shape[0], size=(k,))
    best = idx[np.argmax(fit[idx])]
    return pop[best].copy()


def blx_alpha_crossover(
    rng: np.random.Generator, p1: np.ndarray, p2: np.ndarray, alpha: float = 0.45
) -> Tuple[np.ndarray, np.ndarray]:
    lo = np.minimum(p1, p2)
    hi = np.maximum(p1, p2)
    d = hi - lo
    a = lo - alpha * d
    b = hi + alpha * d
    c1 = rng.uniform(a, b)
    c2 = rng.uniform(a, b)
    return c1.astype(float), c2.astype(float)


def mutate_gaussian(rng: np.random.Generator, x: np.ndarray, sigma: float, p: float) -> np.ndarray:
    m = rng.random(size=x.shape) < p
    noise = rng.normal(0.0, sigma, size=x.shape)
    y = x + m * noise
    return y.astype(float)


def make_snaps(rng: np.random.Generator) -> List[Snap]:
    pop = rng.uniform(-RANGE, RANGE, size=(POP, 2)).astype(float)
    fit = fitness(pop)

    best_idx = int(np.argmax(fit))
    best = pop[best_idx].copy()
    best_fit = float(fit[best_idx])

    best_so_far = best.copy()
    best_so_far_fit = best_fit

    snaps: List[Snap] = [
        Snap(pop=pop.copy(), fit=fit.copy(), best=best.copy(), best_fit=best_fit,
             best_so_far=best_so_far.copy(), best_so_far_fit=best_so_far_fit, gen=0)
    ]

    for g in range(1, GENERATIONS + 1):
        # annealing da mutação
        t = (g - 1) / max(1, (GENERATIONS - 1))
        sigma = max(SIGMA_MIN, SIGMA0 * (1.0 - 0.65 * t))

        elite_idx = np.argsort(-fit)[:ELITE]
        new_pop = [pop[i].copy() for i in elite_idx]

        while len(new_pop) < POP:
            p1 = tournament_select(rng, pop, fit, TOURNAMENT_K)
            p2 = tournament_select(rng, pop, fit, TOURNAMENT_K)

            if rng.random() < CROSS_RATE:
                c1, c2 = blx_alpha_crossover(rng, p1, p2, alpha=0.45)
            else:
                c1, c2 = p1.copy(), p2.copy()

            c1 = mutate_gaussian(rng, c1, sigma=sigma, p=MUT_RATE)
            c2 = mutate_gaussian(rng, c2, sigma=sigma, p=MUT_RATE)

            new_pop.append(clip_domain(c1))
            if len(new_pop) < POP:
                new_pop.append(clip_domain(c2))

        pop = np.vstack(new_pop).astype(float)
        fit = fitness(pop)

        best_idx = int(np.argmax(fit))
        best = pop[best_idx].copy()
        best_fit = float(fit[best_idx])

        if best_fit > best_so_far_fit:
            best_so_far_fit = best_fit
            best_so_far = best.copy()

        snaps.append(
            Snap(pop=pop.copy(), fit=fit.copy(), best=best.copy(), best_fit=best_fit,
                 best_so_far=best_so_far.copy(), best_so_far_fit=best_so_far_fit, gen=g)
        )

    return snaps


def lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return (1.0 - t) * a + t * b


def main() -> None:
    Path("assets").mkdir(parents=True, exist_ok=True)
    out = Path("assets/genetic.gif")

    seed = seed_from_today()
    rng = np.random.default_rng(seed)

    snaps = make_snaps(rng)

    # Layout
    inner_x, inner_y = CARD_PAD, CARD_PAD
    inner_w, inner_h = CANVAS_W - 2 * CARD_PAD, CANVAS_H - 2 * CARD_PAD

    plot_x0 = inner_x + LEFT_W + GAP
    plot_y0 = inner_y + 22
    plot_w = inner_w - LEFT_W - GAP - 18
    plot_h = inner_h - 44

    px0, py0 = plot_x0 + 24, plot_y0 + 24
    pw, ph = plot_w - 48, plot_h - 48

    def to_px(xy: np.ndarray) -> Tuple[float, float]:
        x = (xy[0] + RANGE) / (2.0 * RANGE)
        y = (xy[1] + RANGE) / (2.0 * RANGE)
        px = px0 + x * pw
        py = py0 + (1.0 - y) * ph
        return float(px), float(py)

    # Campo (heatmap + contornos)
    xs = np.linspace(-RANGE, RANGE, GRID_W)
    ys = np.linspace(-RANGE, RANGE, GRID_H)
    field = np.zeros((GRID_H, GRID_W), dtype=float)
    for j, yy in enumerate(ys):
        for i, xx in enumerate(xs):
            field[j, i] = rastrigin(np.array([xx, yy], dtype=float))

    fmin, fmax = float(field.min()), float(field.max())
    if fmax - fmin < 1e-12:
        fmax = fmin + 1.0

    fld = np.log(field - fmin + 1e-6)
    fld_min, fld_max = float(fld.min()), float(fld.max())
    if fld_max - fld_min < 1e-12:
        fld_max = fld_min + 1.0
    levels = np.linspace(float(fld.min()), float(fld.max()), CONTOUR_LEVELS)

    fldn = (fld - fld_min) / (fld_max - fld_min)
    hm = np.zeros((GRID_H, GRID_W, 3), dtype=np.uint8)
    inten = (1.0 - fldn)
    base = (12 + inten * 28).astype(np.uint8)
    hm[..., 0] = base
    hm[..., 1] = (base + 7).clip(0, 255)
    hm[..., 2] = (base + 16).clip(0, 255)
    heat_img = Image.fromarray(hm, mode="RGB").resize((int(pw), int(ph)), resample=Image.Resampling.BILINEAR)

    left_text_max_w = int(LEFT_W - 44)

    # Barra: normaliza por best-so-far (monótona)
    best0 = float(snaps[0].best_so_far_fit)
    bestN = float(max(s.best_so_far_fit for s in snaps))
    denom = (bestN - best0) if abs(bestN - best0) > 1e-12 else 1.0

    images: List[Image.Image] = []
    best_trail: List[Tuple[float, float]] = []

    def render_frame(
        gen_label: int,
        pop_xy: np.ndarray,
        best_xy: np.ndarray,
        best_fit: float,
        best_so_far_xy: np.ndarray,
        best_so_far_fit: float,
    ) -> Image.Image:
        im = Image.new("RGB", (CANVAS_W, CANVAS_H), BG)
        dr = ImageDraw.Draw(im)

        dr.rounded_rectangle(
            [inner_x, inner_y, inner_x + inner_w, inner_y + inner_h],
            radius=16, fill=CARD, outline=STROKE, width=2
        )

        dr.text((inner_x + 22, inner_y + 16), "Genetic Algorithm (Rastrigin)", fill=TEXT, font=FONT)
        meta = f"gen={gen_label}/{GENERATIONS} | pop={POP} | seed={seed}"
        dr.text((inner_x + 22, inner_y + 36), fit_text(dr, meta, left_text_max_w), fill=MUTED, font=FONT)

        bar_x, bar_y, bar_w, bar_h = inner_x + 22, inner_y + 56, LEFT_W - 44, 12
        dr.rounded_rectangle([bar_x, bar_y, bar_x + bar_w, bar_y + bar_h],
                             radius=8, fill=BAR_BG, outline=STROKE)
        prog = (best_so_far_fit - best0) / denom
        prog = max(0.0, min(1.0, prog))
        dr.rounded_rectangle([bar_x, bar_y, bar_x + int(bar_w * prog), bar_y + bar_h],
                             radius=8, fill=BAR_FILL)

        y_cursor = inner_y + 78
        dr.text((inner_x + 22, y_cursor),
                fit_text(dr, f"best fitness: {best_fit:.3f}   best-so-far: {best_so_far_fit:.3f}", left_text_max_w),
                fill=MUTED, font=FONT)
        y_cursor += 20
        dr.text((inner_x + 22, y_cursor),
                fit_text(dr, f"crossover={CROSS_RATE:.2f}   mut={MUT_RATE:.2f}   elite={ELITE}", left_text_max_w),
                fill=MUTED, font=FONT)
        y_cursor += 22

        exp = "visualização: população evolui sobre a paisagem; fitness = -f(x,y) (máximo em torno de 0,0)"
        for ln in wrap_text(dr, exp, left_text_max_w):
            dr.text((inner_x + 22, y_cursor), ln, fill=(120, 135, 155), font=FONT)
            y_cursor += 18

        # Plot BG + heatmap
        dr.rounded_rectangle(
            [plot_x0 - 12, plot_y0 - 12, plot_x0 + plot_w + 12, plot_y0 + plot_h + 12],
            radius=14, fill=(11, 18, 32), outline=STROKE
        )
        im.paste(heat_img, (int(px0), int(py0)))

        # Contornos
        for lv in levels:
            for gy in range(GRID_H - 1):
                for gx in range(GRID_W - 1):
                    v00 = fld[gy, gx]
                    v10 = fld[gy, gx + 1]
                    v01 = fld[gy + 1, gx]
                    v11 = fld[gy + 1, gx + 1]
                    if min(v00, v10, v01, v11) <= lv <= max(v00, v10, v01, v11):
                        x1 = px0 + (gx / (GRID_W - 1)) * pw
                        y1 = py0 + (gy / (GRID_H - 1)) * ph
                        dr.point((x1, y1), fill=C_CONTOUR)

        # Pop
        for i in range(pop_xy.shape[0]):
            px, py = to_px(pop_xy[i])
            dr.ellipse([px - 2.5, py - 2.5, px + 2.5, py + 2.5], fill=C_POP)

        # best da geração (laranja)
        bx, by = to_px(best_xy)
        dr.ellipse([bx - 5, by - 5, bx + 5, by + 5], fill=C_BEST)

        # best-so-far (verde) + rastro
        sx, sy = to_px(best_so_far_xy)
        best_trail.append((sx, sy))
        trail = best_trail[-TRAIL:]
        if len(trail) >= 2:
            for k in range(1, len(trail)):
                a = k / (len(trail) - 1)
                col = (
                    int(C_BEST_SO_FAR[0] * a + 20 * (1 - a)),
                    int(C_BEST_SO_FAR[1] * a + 30 * (1 - a)),
                    int(C_BEST_SO_FAR[2] * a + 40 * (1 - a)),
                )
                dr.line([trail[k - 1], trail[k]], fill=col, width=3)
        dr.ellipse([sx - 6, sy - 6, sx + 6, sy + 6], outline=C_BEST_SO_FAR, width=2)

        dr.text((plot_x0 + 30, plot_y0 + 10), "y", fill=MUTED, font=FONT)
        dr.text((plot_x0 + plot_w - 36, plot_y0 + plot_h - 26), "x", fill=MUTED, font=FONT)

        footer = f"best-so-far xy=[{best_so_far_xy[0]: .2f}, {best_so_far_xy[1]: .2f}]"
        dr.text((inner_x + 22, inner_y + inner_h - 26), fit_text(dr, footer, int(inner_w - 44)),
                fill=(100, 116, 139), font=FONT)

        return im

    # Segura gen=0 para leitura do mapa
    s0 = snaps[0]
    img0 = render_frame(
        gen_label=0,
        pop_xy=s0.pop,
        best_xy=s0.best,
        best_fit=s0.best_fit,
        best_so_far_xy=s0.best_so_far,
        best_so_far_fit=s0.best_so_far_fit,
    )
    images.append(img0)
    for _ in range(HOLD_GEN0):
        images.append(img0)

    # Tween entre gerações
    for g in range(len(snaps) - 1):
        a = snaps[g]
        b = snaps[g + 1]

        for t_i in range(TWEEN_PER_GEN):
            t = t_i / float(TWEEN_PER_GEN)  # 0..(TWEEN-1)/TWEEN
            pop_xy = lerp(a.pop, b.pop, t)
            best_xy = lerp(a.best, b.best, t)
            # best-so-far é “degrau” (monótono): usa do próximo snap para estabilidade visual
            bsf_xy = b.best_so_far
            bsf_fit = b.best_so_far_fit

            # best_fit interpolado só para texto não “pular” (visual)
            best_fit = float((1.0 - t) * a.best_fit + t * b.best_fit)

            images.append(
                render_frame(
                    gen_label=a.gen,  # mantém label consistente durante tween
                    pop_xy=pop_xy,
                    best_xy=best_xy,
                    best_fit=best_fit,
                    best_so_far_xy=bsf_xy,
                    best_so_far_fit=bsf_fit,
                )
            )

    # Hold final
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

    print(f"[ok] wrote {out} | frames={len(images)} | seed={seed} | tween/gen={TWEEN_PER_GEN}")


if __name__ == "__main__":
    main()