#!/usr/bin/env python3
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import List, Tuple

import numpy as np


@dataclass(frozen=True)
class KMeansFrame:
    centers: np.ndarray          # (k, 2)
    labels: np.ndarray           # (n,)
    inertia: float


def _seed_from_today() -> int:
    # muda diariamente (mas determinístico no dia)
    d = date.today()
    return (d.year * 10000 + d.month * 100 + d.day) % (2**32 - 1)


def _generate_points(rng: np.random.Generator, n: int = 240) -> np.ndarray:
    # mistura de gaussianas para ficar “bonito”
    means = np.array([[-2.0, -1.5], [2.2, 1.7], [-1.0, 2.3], [2.5, -2.0]])
    covs = [
        np.array([[0.45, 0.12], [0.12, 0.35]]),
        np.array([[0.35, -0.10], [-0.10, 0.40]]),
        np.array([[0.30, 0.08], [0.08, 0.30]]),
        np.array([[0.40, -0.06], [-0.06, 0.30]]),
    ]
    pts = []
    for i in range(n):
        j = int(rng.integers(0, len(means)))
        pts.append(rng.multivariate_normal(means[j], covs[j]))
    return np.array(pts, dtype=float)


def _kmeans_frames(X: np.ndarray, k: int, iters: int, rng: np.random.Generator) -> List[KMeansFrame]:
    n = X.shape[0]
    # init: amostra pontos como centróides
    centers = X[rng.choice(n, size=k, replace=False)].copy()

    frames: List[KMeansFrame] = []
    for _ in range(iters):
        # atribuição
        d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)  # (n,k)
        labels = d2.argmin(axis=1)

        # atualização
        new_centers = centers.copy()
        for j in range(k):
            mask = labels == j
            if mask.any():
                new_centers[j] = X[mask].mean(axis=0)

        inertia = float(np.take_along_axis(d2, labels[:, None], axis=1).sum())
        frames.append(KMeansFrame(centers=new_centers.copy(), labels=labels.copy(), inertia=inertia))

        centers = new_centers

    return frames


def _normalize_to_viewbox(X: np.ndarray, W: int, H: int, pad: float = 50.0) -> np.ndarray:
    # mapeia X para [pad, W-pad] x [pad, H-pad]
    xmin, ymin = X.min(axis=0)
    xmax, ymax = X.max(axis=0)
    sx = (W - 2 * pad) / (xmax - xmin + 1e-9)
    sy = (H - 2 * pad) / (ymax - ymin + 1e-9)
    s = min(sx, sy)
    Xn = (X - np.array([xmin, ymin])) * s + pad
    # inverte Y (SVG cresce para baixo)
    Xn[:, 1] = H - Xn[:, 1]
    return Xn


def _svg_escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def main() -> None:
    out = Path("assets/kmeans.svg")
    out.parent.mkdir(parents=True, exist_ok=True)

    seed = _seed_from_today()
    rng = np.random.default_rng(seed)

    W, H = 980, 360  # “grande” e com boa visualização
    k = 4
    iters = 18

    X = _generate_points(rng, n=260)
    frames = _kmeans_frames(X, k=k, iters=iters, rng=rng)

    # normaliza pontos fixos
    Xv = _normalize_to_viewbox(X, W, H, pad=60.0)

    # normaliza centróides por frame (mesmo transform do espaço original)
    # reaplica normalize usando os limites de X (coerente visualmente)
    # (para isso, convertemos centers no mesmo “scale” de X)
    # Reutiliza a função, mas precisa do mesmo xmin/xmax -> solução: normalizar concatenado e fatiar
    all_centers = np.vstack([f.centers for f in frames])
    concat = np.vstack([X, all_centers])
    concat_v = _normalize_to_viewbox(concat, W, H, pad=60.0)
    Xv2 = concat_v[: X.shape[0]]
    centers_v = concat_v[X.shape[0] :].reshape(len(frames), k, 2)

    # paleta discreta (sem depender de CSS externo)
    colors = ["#2563eb", "#16a34a", "#f59e0b", "#ef4444", "#a855f7", "#06b6d4"]

    # para animar cor por label: usa opacity por cluster em “camadas” (mais barato que animar fill por ponto)
    # Abordagem: desenhar cada ponto uma vez por cluster, e animar opacity por frame.
    T = len(frames)
    dur = 10.0  # segundos
    key_times = [i / (T - 1) for i in range(T)]
    key_times_str = ";".join(f"{t:.6f}" for t in key_times)

    # prepara matriz label por frame (n,T)
    labels_T = np.stack([f.labels for f in frames], axis=1)  # (n,T)

    def opacity_values_for_cluster(i_point: int, c: int) -> str:
        vals = ["1" if labels_T[i_point, t] == c else "0.08" for t in range(T)]
        return ";".join(vals)

    # centróides animados por posição
    def anim_values_centroid(j: int, axis: int) -> str:
        vals = [centers_v[t, j, axis] for t in range(T)]
        return ";".join(f"{v:.2f}" for v in vals)

    inertia0 = frames[0].inertia
    inertiaN = frames[-1].inertia

    title = f"K-means (k={k}) — seed diária={seed}"

    svg = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-label="{_svg_escape(title)}">')
    svg.append("<defs>")
    svg.append("""
      <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
        <feDropShadow dx="0" dy="2" stdDeviation="2" flood-opacity="0.25"/>
      </filter>
    """)
    svg.append("</defs>")

    # fundo
    svg.append(f'<rect x="0" y="0" width="{W}" height="{H}" rx="18" fill="#0b1220"/>')
    svg.append(f'<rect x="18" y="18" width="{W-36}" height="{H-36}" rx="14" fill="#0f172a" stroke="#1f2937"/>')

    # header
    svg.append(f'<text x="40" y="58" font-family="ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto" font-size="22" fill="#e5e7eb">K-means</text>')
    svg.append(f'<text x="140" y="58" font-family="ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas" font-size="14" fill="#94a3b8">k={k} • iterações={T} • seed diária={seed}</text>')

    # “inertia bar” animada (indicativo visual)
    bar_x, bar_y, bar_w, bar_h = 40, 80, 360, 10
    svg.append(f'<rect x="{bar_x}" y="{bar_y}" width="{bar_w}" height="{bar_h}" rx="6" fill="#111827" stroke="#1f2937"/>')
    # barra interna: mapeia inertia decrescente -> cresce preenchimento
    def inertia_fill_values() -> str:
        vals = []
        for f in frames:
            # progresso: 0..1 (quanto menor inertia, maior progresso)
            prog = 1.0 - (f.inertia - inertiaN) / (inertia0 - inertiaN + 1e-9)
            vals.append(prog)
        return ";".join(f"{bar_w * v:.2f}" for v in vals)

    svg.append(f'<rect x="{bar_x}" y="{bar_y}" width="0" height="{bar_h}" rx="6" fill="#22c55e">')
    svg.append(f'  <animate attributeName="width" dur="{dur}s" repeatCount="indefinite" keyTimes="{key_times_str}" values="{inertia_fill_values()}" />')
    svg.append("</rect>")
    svg.append(f'<text x="{bar_x+bar_w+12}" y="{bar_y+9}" font-family="ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas" font-size="12" fill="#94a3b8">inertia</text>')

    # área de plot
    plot_x0, plot_y0 = 420, 40
    plot_w, plot_h = W - plot_x0 - 30, H - 70
    svg.append(f'<rect x="{plot_x0}" y="{plot_y0}" width="{plot_w}" height="{plot_h}" rx="12" fill="#0b1220" stroke="#1f2937"/>')

    # pontos (por cluster “virtual”: anima opacity por ponto)
    svg.append('<g filter="url(#shadow)">')
    r = 2.6
    for i, (px, py) in enumerate(Xv2):
        for c in range(k):
            svg.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="{r}" fill="{colors[c]}">')
            svg.append(f'  <animate attributeName="opacity" dur="{dur}s" repeatCount="indefinite" keyTimes="{key_times_str}" values="{opacity_values_for_cluster(i, c)}" />')
            svg.append("</circle>")
    svg.append("</g>")

    # centróides (animados)
    svg.append('<g filter="url(#shadow)">')
    for j in range(k):
        svg.append(f'<circle cx="{centers_v[0, j, 0]:.2f}" cy="{centers_v[0, j, 1]:.2f}" r="9" fill="{colors[j]}" stroke="#e5e7eb" stroke-width="2">')
        svg.append(f'  <animate attributeName="cx" dur="{dur}s" repeatCount="indefinite" keyTimes="{key_times_str}" values="{anim_values_centroid(j, 0)}" />')
        svg.append(f'  <animate attributeName="cy" dur="{dur}s" repeatCount="indefinite" keyTimes="{key_times_str}" values="{anim_values_centroid(j, 1)}" />')
        svg.append("</circle>")
    svg.append("</g>")

    # legenda compacta
    lx, ly = 40, 120
    for j in range(k):
        svg.append(f'<rect x="{lx}" y="{ly + j*22 - 10}" width="12" height="12" rx="3" fill="{colors[j]}"/>')
        svg.append(f'<text x="{lx+18}" y="{ly + j*22}" font-family="ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas" font-size="12" fill="#cbd5e1">cluster {j}</text>')

    svg.append(f'<text x="40" y="{H-24}" font-family="ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas" font-size="12" fill="#64748b">gerado automaticamente (GitHub Actions) • animação SVG</text>')
    svg.append("</svg>")

    out.write_text("\n".join(svg), encoding="utf-8")
    print(f"[ok] wrote {out}")


if __name__ == "__main__":
    main()