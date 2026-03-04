#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import heapq
import numpy as np


Pos = Tuple[int, int]


@dataclass(frozen=True)
class AStarFrame:
    open_set: List[Pos]
    closed_set: List[Pos]
    path: List[Pos]


def _seed_from_today() -> int:
    d = date.today()
    return (d.year * 10000 + d.month * 100 + d.day) % (2**32 - 1)


def _heuristic(a: Pos, b: Pos) -> int:
    # Manhattan
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _neighbors(p: Pos, W: int, H: int) -> List[Pos]:
    x, y = p
    cand = [(x+1,y), (x-1,y), (x,y+1), (x,y-1)]
    return [(nx, ny) for nx, ny in cand if 0 <= nx < W and 0 <= ny < H]


def _reconstruct(came: Dict[Pos, Pos], cur: Pos) -> List[Pos]:
    path = [cur]
    while cur in came:
        cur = came[cur]
        path.append(cur)
    path.reverse()
    return path


def simulate_astar(grid: np.ndarray, start: Pos, goal: Pos, max_frames: int = 44) -> List[AStarFrame]:
    W, H = grid.shape[1], grid.shape[0]

    open_heap: List[Tuple[int, int, Pos]] = []
    g: Dict[Pos, int] = {start: 0}
    came: Dict[Pos, Pos] = {}
    open_set = {start}
    closed_set = set()

    # (f, tie, node)
    heapq.heappush(open_heap, (_heuristic(start, goal), 0, start))
    tie = 1

    frames: List[AStarFrame] = []

    def snapshot(cur: Optional[Pos]) -> None:
        # path parcial (se existir cur)
        path = _reconstruct(came, cur) if cur is not None else []
        frames.append(AStarFrame(
            open_set=sorted(open_set),
            closed_set=sorted(closed_set),
            path=path
        ))

    snapshot(start)

    while open_heap and len(frames) < max_frames:
        _, _, cur = heapq.heappop(open_heap)
        if cur not in open_set:
            continue
        open_set.remove(cur)
        closed_set.add(cur)

        if cur == goal:
            snapshot(cur)
            break

        for nb in _neighbors(cur, W, H):
            if grid[nb[1], nb[0]] == 1:
                continue
            if nb in closed_set:
                continue
            ng = g[cur] + 1
            if ng < g.get(nb, 10**9):
                came[nb] = cur
                g[nb] = ng
                f = ng + _heuristic(nb, goal)
                heapq.heappush(open_heap, (f, tie, nb))
                tie += 1
                open_set.add(nb)

        snapshot(cur)

    # final: caminho completo (se alcançou)
    if goal in closed_set or goal in open_set:
        frames.append(AStarFrame(
            open_set=sorted(open_set),
            closed_set=sorted(closed_set),
            path=_reconstruct(came, goal) if goal in came or goal == start else []
        ))
    return frames


def main() -> None:
    out = Path("assets/astar.svg")
    out.parent.mkdir(parents=True, exist_ok=True)

    seed = _seed_from_today()
    rng = np.random.default_rng(seed)

    # grid “grande”, mas não excessivo para SVG
    GW, GH = 34, 16  # 544 células
    obstacle_p = 0.22

    grid = (rng.random((GH, GW)) < obstacle_p).astype(np.uint8)
    start: Pos = (2, GH // 2)
    goal: Pos = (GW - 3, GH // 2)
    grid[start[1], start[0]] = 0
    grid[goal[1], goal[0]] = 0

    # cria um “corredor” mínimo para garantir caminho na maioria dos dias
    for x in range(start[0], goal[0] + 1):
        grid[start[1], x] = 0

    frames = simulate_astar(grid, start, goal, max_frames=46)
    T = len(frames)
    dur = 10.0
    key_times = [i / (T - 1) for i in range(T)]
    key_times_str = ";".join(f"{t:.6f}" for t in key_times)

    W, H = 980, 360
    pad = 40
    cell_w = (W - 2 * pad) / GW
    cell_h = (H - 2 * pad) / GH
    cell = min(cell_w, cell_h)
    # centraliza
    px0 = (W - GW * cell) / 2
    py0 = (H - GH * cell) / 2 + 12

    # pré-computa estados por frame: open/closed/path
    open_T = [set(f.open_set) for f in frames]
    closed_T = [set(f.closed_set) for f in frames]
    path_T = [set(f.path) for f in frames]

    def cell_state_values(p: Pos) -> str:
        # prioridade visual: obstacle > start/goal > path > closed > open > empty
        x, y = p
        if grid[y, x] == 1:
            return ";".join(["obs"] * T)

        vals = []
        for t in range(T):
            if p == start:
                vals.append("start")
            elif p == goal:
                vals.append("goal")
            elif p in path_T[t]:
                vals.append("path")
            elif p in closed_T[t]:
                vals.append("closed")
            elif p in open_T[t]:
                vals.append("open")
            else:
                vals.append("empty")
        return ";".join(vals)

    # mapeia estado -> cor (fixa)
    colors = {
        "empty":  "#0b1220",
        "open":   "#1d4ed8",
        "closed": "#0ea5e9",
        "path":   "#22c55e",
        "start":  "#a855f7",
        "goal":   "#f97316",
        "obs":    "#111827",
    }

    # truque: anima "fill" com valores de cor (string). SMIL aceita para fill.
    def fill_values_for(p: Pos) -> str:
        states = cell_state_values(p).split(";")
        return ";".join(colors[s] for s in states)

    title = f"A* (grid {GW}x{GH}) — seed diária={seed}"

    svg = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-label="{title}">')
    svg.append("<defs>")
    svg.append("""
      <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
        <feDropShadow dx="0" dy="2" stdDeviation="2" flood-opacity="0.25"/>
      </filter>
    """)
    svg.append("</defs>")

    svg.append(f'<rect x="0" y="0" width="{W}" height="{H}" rx="18" fill="#0b1220"/>')
    svg.append(f'<rect x="18" y="18" width="{W-36}" height="{H-36}" rx="14" fill="#0f172a" stroke="#1f2937"/>')

    svg.append(f'<text x="40" y="58" font-family="ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto" font-size="22" fill="#e5e7eb">A*</text>')
    svg.append(f'<text x="84" y="58" font-family="ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas" font-size="14" fill="#94a3b8">grid={GW}x{GH} • frames={T} • seed diária={seed}</text>')

    # legenda
    lx, ly = 40, 92
    legend = [("open", "open set"), ("closed", "closed"), ("path", "path"), ("start", "start"), ("goal", "goal"), ("obs", "obstacles")]
    for i, (k, label) in enumerate(legend):
        svg.append(f'<rect x="{lx}" y="{ly + i*20 - 10}" width="12" height="12" rx="3" fill="{colors[k]}"/>')
        svg.append(f'<text x="{lx+18}" y="{ly + i*20}" font-family="ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas" font-size="12" fill="#cbd5e1">{label}</text>')

    # grid
    svg.append('<g filter="url(#shadow)">')
    svg.append(f'<rect x="{px0-8:.2f}" y="{py0-8:.2f}" width="{GW*cell+16:.2f}" height="{GH*cell+16:.2f}" rx="12" fill="#0b1220" stroke="#1f2937"/>')
    for y in range(GH):
        for x in range(GW):
            rx = px0 + x * cell
            ry = py0 + y * cell
            p = (x, y)
            svg.append(f'<rect x="{rx:.2f}" y="{ry:.2f}" width="{cell-1.1:.2f}" height="{cell-1.1:.2f}" rx="3" fill="{colors["empty"]}" stroke="#0f172a" stroke-width="0.5">')
            svg.append(f'  <animate attributeName="fill" dur="{dur}s" repeatCount="indefinite" keyTimes="{key_times_str}" values="{fill_values_for(p)}" />')
            svg.append("</rect>")
    svg.append("</g>")

    svg.append(f'<text x="40" y="{H-24}" font-family="ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas" font-size="12" fill="#64748b">gerado automaticamente (GitHub Actions) • animação SVG</text>')
    svg.append("</svg>")

    out.write_text("\n".join(svg), encoding="utf-8")
    print(f"[ok] wrote {out}")


if __name__ == "__main__":
    main()