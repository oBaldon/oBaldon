#!/usr/bin/env python3
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import heapq

import numpy as np
from PIL import Image, ImageDraw


Pos = Tuple[int, int]


@dataclass(frozen=True)
class AStarFrame:
    open_set: List[Pos]
    closed_set: List[Pos]
    path: List[Pos]


def seed_from_today() -> int:
    d = date.today()
    return (d.year * 10000 + d.month * 100 + d.day) % (2**32 - 1)


def has_path(grid: np.ndarray, start: Pos, goal: Pos) -> bool:
    GH, GW = grid.shape
    q = deque([start])
    vis = {start}
    while q:
        x, y = q.popleft()
        if (x, y) == goal:
            return True
        for nx, ny in ((x+1,y),(x-1,y),(x,y+1),(x,y-1)):
            if 0 <= nx < GW and 0 <= ny < GH and grid[ny, nx] == 0 and (nx, ny) not in vis:
                vis.add((nx, ny))
                q.append((nx, ny))
    return False


def heuristic(a: Pos, b: Pos) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def neighbors(p: Pos, GW: int, GH: int) -> List[Pos]:
    x, y = p
    cand = [(x+1,y),(x-1,y),(x,y+1),(x,y-1)]
    return [(nx, ny) for nx, ny in cand if 0 <= nx < GW and 0 <= ny < GH]


def reconstruct(came: Dict[Pos, Pos], cur: Pos) -> List[Pos]:
    path = [cur]
    while cur in came:
        cur = came[cur]
        path.append(cur)
    path.reverse()
    return path


def simulate_astar(grid: np.ndarray, start: Pos, goal: Pos, max_frames: int = 70) -> List[AStarFrame]:
    GH, GW = grid.shape
    open_heap: List[Tuple[int, int, Pos]] = []
    g: Dict[Pos, int] = {start: 0}
    came: Dict[Pos, Pos] = {}

    open_set = {start}
    closed_set = set()

    tie = 0
    heapq.heappush(open_heap, (heuristic(start, goal), tie, start))

    frames: List[AStarFrame] = []

    def snap(cur: Optional[Pos]) -> None:
        path = reconstruct(came, cur) if cur is not None and (cur in came or cur == start) else []
        frames.append(AStarFrame(sorted(open_set), sorted(closed_set), path))

    snap(start)

    while open_heap and len(frames) < max_frames:
        _, _, cur = heapq.heappop(open_heap)
        if cur not in open_set:
            continue
        open_set.remove(cur)
        closed_set.add(cur)

        if cur == goal:
            snap(cur)
            break

        for nb in neighbors(cur, GW, GH):
            if grid[nb[1], nb[0]] == 1:
                continue
            if nb in closed_set:
                continue
            ng = g[cur] + 1
            if ng < g.get(nb, 10**9):
                came[nb] = cur
                g[nb] = ng
                tie += 1
                f = ng + heuristic(nb, goal)
                heapq.heappush(open_heap, (f, tie, nb))
                open_set.add(nb)

        snap(cur)

    # frame final com caminho completo
    if goal in came:
        frames.append(AStarFrame(sorted(open_set), sorted(closed_set), reconstruct(came, goal)))
    return frames


def main() -> None:
    Path("assets").mkdir(parents=True, exist_ok=True)
    out = Path("assets/astar.gif")

    seed = seed_from_today()
    rng = np.random.default_rng(seed)

    # grid
    GW, GH = 44, 18  # maior para visualização
    start = (2, GH // 2)
    goal = (GW - 3, GH // 2)

    obstacle_p = 0.27
    grid = None

    # tenta gerar grids com caminho, mas não trivial
    for _ in range(80):
        g = (rng.random((GH, GW)) < obstacle_p).astype(np.uint8)
        g[start[1], start[0]] = 0
        g[goal[1], goal[0]] = 0
        if has_path(g, start, goal):
            grid = g
            break
    if grid is None:
        obstacle_p = 0.18
        grid = (rng.random((GH, GW)) < obstacle_p).astype(np.uint8)
        grid[start[1], start[0]] = 0
        grid[goal[1], goal[0]] = 0

    frames = simulate_astar(grid, start, goal, max_frames=80)

    # render
    W, H = 980, 360
    bg = (11, 18, 32)
    card = (15, 23, 42)
    stroke = (31, 41, 55)

    empty = (11, 18, 32)
    obs = (17, 24, 39)
    open_c = (29, 78, 216)
    closed_c = (14, 165, 233)
    path_c = (34, 197, 94)
    start_c = (168, 85, 247)
    goal_c = (249, 115, 22)

    pad = 18
    inner_x, inner_y = pad, pad
    inner_w, inner_h = W - 2 * pad, H - 2 * pad

    left_w = 260
    gap = 18
    grid_x0 = inner_x + left_w + gap
    grid_y0 = inner_y + 38
    grid_w = inner_w - left_w - gap - 18
    grid_h = inner_h - 60

    cell = min(grid_w / GW, grid_h / GH)
    gx = grid_x0 + (grid_w - GW * cell) / 2
    gy = grid_y0 + (grid_h - GH * cell) / 2

    images: List[Image.Image] = []

    for t, fr in enumerate(frames):
        im = Image.new("RGB", (W, H), bg)
        dr = ImageDraw.Draw(im)

        dr.rounded_rectangle([inner_x, inner_y, inner_x + inner_w, inner_y + inner_h], radius=16, fill=card, outline=stroke, width=2)

        dr.text((inner_x + 22, inner_y + 16), "A*", fill=(229, 231, 235))
        dr.text((inner_x + 60, inner_y + 20), f"grid={GW}x{GH} | frame={t+1}/{len(frames)} | seed={seed}", fill=(148, 163, 184))

        # legenda simples
        lx, ly = inner_x + 22, inner_y + 64
        legend = [("open", open_c), ("closed", closed_c), ("path", path_c), ("start", start_c), ("goal", goal_c), ("obstacles", obs)]
        for i, (name, col) in enumerate(legend):
            dr.rectangle([lx, ly + i * 20 - 10, lx + 12, ly + i * 20 + 2], fill=col)
            dr.text((lx + 18, ly + i * 20 - 12), name, fill=(148, 163, 184))

        # grid background
        dr.rounded_rectangle([gx - 12, gy - 12, gx + GW * cell + 12, gy + GH * cell + 12], radius=14, fill=(11, 18, 32), outline=stroke)

        open_set = set(fr.open_set)
        closed_set = set(fr.closed_set)
        path_set = set(fr.path)

        for y in range(GH):
            for x in range(GW):
                rx = gx + x * cell
                ry = gy + y * cell
                p = (x, y)
                if grid[y, x] == 1:
                    col = obs
                else:
                    col = empty
                    if p in open_set:
                        col = open_c
                    if p in closed_set:
                        col = closed_c
                    if p in path_set:
                        col = path_c
                    if p == start:
                        col = start_c
                    if p == goal:
                        col = goal_c

                dr.rectangle([rx, ry, rx + cell - 1.0, ry + cell - 1.0], fill=col, outline=(15, 23, 42))

        images.append(im)

    images[0].save(
        out,
        save_all=True,
        append_images=images[1:],
        duration=110,
        loop=0,
        optimize=True,
    )
    print(f"[ok] wrote {out}")


if __name__ == "__main__":
    main()