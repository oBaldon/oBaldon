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


# =========================
# Configuração (ajuste aqui)
# =========================
CANVAS_W, CANVAS_H = 980, 360  # "grande"

GW, GH = 44, 18          # grid maior (visual melhor)
OBSTACLE_P = 0.27
MAX_GEN_TRIES = 120      # tentativas para gerar grid com caminho

# Snapshots/tempo
SNAP_EVERY = 2           # fotografa a cada N expansões (controle de duração)
MAX_SNAPS = 220          # limita quantidade de frames na fase de busca (não limita a busca)
PATH_REVEAL_STEP = 2     # revela o caminho em passos (2 células por frame)
HOLD_LAST = 18
FRAME_MS = 55

# Layout: painel esquerdo + grid à direita
CARD_PAD = 18
LEFT_W = 260
GAP = 18

# Cores
BG = (11, 18, 32)
CARD = (15, 23, 42)
STROKE = (31, 41, 55)
TEXT = (229, 231, 235)
MUTED = (148, 163, 184)

EMPTY = (11, 18, 32)
OBS = (17, 24, 39)
OPEN_C = (29, 78, 216)
CLOSED_C = (14, 165, 233)
PATH_C = (34, 197, 94)
START_C = (168, 85, 247)
GOAL_C = (249, 115, 22)


Pos = Tuple[int, int]


@dataclass(frozen=True)
class AStarSnap:
    open_set: List[Pos]
    closed_set: List[Pos]
    path: List[Pos]


def seed_from_today() -> int:
    d = date.today()
    return (d.year * 10000 + d.month * 100 + d.day) % (2**32 - 1)


def has_path(grid: np.ndarray, start: Pos, goal: Pos) -> bool:
    GH_, GW_ = grid.shape
    q = deque([start])
    vis = {start}
    while q:
        x, y = q.popleft()
        if (x, y) == goal:
            return True
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= nx < GW_ and 0 <= ny < GH_ and grid[ny, nx] == 0 and (nx, ny) not in vis:
                vis.add((nx, ny))
                q.append((nx, ny))
    return False


def heuristic(a: Pos, b: Pos) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def neighbors(p: Pos, GW_: int, GH_: int) -> List[Pos]:
    x, y = p
    cand = ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))
    return [(nx, ny) for nx, ny in cand if 0 <= nx < GW_ and 0 <= ny < GH_]


def reconstruct(came: Dict[Pos, Pos], cur: Pos) -> List[Pos]:
    path = [cur]
    while cur in came:
        cur = came[cur]
        path.append(cur)
    path.reverse()
    return path


def astar_collect_snaps(grid: np.ndarray, start: Pos, goal: Pos) -> Tuple[List[AStarSnap], Optional[List[Pos]]]:
    """
    Executa A* completo até encontrar goal ou esgotar.
    Coleta snapshots limitados por MAX_SNAPS, mas SEM encerrar a busca por limite.
    Retorna (snaps, path_final_ou_None).
    """
    GH_, GW_ = grid.shape

    open_heap: List[Tuple[int, int, Pos]] = []
    g: Dict[Pos, int] = {start: 0}
    came: Dict[Pos, Pos] = {}

    open_set = {start}
    closed_set = set()

    tie = 0
    heapq.heappush(open_heap, (heuristic(start, goal), tie, start))

    snaps: List[AStarSnap] = []

    def snapshot(cur_for_path: Optional[Pos]) -> None:
        if len(snaps) >= MAX_SNAPS:
            return
        if cur_for_path is None:
            path = []
        else:
            # caminho parcial (se existir)
            path = reconstruct(came, cur_for_path) if (cur_for_path == start or cur_for_path in came) else []
        snaps.append(AStarSnap(sorted(open_set), sorted(closed_set), path))

    expansions = 0
    snapshot(start)

    found = False

    while open_heap:
        _, _, cur = heapq.heappop(open_heap)
        if cur not in open_set:
            continue

        open_set.remove(cur)
        closed_set.add(cur)

        if cur == goal:
            found = True
            # snapshot do goal (parcial)
            snapshot(cur)
            break

        for nb in neighbors(cur, GW_, GH_):
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

        expansions += 1
        if expansions % SNAP_EVERY == 0:
            snapshot(cur)

    if not found:
        return snaps, None

    # caminho final completo
    final_path = reconstruct(came, goal) if (goal in came or goal == start) else [start]
    return snaps, final_path


def render_snaps_to_gif(grid: np.ndarray, start: Pos, goal: Pos, snaps: List[AStarSnap], final_path: Optional[List[Pos]], seed: int) -> None:
    Path("assets").mkdir(parents=True, exist_ok=True)
    out = Path("assets/astar.gif")

    # Layout do card
    inner_x, inner_y = CARD_PAD, CARD_PAD
    inner_w, inner_h = CANVAS_W - 2 * CARD_PAD, CANVAS_H - 2 * CARD_PAD

    grid_x0 = inner_x + LEFT_W + GAP
    grid_y0 = inner_y + 38
    grid_w = inner_w - LEFT_W - GAP - 18
    grid_h = inner_h - 60

    cell = min(grid_w / GW, grid_h / GH)
    gx = grid_x0 + (grid_w - GW * cell) / 2
    gy = grid_y0 + (grid_h - GH * cell) / 2

    # Constrói a linha do tempo: snaps + "path reveal" + hold
    timeline: List[AStarSnap] = []
    timeline.extend(snaps)

    if final_path is not None and len(final_path) > 0:
        # revela o caminho progressivamente para alongar e dar "acabamento"
        for i in range(1, len(final_path) + 1, PATH_REVEAL_STEP):
            partial = final_path[:i]
            # usa último open/closed conhecido
            base = timeline[-1] if timeline else AStarSnap([], [], [])
            timeline.append(AStarSnap(base.open_set, base.closed_set, partial))
        # garante o caminho completo
        base = timeline[-1]
        timeline.append(AStarSnap(base.open_set, base.closed_set, final_path))

    if timeline:
        last = timeline[-1]
        for _ in range(HOLD_LAST):
            timeline.append(last)

    images: List[Image.Image] = []

    for t, fr in enumerate(timeline):
        im = Image.new("RGB", (CANVAS_W, CANVAS_H), BG)
        dr = ImageDraw.Draw(im)

        # Card externo
        dr.rounded_rectangle(
            [inner_x, inner_y, inner_x + inner_w, inner_y + inner_h],
            radius=16, fill=CARD, outline=STROKE, width=2
        )

        dr.text((inner_x + 22, inner_y + 16), "A*", fill=TEXT)
        dr.text(
            (inner_x + 60, inner_y + 20),
            f"grid={GW}x{GH} | frames={len(timeline)} | seed={seed}",
            fill=MUTED
        )

        # Legenda
        lx, ly = inner_x + 22, inner_y + 64
        legend = [
            ("open", OPEN_C),
            ("closed", CLOSED_C),
            ("path", PATH_C),
            ("start", START_C),
            ("goal", GOAL_C),
            ("obstacles", OBS),
        ]
        for i, (name, col) in enumerate(legend):
            dr.rectangle([lx, ly + i * 20 - 10, lx + 12, ly + i * 20 + 2], fill=col)
            dr.text((lx + 18, ly + i * 20 - 12), name, fill=MUTED)

        # Grid background
        dr.rounded_rectangle([gx - 12, gy - 12, gx + GW * cell + 12, gy + GH * cell + 12], radius=14, fill=(11, 18, 32), outline=STROKE)

        open_set = set(fr.open_set)
        closed_set = set(fr.closed_set)
        path_set = set(fr.path)

        for y in range(GH):
            for x in range(GW):
                rx = gx + x * cell
                ry = gy + y * cell
                p = (x, y)

                if grid[y, x] == 1:
                    col = OBS
                else:
                    col = EMPTY
                    if p in open_set:
                        col = OPEN_C
                    if p in closed_set:
                        col = CLOSED_C
                    if p in path_set:
                        col = PATH_C
                    if p == start:
                        col = START_C
                    if p == goal:
                        col = GOAL_C

                dr.rectangle([rx, ry, rx + cell - 1.0, ry + cell - 1.0], fill=col, outline=CARD)

        # Rodapé
        status = "found" if final_path is not None else "no-path"
        dr.text((inner_x + 22, inner_y + inner_h - 26), f"status={status} (GIF)", fill=(100, 116, 139))

        images.append(im)

    images[0].save(
        out,
        save_all=True,
        append_images=images[1:],
        duration=FRAME_MS,
        loop=0,
        optimize=False
    )
    print(f"[ok] wrote {out}")


def main() -> None:
    seed = seed_from_today()
    rng = np.random.default_rng(seed)

    start = (2, GH // 2)
    goal = (GW - 3, GH // 2)

    grid = None
    for _ in range(MAX_GEN_TRIES):
        g = (rng.random((GH, GW)) < OBSTACLE_P).astype(np.uint8)
        g[start[1], start[0]] = 0
        g[goal[1], goal[0]] = 0
        if has_path(g, start, goal):
            grid = g
            break

    if grid is None:
        # fallback: reduz obstáculos para aumentar chance de caminho
        p2 = max(0.12, OBSTACLE_P - 0.10)
        grid = (rng.random((GH, GW)) < p2).astype(np.uint8)
        grid[start[1], start[0]] = 0
        grid[goal[1], goal[0]] = 0

    snaps, final_path = astar_collect_snaps(grid, start, goal)
    render_snaps_to_gif(grid, start, goal, snaps, final_path, seed)


if __name__ == "__main__":
    main()