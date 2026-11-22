# Sokoban solver

Push-optimal Sokoban solver for make.fun's `sokoban-solver-01` competition
(Node 22, standard library only, no dependencies).

## What it does

Each level decomposes into independent rooms (a 1-wide hallway plus rooms of
3–4 boxes; a box pushed out of a room can never return, so the level's
push-optimum is the sum of its rooms' optima). The solver splits the level
into rooms and solves each room optimally with breadth-first search over
pushes — a state is the sorted box squares plus the player's reachable region
— with dead-square and frozen-2×2 deadlock pruning, so the first solved
state met is push-optimal. Fallbacks (push-space A* with an admissible
Hungarian-assignment heuristic, then a move-space A*) cover levels that do
not decompose or where the BFS gives up. All budgets are counts of
states/expansions, never wall-clock, so runs are deterministic.

## How to run

```sh
node main.ts
```

Reads `/input/<id>/level.txt` and writes `/output/<id>/solution.txt` for
every instance. `MAKEFUN_INPUT_DIR` and `MAKEFUN_OUTPUT_DIR` override the
input/output directories (e.g. for local testing against the kit's practice
set).

## Files

- `main.ts` — the whole solver
- `LICENSE` — MIT
