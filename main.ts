// Sokoban solver: push-optimal, room by room (Node 22, stdlib only, no dependencies).
//
// Strategy. Every generated level decomposes into independent rooms (a 1-wide
// hallway plus rooms of 3-4 boxes; a box pushed out of a room can never return,
// so a level's push-optimum P* is the sum of its rooms' optima). Each room is
// therefore solved on its own, optimally, by breadth-first search over pushes:
// a state is the sorted box squares plus the player's reachable region (named
// by its lowest square), so the first solved state met is push-optimal.
// Deadlock pruning: a box on a dead square (no push sequence, other boxes
// ignored, can ever bring it to a goal) and a frozen 2x2 block of walls/boxes
// with a box off its goal can never lead to a solution, so those pushes are
// dropped. Nothing here reads the clock or any randomness: every budget is a
// count of states/expansions, so two runs produce byte-identical output.
//
// Fallbacks (only reached if a room's BFS gives up, which the generator's own
// guarantee says should not happen for drawn levels): a push-space A* with an
// admissible Hungarian-assignment heuristic, then a move-space A* that at
// least solves the room, even if not in the fewest pushes.
//
// Usage: node main.ts  (reads /input/<id>/level.txt, writes
// /output/<id>/solution.txt; MAKEFUN_INPUT_DIR / MAKEFUN_OUTPUT_DIR override)
import { lstatSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// Work budgets (states/expansions), never wall-clock, so runs are deterministic.
const ROOM_BFS_CAP = 400_000; // per room; generator guarantees <= 250,000
const WHOLE_BFS_CAP = 400_000;
const PUSH_ASTAR_CAP = 400_000;
const MOVE_ASTAR_CAP = 200_000;
const TOTAL_STATE_CAP = 6_000_000; // across the whole 5-level run

let spentStates = 0;

const DIRS: ReadonlyArray<readonly [number, number, string, string]> = [
  [0, -1, 'u', 'U'],
  [0, 1, 'd', 'D'],
  [-1, 0, 'l', 'L'],
  [1, 0, 'r', 'R'],
];

interface Level {
  w: number;
  h: number;
  wall: Uint8Array;
  goals: number[];
  boxes: number[];
  player: number;
}

interface Solved {
  pushes: number;
  moves: string;
}

function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x]!;
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (x === 0 || y === 0 || x === w - 1 || y === h - 1) wall[y * w + x] = 1;
    }
  }
  return { w, h, wall, goals, boxes: boxes.sort((a, b) => a - b), player };
}

/** Squares from which a box alone can be pushed onto some goal (pulled back from the goals over open floor). */
function liveSquares(level: Level): Uint8Array {
  const { w, wall, goals } = level;
  const live = new Uint8Array(wall.length);
  const queue = [...goals];
  for (const g of goals) live[g] = 1;
  for (let qi = 0; qi < queue.length; qi++) {
    const x = queue[qi]!;
    for (const [dx, dy] of DIRS) {
      const d = dy * w + dx;
      const y = x - d;
      if (!wall[y] && !wall[y - d] && !live[y]) {
        live[y] = 1;
        queue.push(y);
      }
    }
  }
  return live;
}

interface SearchCtx {
  w: number;
  size: number;
  wall: Uint8Array;
  goal: Uint8Array;
  live: Uint8Array;
  seen: Int32Array;
  mark: Int32Array;
  stack: Int32Array;
  stamp: number;
}

function makeCtx(level: Level): SearchCtx {
  const goal = new Uint8Array(level.wall.length);
  for (const g of level.goals) goal[g] = 1;
  return {
    w: level.w,
    size: level.wall.length,
    wall: level.wall,
    goal,
    live: liveSquares(level),
    seen: new Int32Array(level.wall.length),
    mark: new Int32Array(level.wall.length),
    stack: new Int32Array(level.wall.length),
    stamp: 0,
  };
}

/** Flood-fill from `from` with `occupied` in the way, marking `arr` with `stamp`; returns the lowest reached square. */
function reach(ctx: SearchCtx, occupied: Uint8Array, from: number, arr: Int32Array, stamp: number): number {
  const { w, wall, stack } = ctx;
  let low = from;
  let n = 0;
  stack[n++] = from;
  arr[from] = stamp;
  while (n > 0) {
    const p = stack[--n]!;
    if (p < low) low = p;
    // up, down, left, right
    let q = p - w;
    if (wall[q] === 0 && occupied[q] === 0 && arr[q] !== stamp) { arr[q] = stamp; stack[n++] = q; }
    q = p + w;
    if (wall[q] === 0 && occupied[q] === 0 && arr[q] !== stamp) { arr[q] = stamp; stack[n++] = q; }
    q = p - 1;
    if (wall[q] === 0 && occupied[q] === 0 && arr[q] !== stamp) { arr[q] = stamp; stack[n++] = q; }
    q = p + 1;
    if (wall[q] === 0 && occupied[q] === 0 && arr[q] !== stamp) { arr[q] = stamp; stack[n++] = q; }
  }
  return low;
}

/** Whether the box just pushed to `t` sits in a frozen 2x2 block (walls/boxes) with a box off its goal. */
function frozenBlock(ctx: SearchCtx, occupied: Uint8Array, t: number): boolean {
  const { w, wall, goal } = ctx;
  for (const a of [t, t - 1, t - w, t - w - 1]) {
    const b = a + 1;
    const c = a + w;
    const d = c + 1;
    if (!(wall[a] || occupied[a]) || !(wall[b] || occupied[b]) || !(wall[c] || occupied[c]) || !(wall[d] || occupied[d])) continue;
    if ((occupied[a] && !goal[a]) || (occupied[b] && !goal[b]) || (occupied[c] && !goal[c]) || (occupied[d] && !goal[d])) return true;
  }
  return false;
}

// State key: the region's lowest square and the sorted box squares. Numeric
// while size ** (boxes + 1) stays under 2 ** 53 (rooms of <= 5 boxes); a
// string key otherwise (whole-level fallback with many boxes).
function keyOf(boxes: number[], low: number, size: number): number | string {
  if (boxes.length <= 5) {
    let k = low;
    for (const b of boxes) k = k * size + b;
    return k;
  }
  return `${low}|${boxes.join(',')}`;
}

/** Shortest walk from `from` to `to` around walls and boxes, in lower-case LURD (first direction first on ties). */
function walkTo(level: Level, occupied: Uint8Array, from: number, to: number): string {
  const { w, wall } = level;
  if (from === to) return '';
  const prev = new Int32Array(wall.length).fill(-1);
  const how = new Array<string>(wall.length);
  const queue = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi]!;
    if (p === to) break;
    for (const [dx, dy, letter] of DIRS) {
      const q = p + dy * w + dx;
      if (wall[q] || occupied[q] || prev[q] !== -1) continue;
      prev[q] = p;
      how[q] = letter;
      queue.push(q);
    }
  }
  if (prev[to] === -1) throw new Error('walkTo: the push square is out of reach');
  let s = '';
  for (let p = to; p !== from; p = prev[p]!) s = how[p]! + s;
  return s;
}

/** Rebuild the move string from a found state's parent chain. */
function movesFromPath(
  level: Level,
  statesBoxes: number[][],
  statesPlayer: number[],
  parent: number[],
  found: number,
): string {
  const { w } = level;
  const path: number[] = [];
  for (let s = found; s > 0; s = parent[s]!) path.push(s);
  path.reverse();
  const occupied = new Uint8Array(level.wall.length);
  let moves = '';
  let player = level.player;
  let boxes = level.boxes;
  for (const s of path) {
    const nextBoxes = statesBoxes[s]!;
    const from = boxes.find((b) => !nextBoxes.includes(b))!;
    const to = nextBoxes.find((b) => !boxes.includes(b))!;
    const d = to - from;
    const dir = DIRS.find(([dx, dy]) => dy * w + dx === d)!;
    for (const b of boxes) occupied[b] = 1;
    moves += walkTo(level, occupied, player, from - d);
    for (const b of boxes) occupied[b] = 0;
    moves += dir[3];
    player = from;
    boxes = nextBoxes;
  }
  return moves;
}

/**
 * Push-optimal BFS over pushes. First solved state met is optimal.
 * Returns null when the search gives up (cap) or the level is unsolvable.
 */
function solvePushBFS(level: Level, maxStates: number): Solved | null {
  const ctx = makeCtx(level);
  const { w, wall, goal, live, size } = ctx;
  const solved = (boxes: number[]): boolean => boxes.every((b) => goal[b]);
  if (level.boxes.some((b) => !live[b])) return null;
  if (solved(level.boxes)) return { pushes: 0, moves: '' };

  const occupied = new Uint8Array(size);
  const statesBoxes: number[][] = [level.boxes];
  const statesPlayer: number[] = [level.player];
  const parent: number[] = [-1];
  const index = new Map<number | string, number>();
  for (const b of level.boxes) occupied[b] = 1;
  index.set(keyOf(level.boxes, reach(ctx, occupied, level.player, ctx.seen, ++ctx.stamp, ctx.stack), size), 0);
  for (const b of level.boxes) occupied[b] = 0;

  let created = 1;
  let found = -1;
  for (let head = 0; found < 0 && head < statesBoxes.length; head++) {
    const boxes = statesBoxes[head]!;
    for (const b of boxes) occupied[b] = 1;
    reach(ctx, occupied, statesPlayer[head]!, ctx.seen, ++ctx.stamp, ctx.stack);
    const here = ctx.stamp;
    for (let bi = 0; bi < boxes.length && found < 0; bi++) {
      const b = boxes[bi]!;
      for (const [dx, dy] of DIRS) {
        const d = dy * w + dx;
        const t = b + d;
        if (ctx.seen[b - d] !== here || wall[t] || occupied[t] || !live[t]) continue;
        occupied[b] = 0;
        occupied[t] = 1;
        const frozen = frozenBlock(ctx, occupied, t);
        let key: number | string = -1;
        let next: number[] | null = null;
        if (!frozen) {
          next = boxes.slice();
          next[bi] = t;
          next.sort((x, y) => x - y);
          key = keyOf(next, reach(ctx, occupied, b, ctx.mark, ++ctx.stamp, ctx.stack), size);
        }
        occupied[t] = 0;
        occupied[b] = 1;
        if (frozen || index.has(key)) continue;
        if (created >= maxStates || spentStates + created >= TOTAL_STATE_CAP) {
          for (const x of boxes) occupied[x] = 0;
          spentStates += created;
          return null;
        }
        index.set(key, statesBoxes.length);
        statesBoxes.push(next!);
        statesPlayer.push(b);
        parent.push(head);
        created++;
        if (solved(next!)) {
          found = statesBoxes.length - 1;
          break;
        }
      }
    }
    for (const x of boxes) occupied[x] = 0;
  }
  spentStates += created;
  if (found < 0) return null;
  return { pushes: parentChainLength(parent, found), moves: movesFromPath(level, statesBoxes, statesPlayer, parent, found) };
}

function parentChainLength(parent: number[], s: number): number {
  let n = 0;
  for (let t = s; t > 0; t = parent[t]!) n++;
  return n;
}

/** All permutations of 0..n-1 (n <= 4), for the Hungarian assignment heuristic. */
function permutations(n: number): number[][] {
  const out: number[][] = [];
  const cur: number[] = [];
  const used = new Array<boolean>(n).fill(false);
  const rec = (): void => {
    if (cur.length === n) { out.push(cur.slice()); return; }
    for (let i = 0; i < n; i++) {
      if (used[i]) continue;
      used[i] = true;
      cur.push(i);
      rec();
      cur.pop();
      used[i] = false;
    }
  };
  rec();
  return out;
}

/**
 * Push-space A* fallback: admissible heuristic = minimum assignment of boxes
 * to goals by push-distance ignoring other boxes. Returns a solution (not
 * necessarily push-optimal) or null.
 */
function solvePushAStar(level: Level, maxExpansions: number): Solved | null {
  const ctx = makeCtx(level);
  const { w, wall, goal, live, size } = ctx;
  const solved = (boxes: number[]): boolean => boxes.every((b) => goal[b]);
  if (level.boxes.some((b) => !live[b])) return null;
  if (solved(level.boxes)) return { pushes: 0, moves: '' };
  const n = level.boxes.length;
  // Push-distance from every floor square to each goal (BFS ignoring boxes).
  const dist: Int32Array[] = level.goals.map((g) => {
    const d = new Int32Array(size).fill(-1);
    const q = [g];
    d[g] = 0;
    for (let qi = 0; qi < q.length; qi++) {
      const p = q[qi]!;
      for (const [dx, dy] of DIRS) {
        const r = p + dy * w + dx;
        if (!wall[r] && d[r] === -1) { d[r] = d[p]! + 1; q.push(r); }
      }
    }
    return d;
  });
  const perms = permutations(n);
  const heuristic = (boxes: number[]): number => {
    let best = Infinity;
    for (const perm of perms) {
      let s = 0;
      for (let i = 0; i < n; i++) {
        const d = dist[perm[i]!]![boxes[i]!]!;
        if (d < 0) { s = Infinity; break; }
        s += d;
      }
      if (s < best) best = s;
    }
    return best;
  };

  const occupied = new Uint8Array(size);
  const statesBoxes: number[][] = [level.boxes];
  const statesPlayer: number[] = [level.player];
  const parent: number[] = [-1];
  const gScore: number[] = [0];
  const best = new Map<number | string, number>();
  for (const b of level.boxes) occupied[b] = 1;
  best.set(keyOf(level.boxes, reach(ctx, occupied, level.player, ctx.seen, ++ctx.stamp, ctx.stack), size), 0);
  for (const b of level.boxes) occupied[b] = 0;

  // Heap entries [f, g, seq, state]; ties: lower f, then higher g, then lower seq.
  const heap: [number, number, number, number][] = [];
  let seq = 0;
  const h0 = heuristic(level.boxes);
  if (h0 === Infinity) return null;
  heap.push([h0, 0, seq++, 0]);
  const less = (a: [number, number, number, number], b: [number, number, number, number]): boolean =>
    a[0] < b[0] || (a[0] === b[0] && (a[1] > b[1] || (a[1] === b[1] && a[2] < b[2])));
  const heapPush = (e: [number, number, number, number]): void => {
    let i = heap.length;
    heap.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!less(heap[i]!, heap[p]!)) break;
      [heap[i], heap[p]] = [heap[p]!, heap[i]!];
      i = p;
    }
  };
  const heapPop = (): [number, number, number, number] => {
    const top = heap[0]!;
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && less(heap[c + 1]!, heap[c]!)) c++;
        if (!less(heap[c]!, heap[i]!)) break;
        [heap[i], heap[c]] = [heap[c]!, heap[i]!];
        i = c;
      }
    }
    return top;
  };

  let expanded = 0;
  let created = 1;
  while (heap.length > 0 && expanded < maxExpansions && spentStates + created < TOTAL_STATE_CAP) {
    const [, g, , s] = heapPop();
    const boxes = statesBoxes[s]!;
    // Stale heap entry (a better g was recorded since).
    for (const b of boxes) occupied[b] = 1;
    const low = reach(ctx, occupied, statesPlayer[s]!, ctx.seen, ++ctx.stamp, ctx.stack);
    const k0 = keyOf(boxes, low, size);
    if (best.get(k0) !== g) {
      for (const x of boxes) occupied[x] = 0;
      continue;
    }
    if (solved(boxes)) {
      for (const x of boxes) occupied[x] = 0;
      spentStates += created;
      return { pushes: g, moves: movesFromPath(level, statesBoxes, statesPlayer, parent, s) };
    }
    expanded++;
    const here = ctx.stamp;
    for (let bi = 0; bi < boxes.length; bi++) {
      const b = boxes[bi]!;
      for (const [dx, dy] of DIRS) {
        const d = dy * w + dx;
        const t = b + d;
        if (ctx.seen[b - d] !== here || wall[t] || occupied[t] || !live[t]) continue;
        occupied[b] = 0;
        occupied[t] = 1;
        const frozen = frozenBlock(ctx, occupied, t);
        let key: number | string = -1;
        let next: number[] | null = null;
        let h = Infinity;
        if (!frozen) {
          next = boxes.slice();
          next[bi] = t;
          next.sort((x, y) => x - y);
          h = heuristic(next);
          if (h !== Infinity) key = keyOf(next, reach(ctx, occupied, b, ctx.mark, ++ctx.stamp, ctx.stack), size);
        }
        occupied[t] = 0;
        occupied[b] = 1;
        if (frozen || h === Infinity) continue;
        const ng = g + 1;
        const prevBest = best.get(key);
        if (prevBest !== undefined && prevBest <= ng) continue;
        best.set(key, ng);
        statesBoxes.push(next!);
        statesPlayer.push(b);
        parent.push(s);
        gScore.push(ng);
        created++;
        heapPush([ng + h, ng, seq++, statesBoxes.length - 1]);
      }
    }
    for (const x of boxes) occupied[x] = 0;
  }
  spentStates += created;
  return null;
}

/**
 * Move-space A* fallback (fewest moves, not fewest pushes): the house
 * baseline's approach, kept so an awkward level still gets a solved answer.
 */
function solveMoveAStar(level: Level, maxExpansions: number): string {
  const { w, wall, goals } = level;
  const dist = (b: number): number =>
    Math.min(...goals.map((g) => Math.abs((g % w) - (b % w)) + Math.abs(Math.floor(g / w) - Math.floor(b / w))));
  const est = (bs: number[]): number => bs.reduce((a, b) => a + dist(b), 0);
  const key = (p: number, bs: number[]): string => `${p}|${bs.join(',')}`;
  const sp: number[] = [level.player];
  const sb: number[][] = [level.boxes.slice().sort((a, b) => a - b)];
  const sg: number[] = [0];
  const parent: number[] = [-1];
  const step: string[] = [''];
  const best = new Map<string, number>([[key(level.player, sb[0]!), 0]]);
  const heap: [number, number, number][] = [[est(sb[0]!), 0, 0]];
  const less = (a: [number, number, number], b: [number, number, number]): boolean =>
    a[0] < b[0] || (a[0] === b[0] && (a[1] > b[1] || (a[1] === b[1] && a[2] < b[2])));
  const heapPush = (e: [number, number, number]): void => {
    let i = heap.length;
    heap.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!less(heap[i]!, heap[p]!)) break;
      [heap[i], heap[p]] = [heap[p]!, heap[i]!];
      i = p;
    }
  };
  const heapPop = (): [number, number, number] => {
    const top = heap[0]!;
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && less(heap[c + 1]!, heap[c]!)) c++;
        if (!less(heap[c]!, heap[i]!)) break;
        [heap[i], heap[c]] = [heap[c]!, heap[i]!];
        i = c;
      }
    }
    return top;
  };
  let expanded = 0;
  while (heap.length > 0 && expanded < maxExpansions && spentStates + expanded < TOTAL_STATE_CAP) {
    const [, g, s] = heapPop();
    if (best.get(key(sp[s]!, sb[s]!)) !== s) continue;
    const bs = sb[s]!;
    if (bs.every((b) => goals.includes(b))) {
      let moves = '';
      for (let t = s; t > 0; t = parent[t]!) moves = step[t]! + moves;
      spentStates += expanded;
      return moves;
    }
    expanded++;
    const p = sp[s]!;
    for (const [dx, dy, letter] of DIRS) {
      const d = dy * w + dx;
      const q = p + d;
      if (wall[q]) continue;
      let nb = bs;
      let move = letter;
      const at = bs.indexOf(q);
      if (at >= 0) {
        if (wall[q + d] || bs.includes(q + d)) continue;
        nb = bs.slice();
        nb[at] = q + d;
        nb.sort((a, b) => a - b);
        move = letter.toUpperCase();
      }
      const k = key(q, nb);
      const seenIdx = best.get(k);
      if (seenIdx !== undefined && sg[seenIdx]! <= g + 1) continue;
      const t = sp.length;
      sp.push(q);
      sb.push(nb);
      sg.push(g + 1);
      parent.push(s);
      step.push(move);
      best.set(k, t);
      heapPush([g + 1 + est(nb), g + 1, t]);
    }
  }
  spentStates += expanded;
  return '';
}

interface Room {
  squares: number[];
  boxes: number[];
  goals: number[];
  done: boolean;
}

/**
 * Split the level into rooms: cut the hallway row's dead squares (a box there
 * reaches no goal) out of the floor; each remaining part holding boxes is a
 * room. Null when the level is no set of rooms.
 */
function roomsOf(level: Level): Room[] | null {
  const { w, wall, goals, boxes, player } = level;
  const live = liveSquares(level);
  const row = Math.floor(player / w);
  const cut = new Uint8Array(wall.length);
  for (let i = row * w; i < (row + 1) * w; i++) if (!wall[i] && !live[i]) cut[i] = 1;
  if (boxes.some((b) => cut[b])) return null;
  const part = new Int32Array(wall.length).fill(-1);
  const rooms: Room[] = [];
  for (let s = 0; s < wall.length; s++) {
    if (wall[s] || cut[s] || part[s] >= 0) continue;
    const squares = [s];
    part[s] = rooms.length;
    for (let i = 0; i < squares.length; i++) {
      for (const [dx, dy] of DIRS) {
        const q = squares[i]! + dy * w + dx;
        if (!wall[q] && !cut[q] && part[q] < 0) {
          part[q] = rooms.length;
          squares.push(q);
        }
      }
    }
    rooms.push({ squares, boxes: [], goals: [], done: false });
  }
  for (const b of boxes) rooms[part[b]!]!.boxes.push(b);
  for (const g of goals) if (part[g]! >= 0) rooms[part[g]!]!.goals.push(g);
  if (goals.some((g) => part[g]! < 0) || rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  return rooms.filter((r) => r.boxes.length > 0);
}

function endOf(w: number, from: number, moves: string): number {
  let p = from;
  for (const ch of moves.toLowerCase()) p += ch === 'u' ? -w : ch === 'd' ? w : ch === 'l' ? -1 : 1;
  return p;
}

/** Solve one room (as a sub-level) with BFS, then push-A*, then move-A*. */
function solveRoom(level: Level, room: Room, rooms: Room[], player: number): Solved | null {
  const wall = level.wall.slice();
  for (const other of rooms) {
    if (other !== room) for (const b of other.done ? other.goals : other.boxes) wall[b] = 1;
  }
  const sub: Level = {
    w: level.w,
    h: level.h,
    wall,
    goals: room.goals.slice().sort((a, b) => a - b),
    boxes: room.boxes.slice().sort((a, b) => a - b),
    player,
  };
  const bfs = solvePushBFS(sub, ROOM_BFS_CAP);
  if (bfs !== null) return bfs;
  const astar = solvePushAStar(sub, PUSH_ASTAR_CAP);
  if (astar !== null) return astar;
  const moves = solveMoveAStar(sub, MOVE_ASTAR_CAP);
  if (moves !== '') {
    // Count pushes in the move solution for the report value.
    let pushes = 0;
    const box = new Set(sub.boxes);
    let p = sub.player;
    for (const ch of moves.toLowerCase()) {
      const d = ch === 'u' ? -sub.w : ch === 'd' ? sub.w : ch === 'l' ? -1 : 1;
      const q = p + d;
      if (box.has(q)) { box.delete(q); box.add(q + d); pushes++; }
      p = q;
    }
    return { pushes, moves };
  }
  return null;
}

/** Solve a whole level: room by room, falling back to whole-level searches. */
function solveLevel(text: string): string {
  const level = parseLevel(text);
  if (level.boxes.every((b) => level.goals.includes(b))) return '';
  const rooms = roomsOf(level);
  if (rooms !== null) {
    let player = level.player;
    let moves = '';
    let ok = true;
    for (const room of rooms) {
      const solved = solveRoom(level, room, rooms, player);
      if (solved === null) { ok = false; break; }
      moves += solved.moves;
      player = endOf(level.w, player, solved.moves);
      room.done = true;
    }
    if (ok) return moves;
    // A room failed: fall through to whole-level attempts, but keep nothing
    // partial (a partial answer scores 0 anyway); the whole-level search
    // starts from the original level.
  }
  const bfs = solvePushBFS(level, WHOLE_BFS_CAP);
  if (bfs !== null) return bfs.moves;
  const astar = solvePushAStar(level, PUSH_ASTAR_CAP);
  if (astar !== null) return astar.moves;
  return solveMoveAStar(level, MOVE_ASTAR_CAP);
}

for (const id of readdirSync(INPUT_DIR).sort()) {
  // One bad instance must never kill the whole run: skip non-directories,
  // and answer anything unsolvable with an empty (valid, 0-point) solution.
  let isDir = false;
  try {
    isDir = lstatSync(join(INPUT_DIR, id)).isDirectory();
  } catch {
    isDir = false;
  }
  if (!isDir) continue;
  let moves = '';
  try {
    moves = solveLevel(readFileSync(join(INPUT_DIR, id, 'level.txt'), 'utf8'));
  } catch {
    moves = '';
  }
  mkdirSync(join(OUTPUT_DIR, id), { recursive: true });
  writeFileSync(join(OUTPUT_DIR, id, 'solution.txt'), `${moves}\n`);
}
