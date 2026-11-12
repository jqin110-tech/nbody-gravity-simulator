"""Barnes-Hut gravity: approximate distant groups of bodies by their total mass at their centre of mass.

The octree is stored in flat NumPy arrays (one row per node) rather than Python objects,
so the exact same functions can run as plain Python or be compiled by Numba. That makes
the before/after comparison in Stage 4 a test of the compiler alone, not of two different
algorithms.

Each node is a cube. `kind[n]` says what it holds:
    EMPTY (-1)     an empty leaf (only the root before any insertion),
    INTERNAL (-2)  a cell whose up to 8 children are listed in `children[n]`,
    b >= 0         a leaf holding exactly one body, number b.
"""

from collections.abc import Callable
from types import SimpleNamespace

import numpy as np

from nbody.constants import G as G_DEFAULT
from nbody.forces import AccelFn, Array

EMPTY = -1
INTERNAL = -2
MAX_DEPTH = 64


def _make_kernels(jit: Callable, parallel_jit: Callable, prange: Callable) -> SimpleNamespace:
    """Define the tree kernels, wrapped with `jit` (identity for pure Python, numba.njit for Numba)."""

    @jit
    def octant(p: Array, c: Array) -> int:
        """Which of the 8 child cubes around centre c contains point p (one bit per axis)."""
        return int(p[0] >= c[0]) + 2 * int(p[1] >= c[1]) + 4 * int(p[2] >= c[2])

    @jit
    def new_child(parent: int, octant_idx: int, n_nodes: int, center: Array, half: Array,
                  kind: Array, children: Array, parent_of: Array) -> None:
        """Create node number n_nodes as child `octant_idx` of `parent`."""
        h = 0.5 * half[parent]
        half[n_nodes] = h
        for axis in range(3):
            sign = 1.0 if (octant_idx >> axis) & 1 else -1.0
            center[n_nodes, axis] = center[parent, axis] + sign * h
        kind[n_nodes] = EMPTY
        children[n_nodes, :] = -1
        children[parent, octant_idx] = n_nodes
        parent_of[n_nodes] = parent

    @jit
    def build_tree(pos: Array, max_nodes: int):
        """Insert bodies one by one into an octree. Returns the arrays and node count (-1 if out of space)."""
        center = np.zeros((max_nodes, 3))
        half = np.zeros(max_nodes)
        kind = np.full(max_nodes, EMPTY, dtype=np.int64)
        children = np.full((max_nodes, 8), -1, dtype=np.int64)
        parent_of = np.full(max_nodes, -1, dtype=np.int64)
        lo = np.array([pos[:, 0].min(), pos[:, 1].min(), pos[:, 2].min()])
        hi = np.array([pos[:, 0].max(), pos[:, 1].max(), pos[:, 2].max()])
        center[0] = 0.5 * (lo + hi)
        half[0] = 0.5 * max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) * 1.0001 + 1e-300
        n_nodes = 1
        for i in range(pos.shape[0]):
            node, depth = 0, 0
            while True:
                if kind[node] == EMPTY:
                    kind[node] = i
                    break
                if kind[node] >= 0:  # occupied leaf: push its body one level down, then retry
                    if depth >= MAX_DEPTH or n_nodes >= max_nodes:
                        return center, half, kind, children, parent_of, -1
                    other = kind[node]
                    o = octant(pos[other], center[node])
                    new_child(node, o, n_nodes, center, half, kind, children, parent_of)
                    kind[n_nodes] = other
                    n_nodes += 1
                    kind[node] = INTERNAL
                    continue
                o = octant(pos[i], center[node])  # internal node: descend
                child = children[node, o]
                if child == -1:
                    if n_nodes >= max_nodes:
                        return center, half, kind, children, parent_of, -1
                    new_child(node, o, n_nodes, center, half, kind, children, parent_of)
                    kind[n_nodes] = i
                    n_nodes += 1
                    break
                node, depth = child, depth + 1
        return center, half, kind, children, parent_of, n_nodes

    @jit
    def mass_moments(pos: Array, mass: Array, kind: Array, parent_of: Array, n_nodes: int):
        """Total mass and centre of mass of every node, filled from the leaves up.

        Children are always created after their parent, so walking node numbers
        backwards visits every child before its parent.
        """
        node_mass = np.zeros(n_nodes)
        weighted = np.zeros((n_nodes, 3))
        com = np.zeros((n_nodes, 3))
        for n in range(n_nodes - 1, -1, -1):
            b = kind[n]
            if b >= 0:
                node_mass[n] = mass[b]
                weighted[n] = mass[b] * pos[b]
            if node_mass[n] > 0.0:
                com[n] = weighted[n] / node_mass[n]
            p = parent_of[n]
            if p >= 0:
                node_mass[p] += node_mass[n]
                weighted[p] += weighted[n]
        return node_mass, com

    @parallel_jit
    def tree_accel(pos: Array, center: Array, half: Array, kind: Array, children: Array,
                   node_mass: Array, com: Array, theta: float, eps2: float, G: float) -> Array:
        """Walk the tree once per body. A cell of side s at distance d is used as one point mass if s/d < theta."""
        n = pos.shape[0]
        acc = np.zeros((n, 3))
        for i in prange(n):
            stack = np.empty(8 * MAX_DEPTH + 8, dtype=np.int64)
            stack[0] = 0
            top = 1
            ax, ay, az = 0.0, 0.0, 0.0
            px, py, pz = pos[i, 0], pos[i, 1], pos[i, 2]
            while top > 0:
                top -= 1
                node = stack[top]
                dx, dy, dz = com[node, 0] - px, com[node, 1] - py, com[node, 2] - pz
                d2 = dx * dx + dy * dy + dz * dz
                b = kind[node]
                if b >= 0:
                    if b == i:
                        continue
                elif b == INTERNAL:
                    h = half[node]
                    inside = (abs(px - center[node, 0]) <= h and abs(py - center[node, 1]) <= h
                              and abs(pz - center[node, 2]) <= h)
                    if inside or 4.0 * h * h >= theta * theta * d2:  # (side)^2 >= theta^2 d^2: open it
                        for c in range(8):
                            if children[node, c] >= 0:
                                stack[top] = children[node, c]
                                top += 1
                        continue
                else:
                    continue
                r2 = d2 + eps2
                f = G * node_mass[node] / (r2 * np.sqrt(r2))  # sqrt is much cheaper than r2 ** -1.5
                ax += f * dx
                ay += f * dy
                az += f * dz
            acc[i, 0], acc[i, 1], acc[i, 2] = ax, ay, az
        return acc

    @parallel_jit
    def direct_accel_loops(pos: Array, mass: Array, eps2: float, G: float) -> Array:
        """O(N^2) direct sum written as explicit loops, so it can be compiled the same way as the tree."""
        n = pos.shape[0]
        acc = np.zeros((n, 3))
        for i in prange(n):
            ax, ay, az = 0.0, 0.0, 0.0
            for j in range(n):
                if j != i:
                    dx, dy, dz = pos[j, 0] - pos[i, 0], pos[j, 1] - pos[i, 1], pos[j, 2] - pos[i, 2]
                    r2 = dx * dx + dy * dy + dz * dz + eps2
                    f = G * mass[j] / (r2 * np.sqrt(r2))
                    ax += f * dx
                    ay += f * dy
                    az += f * dz
            acc[i, 0], acc[i, 1], acc[i, 2] = ax, ay, az
        return acc

    @parallel_jit
    def direct_potential_loops(pos: Array, mass: Array, eps2: float, G: float) -> float:
        """Total softened potential energy, -G sum_{i<j} m_i m_j / sqrt(r_ij^2 + eps^2), in O(N^2)."""
        n = pos.shape[0]
        total = 0.0
        for i in prange(n):
            for j in range(i + 1, n):
                dx, dy, dz = pos[j, 0] - pos[i, 0], pos[j, 1] - pos[i, 1], pos[j, 2] - pos[i, 2]
                total -= G * mass[i] * mass[j] / np.sqrt(dx * dx + dy * dy + dz * dz + eps2)
        return total

    return SimpleNamespace(build_tree=build_tree, mass_moments=mass_moments, tree_accel=tree_accel,
                           direct_accel_loops=direct_accel_loops, direct_potential_loops=direct_potential_loops)


def _identity(f: Callable) -> Callable:
    return f


PYTHON = _make_kernels(_identity, _identity, range)
try:
    import numba

    NUMBA = _make_kernels(numba.njit(cache=True), numba.njit(parallel=True, cache=True), numba.prange)
except ImportError:  # Numba is optional; the pure-Python kernels always work.
    NUMBA = None


def _kernels(backend: str) -> SimpleNamespace:
    if backend == "python":
        return PYTHON
    if NUMBA is None:
        raise ImportError("backend='numba' needs the numba package")
    return NUMBA


def build(pos: Array, mass: Array, backend: str = "numba") -> tuple:
    """Build the octree and its mass moments, growing the node arrays until everything fits."""
    k = _kernels(backend)
    max_nodes = 4 * len(mass) + 16
    while True:
        center, half, kind, children, parent_of, n_nodes = k.build_tree(pos, max_nodes)
        if n_nodes > 0:
            break
        if max_nodes > 64 * len(mass) + 1024:
            raise ValueError("octree did not fit: are two bodies at exactly the same position?")
        max_nodes *= 2
    node_mass, com = k.mass_moments(pos, mass, kind, parent_of, n_nodes)
    return center[:n_nodes], half[:n_nodes], kind[:n_nodes], children[:n_nodes], node_mass, com


def bh_accel(pos: Array, mass: Array, theta: float = 0.5, softening: float = 0.0,
             G: float = G_DEFAULT, backend: str = "numba") -> Array:
    """Barnes-Hut acceleration on every body. theta = 0 opens every cell, giving the exact direct sum."""
    pos = np.ascontiguousarray(pos, dtype=np.float64)
    mass = np.ascontiguousarray(mass, dtype=np.float64)
    center, half, kind, children, node_mass, com = build(pos, mass, backend)
    return _kernels(backend).tree_accel(pos, center, half, kind, children, node_mass, com,
                                        float(theta), float(softening) ** 2, float(G))


def direct_accel_compiled(pos: Array, mass: Array, softening: float = 0.0, G: float = G_DEFAULT,
                          backend: str = "numba") -> Array:
    """Direct sum using explicit loops (pure Python or Numba), for like-for-like benchmarks."""
    return _kernels(backend).direct_accel_loops(np.ascontiguousarray(pos, dtype=np.float64),
                                                np.ascontiguousarray(mass, dtype=np.float64),
                                                float(softening) ** 2, float(G))


def potential_energy_compiled(pos: Array, mass: Array, softening: float = 0.0, G: float = G_DEFAULT,
                              backend: str = "numba") -> float:
    """Exact softened potential energy with compiled loops: fast enough to check energy for 100k bodies."""
    return float(_kernels(backend).direct_potential_loops(np.ascontiguousarray(pos, dtype=np.float64),
                                                          np.ascontiguousarray(mass, dtype=np.float64),
                                                          float(softening) ** 2, float(G)))


def make_bh_accel(mass: Array, theta: float = 0.5, softening: float = 0.0, G: float = G_DEFAULT,
                  backend: str = "numba") -> AccelFn:
    """Bind everything except positions, matching the interface the integrators expect."""

    def accel(pos: Array) -> Array:
        return bh_accel(pos, mass, theta, softening, G, backend)

    return accel
