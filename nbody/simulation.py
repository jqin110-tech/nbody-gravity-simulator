"""The state of an N-body system and a loop that steps it forward in time."""

from dataclasses import dataclass, field

import numpy as np

from nbody.forces import AccelFn, Array
from nbody.integrators import Stepper


@dataclass
class System:
    """Positions (N, 3), velocities (N, 3) and masses (N,) as float64 NumPy arrays."""

    pos: Array
    vel: Array
    mass: Array

    def __post_init__(self) -> None:
        self.pos = np.asarray(self.pos, dtype=np.float64)
        self.vel = np.asarray(self.vel, dtype=np.float64)
        self.mass = np.asarray(self.mass, dtype=np.float64)
        n = len(self.mass)
        if self.pos.shape != (n, 3) or self.vel.shape != (n, 3):
            raise ValueError(f"expected pos and vel of shape ({n}, 3)")

    @property
    def n(self) -> int:
        return len(self.mass)

    def copy(self) -> "System":
        return System(self.pos.copy(), self.vel.copy(), self.mass.copy())


@dataclass
class Trajectory:
    """Snapshots recorded every `record_every` steps (including the initial state)."""

    times: Array
    pos: Array  # (frames, N, 3)
    vel: Array  # (frames, N, 3)
    mass: Array
    extra: dict = field(default_factory=dict)


def run(
    system: System,
    stepper: Stepper,
    accel: AccelFn,
    dt: float,
    n_steps: int,
    record_every: int = 1,
) -> Trajectory:
    """Advance `system` by `n_steps` steps of size `dt`, returning recorded snapshots.

    The input system is not modified.
    """
    pos, vel = system.pos.copy(), system.vel.copy()
    acc = accel(pos)
    times, pos_rec, vel_rec = [0.0], [pos.copy()], [vel.copy()]
    for step in range(1, n_steps + 1):
        pos, vel, acc = stepper(pos, vel, acc, dt, accel)
        if step % record_every == 0:
            times.append(step * dt)
            pos_rec.append(pos.copy())
            vel_rec.append(vel.copy())
    return Trajectory(np.array(times), np.array(pos_rec), np.array(vel_rec), system.mass.copy())
