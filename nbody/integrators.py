"""Time integrators that share one interface.

Every stepper has the signature

    step(pos, vel, acc, dt, accel) -> (new_pos, new_vel, new_acc)

where `acc` is the acceleration at `pos` and `new_acc` is the acceleration at
`new_pos`. Passing the acceleration along lets leapfrog reuse it, so leapfrog
costs one force evaluation per step, Euler one, and RK4 four.
"""

from collections.abc import Callable

from nbody.forces import AccelFn, Array

Stepper = Callable[[Array, Array, Array, float, AccelFn], tuple[Array, Array, Array]]


def euler_step(pos: Array, vel: Array, acc: Array, dt: float, accel: AccelFn) -> tuple[Array, Array, Array]:
    """Explicit Euler: move both position and velocity using values from the start of the step.

    First-order accurate (global error ~ dt) and not symplectic, so energy drifts.
    """
    new_pos = pos + dt * vel
    new_vel = vel + dt * acc
    return new_pos, new_vel, accel(new_pos)


def leapfrog_step(pos: Array, vel: Array, acc: Array, dt: float, accel: AccelFn) -> tuple[Array, Array, Array]:
    """Leapfrog kick-drift-kick: half kick, full drift, half kick.

    Second-order, time-reversible and symplectic, so energy error stays bounded.
    """
    vel_half = vel + 0.5 * dt * acc
    new_pos = pos + dt * vel_half
    new_acc = accel(new_pos)
    new_vel = vel_half + 0.5 * dt * new_acc
    return new_pos, new_vel, new_acc


def rk4_step(pos: Array, vel: Array, acc: Array, dt: float, accel: AccelFn) -> tuple[Array, Array, Array]:
    """Classical fourth-order Runge-Kutta on the first-order system (x' = v, v' = a(x)).

    Fourth-order accurate (global error ~ dt^4) but not symplectic.
    """
    k1x, k1v = vel, acc
    k2x, k2v = vel + 0.5 * dt * k1v, accel(pos + 0.5 * dt * k1x)
    k3x, k3v = vel + 0.5 * dt * k2v, accel(pos + 0.5 * dt * k2x)
    k4x, k4v = vel + dt * k3v, accel(pos + dt * k3x)
    new_pos = pos + (dt / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x)
    new_vel = vel + (dt / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)
    return new_pos, new_vel, accel(new_pos)


INTEGRATORS: dict[str, Stepper] = {
    "euler": euler_step,
    "leapfrog": leapfrog_step,
    "rk4": rk4_step,
}
