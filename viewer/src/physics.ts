// The same physics as the Python package, written for the browser: softened direct-sum
// gravity and leapfrog (kick-drift-kick). N-body units, G = 1. No DOM code in this file,
// so it can be tested with Node (see viewer/test/physics.test.mjs).

export const G = 1.0;

/** N bodies stored as flat arrays: body i has position pos[3i], pos[3i+1], pos[3i+2]. */
export interface Bodies {
  n: number;
  pos: Float64Array;
  vel: Float64Array;
  acc: Float64Array;
  mass: Float64Array;
  group: Uint8Array; // which galaxy a body belongs to (0 or 1), only used for colour
}

export function emptyBodies(n: number): Bodies {
  return {
    n,
    pos: new Float64Array(3 * n),
    vel: new Float64Array(3 * n),
    acc: new Float64Array(3 * n),
    mass: new Float64Array(n),
    group: new Uint8Array(n),
  };
}

/**
 * Direct-sum accelerations with Plummer softening:
 * a_i = G * sum_j m_j (r_j - r_i) / (|r_j - r_i|^2 + eps^2)^(3/2).
 * Each pair is visited once and applied to both bodies (Newton's third law), which halves the work.
 */
export function computeAccelerations(b: Bodies, eps: number): void {
  const { n, pos, acc, mass } = b;
  const eps2 = eps * eps;
  acc.fill(0);
  for (let i = 0; i < n; i++) {
    const xi = pos[3 * i], yi = pos[3 * i + 1], zi = pos[3 * i + 2];
    let ax = 0, ay = 0, az = 0;
    for (let j = i + 1; j < n; j++) {
      const dx = pos[3 * j] - xi, dy = pos[3 * j + 1] - yi, dz = pos[3 * j + 2] - zi;
      const r2 = dx * dx + dy * dy + dz * dz + eps2;
      const invR3 = G / (r2 * Math.sqrt(r2));
      const fi = invR3 * mass[j], fj = invR3 * mass[i];
      ax += fi * dx; ay += fi * dy; az += fi * dz;
      acc[3 * j] -= fj * dx; acc[3 * j + 1] -= fj * dy; acc[3 * j + 2] -= fj * dz;
    }
    acc[3 * i] += ax; acc[3 * i + 1] += ay; acc[3 * i + 2] += az;
  }
}

/**
 * One leapfrog kick-drift-kick step. b.acc must hold the accelerations at the current
 * positions on entry, and holds them at the new positions on exit (one force call per step).
 */
export function leapfrogStep(b: Bodies, dt: number, eps: number): void {
  const { pos, vel, acc } = b;
  const half = 0.5 * dt;
  for (let k = 0; k < pos.length; k++) {
    vel[k] += half * acc[k]; // half kick
    pos[k] += dt * vel[k]; // drift
  }
  computeAccelerations(b, eps);
  for (let k = 0; k < vel.length; k++) vel[k] += half * acc[k]; // half kick
}

/** Kinetic plus softened potential energy; conserved by the exact dynamics. */
export function totalEnergy(b: Bodies, eps: number): number {
  const { n, pos, vel, mass } = b;
  const eps2 = eps * eps;
  let energy = 0;
  for (let i = 0; i < n; i++) {
    const vx = vel[3 * i], vy = vel[3 * i + 1], vz = vel[3 * i + 2];
    energy += 0.5 * mass[i] * (vx * vx + vy * vy + vz * vz);
    for (let j = i + 1; j < n; j++) {
      const dx = pos[3 * j] - pos[3 * i], dy = pos[3 * j + 1] - pos[3 * i + 1], dz = pos[3 * j + 2] - pos[3 * i + 2];
      energy -= (G * mass[i] * mass[j]) / Math.sqrt(dx * dx + dy * dy + dz * dz + eps2);
    }
  }
  return energy;
}

/** Total momentum vector, sum of m v. Pairwise forces cancel, so it should not change. */
export function totalMomentum(b: Bodies): [number, number, number] {
  let px = 0, py = 0, pz = 0;
  for (let i = 0; i < b.n; i++) {
    px += b.mass[i] * b.vel[3 * i];
    py += b.mass[i] * b.vel[3 * i + 1];
    pz += b.mass[i] * b.vel[3 * i + 2];
  }
  return [px, py, pz];
}

/** Small seeded random number generator (mulberry32), so every visitor sees the same galaxy. */
export function seededRandom(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export interface GalaxyModel {
  nDisc: number; // number of disc bodies; one extra body is the heavy centre
  centralMass: number;
  discMass: number;
  scaleLength: number; // exponential disc: surface density ~ exp(-R / scaleLength) (Freeman 1970)
  scaleHeight: number;
  rMax: number;
}

/**
 * One galaxy at rest at the origin: a heavy central body plus an exponential disc.
 * Disc radii follow R exp(-R / R_d), a Gamma(2) distribution, sampled as the sum of two
 * exponential variables. Velocities are set afterwards by setCircularVelocities.
 */
export function makeGalaxy(model: GalaxyModel, random: () => number): Bodies {
  const b = emptyBodies(model.nDisc + 1);
  b.mass[0] = model.centralMass;
  for (let i = 1; i <= model.nDisc; i++) {
    let radius = Infinity;
    while (radius > model.rMax) radius = -model.scaleLength * Math.log((1 - random()) * (1 - random()));
    const phi = 2 * Math.PI * random();
    // Box-Muller: a Gaussian height above the disc plane.
    const height = model.scaleHeight * Math.sqrt(-2 * Math.log(1 - random())) * Math.cos(2 * Math.PI * random());
    b.pos[3 * i] = radius * Math.cos(phi);
    b.pos[3 * i + 1] = radius * Math.sin(phi);
    b.pos[3 * i + 2] = height;
    b.mass[i] = model.discMass / model.nDisc;
  }
  return b;
}

/**
 * Put every body except body 0 on a circular orbit in the x-y plane: v^2 / R equals the
 * inward pull the code itself computes, so the disc starts in balance with the real forces.
 */
export function setCircularVelocities(b: Bodies, eps: number): void {
  computeAccelerations(b, eps);
  for (let i = 1; i < b.n; i++) {
    const x = b.pos[3 * i], y = b.pos[3 * i + 1];
    const radius = Math.hypot(x, y);
    const inward = -(b.acc[3 * i] * x + b.acc[3 * i + 1] * y) / radius;
    const speed = Math.sqrt(radius * Math.max(inward, 0));
    b.vel[3 * i] = (-speed * y) / radius;
    b.vel[3 * i + 1] = (speed * x) / radius;
    b.vel[3 * i + 2] = 0;
  }
}

/** Rotate every position and velocity by `angle` radians about the x axis (tilts a disc). */
export function tiltAboutX(b: Bodies, angle: number): void {
  const c = Math.cos(angle), s = Math.sin(angle);
  for (const v of [b.pos, b.vel]) {
    for (let i = 0; i < b.n; i++) {
      const y = v[3 * i + 1], z = v[3 * i + 2];
      v[3 * i + 1] = c * y - s * z;
      v[3 * i + 2] = s * y + c * z;
    }
  }
}

/** Shift a whole galaxy by a position offset and a velocity offset. */
export function shift(b: Bodies, dPos: number[], dVel: number[]): void {
  for (let i = 0; i < b.n; i++) {
    for (let k = 0; k < 3; k++) {
      b.pos[3 * i + k] += dPos[k];
      b.vel[3 * i + k] += dVel[k];
    }
  }
}

/** Join two sets of bodies into one; the second set is marked as group 1. */
export function combine(a: Bodies, c: Bodies): Bodies {
  const out = emptyBodies(a.n + c.n);
  out.pos.set(a.pos); out.pos.set(c.pos, 3 * a.n);
  out.vel.set(a.vel); out.vel.set(c.vel, 3 * a.n);
  out.mass.set(a.mass); out.mass.set(c.mass, a.n);
  out.group.set(a.group); out.group.fill(1, a.n);
  return out;
}

/**
 * Relative position and velocity for two bodies approaching on a parabolic (zero-energy)
 * orbit, as in Toomre & Toomre (1972): r = 2q / (1 + cos f), v = sqrt(G M / 2q) (-sin f, 1 + cos f).
 */
export function parabolicOrbit(totalMass: number, pericentre: number, separation: number): { pos: number[]; vel: number[] } {
  const f = -Math.acos((2 * pericentre) / separation - 1); // negative: still falling in
  const speedScale = Math.sqrt((G * totalMass) / (2 * pericentre));
  return {
    pos: [separation * Math.cos(f), separation * Math.sin(f), 0],
    vel: [-speedScale * Math.sin(f), speedScale * (1 + Math.cos(f)), 0],
  };
}
