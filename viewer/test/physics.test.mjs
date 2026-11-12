// Physics tests for the browser simulation. Run with: npm test
import assert from "node:assert/strict";
import test from "node:test";

import {
  combine, computeAccelerations, emptyBodies, leapfrogStep, makeGalaxy, parabolicOrbit, seededRandom,
  setCircularVelocities, totalEnergy, totalMomentum,
} from "../dist/physics.js";

/** Two bodies on a circular orbit of separation 1 in the centre-of-mass frame (G = 1). */
function circularPair(m1, m2) {
  const b = emptyBodies(2);
  const total = m1 + m2;
  const speed = Math.sqrt(total); // relative circular speed sqrt(G M / r) with r = 1
  b.mass.set([m1, m2]);
  b.pos.set([-m2 / total, 0, 0, m1 / total, 0, 0]);
  b.vel.set([0, (-m2 / total) * speed, 0, 0, (m1 / total) * speed, 0]);
  computeAccelerations(b, 0);
  return b;
}

/** Distance of the relative position from where it started, after one orbital period. */
function errorAfterOnePeriod(steps) {
  const b = circularPair(1, 0.5);
  const period = (2 * Math.PI) / Math.sqrt(1.5); // Kepler: T = 2 pi sqrt(r^3 / (G M))
  for (let s = 0; s < steps; s++) leapfrogStep(b, period / steps, 0);
  return Math.hypot(b.pos[3] - b.pos[0] - 1, b.pos[4] - b.pos[1], b.pos[5] - b.pos[2]);
}

test("two-body force matches Newton's law", () => {
  const b = emptyBodies(2);
  b.mass.set([1, 3]);
  b.pos.set([0, 0, 0, 2, 0, 0]);
  computeAccelerations(b, 0);
  assert.ok(Math.abs(b.acc[0] - 3 / 4) < 1e-15); // G m2 / r^2 toward body 2
  assert.ok(Math.abs(b.acc[3] + 1 / 4) < 1e-15); // G m1 / r^2 toward body 1
});

test("a circular orbit returns to its start after one Kepler period", () => {
  assert.ok(errorAfterOnePeriod(2000) < 1e-4);
});

test("leapfrog is second order: halving dt cuts the error by about 4", () => {
  const ratio = errorAfterOnePeriod(500) / errorAfterOnePeriod(1000);
  assert.ok(ratio > 3.8 && ratio < 4.2, `ratio was ${ratio}`);
});

test("energy stays bounded and momentum is conserved for a galaxy", () => {
  const model = { nDisc: 200, centralMass: 1, discMass: 0.2, scaleLength: 0.6, scaleHeight: 0.02, rMax: 3 };
  const b = makeGalaxy(model, seededRandom(1));
  setCircularVelocities(b, 0.05);
  const e0 = totalEnergy(b, 0.05);
  const p0 = totalMomentum(b);
  for (let s = 0; s < 1000; s++) leapfrogStep(b, 0.005, 0.05);
  assert.ok(Math.abs((totalEnergy(b, 0.05) - e0) / e0) < 1e-3);
  const p1 = totalMomentum(b);
  assert.ok(Math.hypot(p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]) < 1e-12);
});

test("a massless disc around a point mass moves at the Kepler circular speed", () => {
  const model = { nDisc: 50, centralMass: 1, discMass: 0, scaleLength: 0.6, scaleHeight: 0, rMax: 3 };
  const b = makeGalaxy(model, seededRandom(3));
  setCircularVelocities(b, 0);
  for (let i = 1; i < b.n; i++) {
    const radius = Math.hypot(b.pos[3 * i], b.pos[3 * i + 1]);
    const speed = Math.hypot(b.vel[3 * i], b.vel[3 * i + 1]);
    assert.ok(Math.abs(speed - 1 / Math.sqrt(radius)) < 1e-12 * speed + 1e-12);
  }
});

test("a parabolic orbit has zero energy and the requested separation", () => {
  const { pos, vel } = parabolicOrbit(2.4, 1.5, 9);
  assert.ok(Math.abs(Math.hypot(...pos) - 9) < 1e-12);
  assert.ok(Math.abs(0.5 * (vel[0] ** 2 + vel[1] ** 2) - 2.4 / 9) < 1e-12);
  assert.ok(pos[0] * vel[0] + pos[1] * vel[1] < 0); // approaching
});

test("combining two galaxies keeps every body and marks the second group", () => {
  const model = { nDisc: 10, centralMass: 1, discMass: 0.2, scaleLength: 0.6, scaleHeight: 0.02, rMax: 3 };
  const both = combine(makeGalaxy(model, seededRandom(1)), makeGalaxy(model, seededRandom(2)));
  assert.equal(both.n, 22);
  assert.equal(both.group.reduce((a, g) => a + g, 0), 11);
  assert.ok(Math.abs(both.mass.reduce((a, m) => a + m, 0) - 2.4) < 1e-12);
});
