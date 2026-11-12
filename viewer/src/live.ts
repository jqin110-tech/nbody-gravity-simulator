// A live N-body simulation that runs in the browser: every frame takes real leapfrog steps
// with direct-sum gravity (physics.ts) and draws the bodies. Nothing here is pre-recorded.

import {
  Bodies, GalaxyModel, combine, computeAccelerations, leapfrogStep, makeGalaxy, parabolicOrbit,
  seededRandom, setCircularVelocities, shift, tiltAboutX, totalEnergy,
} from "./physics.js";

// Model choices (N-body units, G = 1). These are picked for a clear, stable picture.
const CENTRAL_MASS = 1.0;
const DISC_MASS = 0.2;
const SCALE_LENGTH = 0.6;
const SCALE_HEIGHT = 0.02;
const R_MAX = 3.0;
const SOFTENING = 0.05;
const DT = 0.005;
const COMPANION_PERICENTRE = 1.5;
const COMPANION_START = 9.0;
const COMPANION_TILT = Math.PI / 3;
const ENERGY_CHECK_EVERY = 120; // steps between energy measurements (each costs about one force call)

const DISC_COLOURS = ["rgb(255, 150, 70)", "rgb(90, 160, 255)"];

export interface LiveReadout {
  bodies: number;
  time: number;
  msPerStep: number; // time the physics takes for one step, averaged over recent frames
  energyDrift: number;
}

export class LiveSimulation {
  private bodies!: Bodies;
  private time = 0;
  private steps = 0;
  private energyStart = 0;
  private energyDrift = 0;
  private hasCompanion = false;
  private running = false;
  private paused = false;
  private stepsPerFrame = 2;
  private trails = true;
  private nDisc = 1000;
  // Camera: spin about the disc axis, tilt away from face-on, and zoom (pixels per length unit).
  private spin = 0;
  private tilt = 1.1;
  private zoom = 110;
  private stepTimes: number[] = [];

  constructor(private readonly ctx: CanvasRenderingContext2D, private readonly onReadout: (r: LiveReadout) => void) {
    this.reset();
    this.bindCamera();
  }

  private model(): GalaxyModel {
    return { nDisc: this.nDisc, centralMass: CENTRAL_MASS, discMass: DISC_MASS, scaleLength: SCALE_LENGTH,
             scaleHeight: SCALE_HEIGHT, rMax: R_MAX };
  }

  /** Start again from one fresh galaxy. */
  reset(): void {
    this.bodies = makeGalaxy(this.model(), seededRandom(1));
    setCircularVelocities(this.bodies, SOFTENING);
    this.hasCompanion = false;
    this.zoom = 110;
    this.restartClock();
  }

  /** Drop in a second, tilted galaxy on a parabolic collision course with the first. */
  addCompanion(): void {
    if (this.hasCompanion) return;
    const companion = makeGalaxy(this.model(), seededRandom(2));
    setCircularVelocities(companion, SOFTENING);
    tiltAboutX(companion, COMPANION_TILT);
    const orbit = parabolicOrbit(2 * (CENTRAL_MASS + DISC_MASS), COMPANION_PERICENTRE, COMPANION_START);
    // Equal masses: each galaxy sits half the separation from the shared centre of mass.
    const [com, vcom] = this.centreOfMass();
    shift(this.bodies, [0, 1, 2].map(k => -com[k] - 0.5 * orbit.pos[k]), [0, 1, 2].map(k => -vcom[k] - 0.5 * orbit.vel[k]));
    shift(companion, orbit.pos.map(x => 0.5 * x), orbit.vel.map(v => 0.5 * v));
    this.bodies = combine(this.bodies, companion);
    this.hasCompanion = true;
    this.zoom = 55;
    this.restartClock();
  }

  private restartClock(): void {
    computeAccelerations(this.bodies, SOFTENING);
    this.energyStart = totalEnergy(this.bodies, SOFTENING);
    this.energyDrift = 0;
    this.time = 0;
    this.steps = 0;
    this.clearCanvas();
  }

  private centreOfMass(): [number[], number[]] {
    const b = this.bodies;
    const com = [0, 0, 0], vcom = [0, 0, 0];
    let total = 0;
    for (let i = 0; i < b.n; i++) {
      total += b.mass[i];
      for (let k = 0; k < 3; k++) {
        com[k] += b.mass[i] * b.pos[3 * i + k];
        vcom[k] += b.mass[i] * b.vel[3 * i + k];
      }
    }
    return [com.map(x => x / total), vcom.map(v => v / total)];
  }

  setBodyCount(nDisc: number): void { this.nDisc = nDisc; this.reset(); }
  setStepsPerFrame(n: number): void { this.stepsPerFrame = n; }
  setTrails(on: boolean): void { this.trails = on; }
  togglePause(): boolean { this.paused = !this.paused; return this.paused; }
  canAddCompanion(): boolean { return !this.hasCompanion; }

  start(): void {
    if (this.running) return;
    this.running = true;
    this.clearCanvas();
    requestAnimationFrame(this.frame);
  }

  stop(): void { this.running = false; }

  private frame = (): void => {
    if (!this.running) return;
    if (!this.paused) {
      const begin = performance.now();
      for (let s = 0; s < this.stepsPerFrame; s++) this.step();
      this.recordSpeed((performance.now() - begin) / this.stepsPerFrame);
    }
    this.draw();
    this.onReadout({ bodies: this.bodies.n, time: this.time, msPerStep: this.msPerStep(), energyDrift: this.energyDrift });
    requestAnimationFrame(this.frame);
  };

  private step(): void {
    leapfrogStep(this.bodies, DT, SOFTENING);
    this.time += DT;
    this.steps += 1;
    if (this.steps % ENERGY_CHECK_EVERY === 0) {
      this.energyDrift = Math.abs((totalEnergy(this.bodies, SOFTENING) - this.energyStart) / this.energyStart);
    }
  }

  private recordSpeed(msPerStep: number): void {
    this.stepTimes.push(msPerStep);
    if (this.stepTimes.length > 60) this.stepTimes.shift();
  }

  private msPerStep(): number {
    if (this.stepTimes.length === 0) return 0;
    return this.stepTimes.reduce((a, b) => a + b, 0) / this.stepTimes.length;
  }

  private clearCanvas(): void {
    this.ctx.globalCompositeOperation = "source-over";
    this.ctx.fillStyle = "#000";
    this.ctx.fillRect(0, 0, this.ctx.canvas.width, this.ctx.canvas.height);
  }

  /** Project every body through the camera and draw it; old frames fade instead of vanishing, leaving trails. */
  private draw(): void {
    const { ctx, bodies: b } = this;
    const { width, height } = ctx.canvas;
    ctx.globalCompositeOperation = "source-over";
    ctx.fillStyle = this.trails ? "rgba(0, 0, 0, 0.22)" : "#000";
    ctx.fillRect(0, 0, width, height);
    ctx.globalCompositeOperation = "lighter"; // overlapping bodies add up to brighter light
    const cosSpin = Math.cos(this.spin), sinSpin = Math.sin(this.spin);
    const cosTilt = Math.cos(this.tilt), sinTilt = Math.sin(this.tilt);
    for (let pass = 0; pass < 2; pass++) { // pass 0: disc bodies, pass 1: the heavy centres on top
      for (let i = 0; i < b.n; i++) {
        const heavy = b.mass[i] >= CENTRAL_MASS;
        if (heavy !== (pass === 1)) continue;
        const x = b.pos[3 * i], y = b.pos[3 * i + 1], z = b.pos[3 * i + 2];
        const xr = x * cosSpin - y * sinSpin;
        const yr = x * sinSpin + y * cosSpin;
        const sx = width / 2 + this.zoom * xr;
        const sy = height / 2 - this.zoom * (yr * cosTilt + z * sinTilt);
        if (heavy) {
          const glow = ctx.createRadialGradient(sx, sy, 0, sx, sy, 9);
          glow.addColorStop(0, "rgba(255, 255, 255, 1)");
          glow.addColorStop(1, "rgba(255, 255, 255, 0)");
          ctx.fillStyle = glow;
          ctx.fillRect(sx - 9, sy - 9, 18, 18);
        } else {
          ctx.fillStyle = DISC_COLOURS[b.group[i]];
          ctx.fillRect(sx - 1, sy - 1, 2, 2);
        }
      }
    }
  }

  /** Drag to turn the camera, scroll to zoom. */
  private bindCamera(): void {
    const canvas = this.ctx.canvas;
    let dragging = false, lastX = 0, lastY = 0;
    canvas.addEventListener("pointerdown", e => { dragging = true; lastX = e.clientX; lastY = e.clientY; canvas.setPointerCapture(e.pointerId); });
    canvas.addEventListener("pointerup", () => { dragging = false; });
    canvas.addEventListener("pointermove", e => {
      if (!dragging) return;
      this.spin += 0.01 * (e.clientX - lastX);
      this.tilt = Math.min(Math.PI / 2, Math.max(0, this.tilt + 0.01 * (e.clientY - lastY)));
      lastX = e.clientX;
      lastY = e.clientY;
    });
    canvas.addEventListener("wheel", e => {
      e.preventDefault();
      this.zoom = Math.min(600, Math.max(15, this.zoom * Math.exp(-0.001 * e.deltaY)));
    }, { passive: false });
  }
}
