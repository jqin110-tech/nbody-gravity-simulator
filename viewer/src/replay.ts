// Plays the recorded 100,000-body run: frames exported by scripts/stage5_render.py.
// data/meta.json describes the run; data/frames.bin holds, for every frame, every particle's
// (x, y) as little-endian int16 scaled so that +-32767 means +-meta.box; data/groups.bin
// holds one byte per particle: 0/2 = halo of galaxy 1/2, 1/3 = disc of galaxy 1/2.

interface Meta {
  n_particles: number;
  n_frames: number;
  box: number;
  times: number[];
  n_bodies_simulated: number;
}

// RGB colour added to a pixel by one particle of each group.
const GROUP_COLOURS: [number, number, number][] = [
  [70, 50, 35], [255, 160, 80], [35, 50, 70], [90, 160, 255],
];

import { $ } from "./dom.js";

async function fetchBytes(url: string): Promise<ArrayBuffer> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  return response.arrayBuffer();
}

export class Player {
  private frame = 0;
  private playing = true;
  private running = false;
  private speed = 1;
  private showHalo = false;
  private lastTick = 0;
  private readonly image: ImageData;
  private readonly light: Float32Array; // accumulated brightness, 3 channels per pixel

  constructor(
    private readonly meta: Meta,
    private readonly frames: Int16Array,
    private readonly groups: Uint8Array,
    private readonly ctx: CanvasRenderingContext2D,
  ) {
    const { width, height } = ctx.canvas;
    this.image = ctx.createImageData(width, height);
    this.light = new Float32Array(width * height * 3);
  }

  /** Splat every particle of one frame into the light buffer, then tone-map it to pixels. */
  draw(): void {
    const { width, height } = this.ctx.canvas;
    const n = this.meta.n_particles;
    const offset = this.frame * n * 2;
    this.light.fill(0);
    for (let i = 0; i < n; i++) {
      const g = this.groups[i];
      if (!this.showHalo && (g === 0 || g === 2)) continue;
      const px = Math.floor(((this.frames[offset + 2 * i] / 32767) * 0.5 + 0.5) * width);
      const py = Math.floor((0.5 - (this.frames[offset + 2 * i + 1] / 32767) * 0.5) * height);
      if (px < 0 || px + 1 >= width || py < 0 || py + 1 >= height) continue;
      const c = GROUP_COLOURS[g];
      // Each particle lights a 2 x 2 block of pixels.
      for (const k of [py * width + px, py * width + px + 1, (py + 1) * width + px, (py + 1) * width + px + 1]) {
        this.light[3 * k] += c[0];
        this.light[3 * k + 1] += c[1];
        this.light[3 * k + 2] += c[2];
      }
    }
    const px = this.image.data;
    for (let p = 0, k = 0; p < width * height; p++, k += 3) {
      // 1 - exp(-x) keeps a lone particle coloured and lets dense cores saturate smoothly to white.
      px[4 * p] = 255 * (1 - Math.exp(-this.light[k] / 180));
      px[4 * p + 1] = 255 * (1 - Math.exp(-this.light[k + 1] / 180));
      px[4 * p + 2] = 255 * (1 - Math.exp(-this.light[k + 2] / 180));
      px[4 * p + 3] = 255;
    }
    this.ctx.putImageData(this.image, 0, 0);
    ($("scrub") as HTMLInputElement).value = String(this.frame);
    $("time").textContent = `t = ${this.meta.times[this.frame].toFixed(2)}`;
  }

  start(): void {
    if (this.running) return;
    this.running = true;
    this.draw();
    requestAnimationFrame(this.tick);
  }

  stop(): void { this.running = false; }

  /** Advance at about 20 frames per second times the chosen speed; stop on the last frame instead of looping. */
  private tick = (now: number): void => {
    if (!this.running) return;
    if (this.playing && now - this.lastTick > 50 / this.speed) {
      if (this.frame === this.meta.n_frames - 1) {
        this.setPlaying(false);
      } else {
        this.frame += 1;
        this.draw();
      }
      this.lastTick = now;
    }
    requestAnimationFrame(this.tick);
  };

  private setPlaying(playing: boolean): void {
    this.playing = playing;
    const atEnd = this.frame === this.meta.n_frames - 1;
    $("play").textContent = playing ? "Pause" : atEnd ? "Replay" : "Play";
  }

  bindControls(): void {
    ($("play") as HTMLButtonElement).onclick = () => {
      if (!this.playing && this.frame === this.meta.n_frames - 1) this.frame = 0; // replay from the start
      this.setPlaying(!this.playing);
    };
    const scrub = $("scrub") as HTMLInputElement;
    scrub.max = String(this.meta.n_frames - 1);
    scrub.oninput = () => {
      this.frame = Number(scrub.value);
      this.draw();
    };
    const speed = $("speed") as HTMLSelectElement;
    speed.onchange = () => (this.speed = Number(speed.value));
    const halo = $("halo") as HTMLInputElement;
    halo.onchange = () => {
      this.showHalo = halo.checked;
      this.draw();
    };
  }
}

/** Download the recorded frames and build a player for them. */
export async function loadPlayer(canvas: HTMLCanvasElement): Promise<Player> {
  const meta: Meta = await (await fetch("data/meta.json")).json();
  const [frameBytes, groupBytes] = await Promise.all([fetchBytes("data/frames.bin"), fetchBytes("data/groups.bin")]);
  const frames = new Int16Array(frameBytes);
  if (frames.length !== meta.n_frames * meta.n_particles * 2) throw new Error("frames.bin has the wrong size");
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("no 2D canvas");
  $("replay-info").textContent = `Recorded with the Python code: ${meta.n_bodies_simulated.toLocaleString()} bodies, Barnes-Hut forces. `
    + `Showing ${meta.n_particles.toLocaleString()} of them over ${meta.n_frames} frames.`;
  const player = new Player(meta, frames, new Uint8Array(groupBytes), ctx);
  player.bindControls();
  return player;
}
