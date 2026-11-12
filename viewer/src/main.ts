// Entry point: wires the page's buttons to the two modes.
//   Live      a simulation computed in this browser, step by step (live.ts, physics.ts)
//   Recorded  playback of the 100,000-body run made with the Python code (replay.ts)

import { $ } from "./dom.js";
import { LiveSimulation } from "./live.js";
import { Player, loadPlayer } from "./replay.js";

function canvasContext(id: string): CanvasRenderingContext2D {
  const ctx = ($(id) as HTMLCanvasElement).getContext("2d");
  if (!ctx) throw new Error("no 2D canvas");
  return ctx;
}

const live = new LiveSimulation(canvasContext("live-sky"), readout => {
  $("live-readout").textContent = `${readout.bodies.toLocaleString()} bodies · t = ${readout.time.toFixed(1)} · `
    + `${readout.msPerStep.toFixed(1)} ms per step · energy drift ${readout.energyDrift.toExponential(1)}`;
});
let player: Player | null = null;

function bindLiveControls(): void {
  const pause = $("live-pause") as HTMLButtonElement;
  const collide = $("live-collide") as HTMLButtonElement;
  pause.onclick = () => (pause.textContent = live.togglePause() ? "Play" : "Pause");
  collide.onclick = () => {
    live.addCompanion();
    collide.disabled = true;
  };
  ($("live-reset") as HTMLButtonElement).onclick = () => {
    live.reset();
    collide.disabled = false;
  };
  const count = $("live-count") as HTMLSelectElement;
  count.onchange = () => {
    live.setBodyCount(Number(count.value));
    collide.disabled = false;
  };
  const speed = $("live-speed") as HTMLSelectElement;
  speed.onchange = () => live.setStepsPerFrame(Number(speed.value));
  const trails = $("live-trails") as HTMLInputElement;
  trails.onchange = () => live.setTrails(trails.checked);
}

/** Show one mode and stop the other, so only one animation loop runs at a time. */
async function showMode(mode: "live" | "replay"): Promise<void> {
  $("live-panel").hidden = mode !== "live";
  $("replay-panel").hidden = mode !== "replay";
  $("tab-live").setAttribute("aria-pressed", String(mode === "live"));
  $("tab-replay").setAttribute("aria-pressed", String(mode === "replay"));
  if (mode === "live") {
    player?.stop();
    live.start();
    return;
  }
  live.stop();
  try {
    player ??= await loadPlayer($("sky") as HTMLCanvasElement); // the 9 MB of frames load only when asked for
    player.start();
  } catch (err) {
    $("replay-info").textContent = `Could not load the recording: ${(err as Error).message}`;
  }
}

bindLiveControls();
$("tab-live").onclick = () => showMode("live");
$("tab-replay").onclick = () => showMode("replay");
showMode("live");
