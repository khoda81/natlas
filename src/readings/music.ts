// Naming and coloring music: pitches, instruments and times.

export const DRUM_PROGRAM = 128;
const NAMES = ["C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B"];

export function pitchName(pitch: number): string {
  return `${NAMES[pitch % 12]}${Math.floor(pitch / 12) - 1}`;
}

/** One color per instrument, spread around the hue circle at equal lightness. */
export function tint(program: number): string {
  return program === DRUM_PROGRAM ? "oklch(0.74 0.05 70)" : `oklch(0.74 0.15 ${(program * 137.508 + 20) % 360})`;
}

/** A time in the recording, to the hundredth: seconds, or minutes and seconds past a minute. */
export function clock(seconds: number, digits = 2): string {
  if (seconds < 60) return `${seconds.toFixed(digits)} s`;
  const minutes = Math.floor(seconds / 60);
  const rest = (seconds - minutes * 60).toFixed(digits);
  return `${minutes}:${rest.padStart(digits === 0 ? 2 : digits + 3, "0")}`;
}
