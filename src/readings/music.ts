// Naming and coloring music: pitches, instruments and times.

import type { Selection, TickScale } from "@qbar/qbar";

export const DRUM_PROGRAM = 128;
const NAMES = ["C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B"];

export function pitchName(pitch: number): string {
  return `${NAMES[pitch % 12]}${Math.floor(pitch / 12) - 1}`;
}

/** Each pitch class's tier in C major, coarsest first: the octave, its
 * fifth, the rest of its triad, the rest of the scale, then the black keys.
 * Each tier holds the ones before it, so the tiers nest as qbar's steps must. */
const TIERS = [0, 4, 3, 4, 2, 3, 4, 1, 4, 3, 4, 3];
/** Semitones from each pitch class to its nearest neighbour as prominent or
 * more. The tiers are not uniform, so the room is the smaller of the gaps on
 * either side: a tick's spacing is how crowded its closer neighbour makes it. */
const ROOM = TIERS.map((tier, pc) => {
  for (let gap = 1; ; gap++) if (TIERS[(pc + gap) % 12]! <= tier || TIERS[(pc + 12 - gap) % 12]! <= tier) return gap;
});

/** MIDI pitches, each tick its tier in C major. */
export const pitchTicks: TickScale<number> = {
  select(lower, upper) {
    let best: Selection<number> | undefined;
    // A C ends the search: none is coarser, and one lies within any 12 pitches.
    for (let pitch = Math.min(127, Math.floor(upper)); lower < pitch && 0 <= pitch; pitch--) {
      const tier = TIERS[pitch % 12]!;
      if (best === undefined || tier < best.step) best = { value: pitch, step: tier, size: ROOM[pitch % 12]! };
      if (tier === 0) break;
    }
    return best;
  },
  format: pitchName,
};

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
