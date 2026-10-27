// Box colors. With p = q, color repeats the conditional weight on a one-hue
// sequential ramp; otherwise it shows the per-slot log ratio ln(p / q) on a
// blue-gray-red diverging ramp, red where q under-weights the slot.

type Rgb = readonly [number, number, number];

export interface Scheme {
  surface: string;
  ink: string;
  inkMuted: string;
  inkOnDark: string;
  inkOnLight: string;
  sequential: string[];
  diverging: string[];
  merged: string;
  tail: string;
  /** How far an unexpanded slot fades toward the surface. */
  leafFade: number;
}

const LIGHT: Scheme = {
  surface: "#fcfcfb",
  ink: "#0b0b0b",
  inkMuted: "#52514e",
  inkOnDark: "#f0efec",
  inkOnLight: "#0b0b0b",
  sequential: ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#104281"],
  diverging: ["#1c5cab", "#5598e7", "#b7d3f6", "#f0efec", "#f6c4c1", "#e34948", "#a92b2b"],
  merged: "#dcdbd6",
  tail: "#c9c8c1",
  leafFade: 0.45,
};

const DARK: Scheme = {
  surface: "#1a1a19",
  ink: "#f0efec",
  inkMuted: "#c3c2b7",
  inkOnDark: "#f0efec",
  inkOnLight: "#0b0b0b",
  sequential: ["#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4"],
  diverging: ["#9ec5f4", "#5598e7", "#2f5f9e", "#383835", "#8a3535", "#e34948", "#f4a3a1"],
  merged: "#3a3a37",
  tail: "#4a4a45",
  leafFade: 0.45,
};

const LEVELS = 256;

function parse(hex: string): Rgb {
  const value = Number.parseInt(hex.slice(1), 16);
  return [(value >> 16) & 255, (value >> 8) & 255, value & 255];
}

function mix(a: Rgb, b: Rgb, t: number): Rgb {
  return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
}

function css([r, g, b]: Rgb): string {
  return `rgb(${Math.round(r)} ${Math.round(g)} ${Math.round(b)})`;
}

function luminance([r, g, b]: Rgb): number {
  const channel = (c: number) => {
    const s = c / 255;
    return s <= 0.04045 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4;
  };
  return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b);
}

/** Precomputed fills and matching label inks for one ramp, solid and faded. */
class Ramp {
  readonly fill: string[] = [];
  readonly faded: string[] = [];
  readonly ink: string[] = [];
  readonly fadedInk: string[] = [];

  constructor(stops: string[], scheme: Scheme) {
    const colors = stops.map(parse);
    const surface = parse(scheme.surface);
    const inkFor = (rgb: Rgb) => (luminance(rgb) > 0.3 ? scheme.inkOnLight : scheme.inkOnDark);
    for (let level = 0; level < LEVELS; level++) {
      const position = (level / (LEVELS - 1)) * (colors.length - 1);
      const index = Math.min(colors.length - 2, Math.floor(position));
      const rgb = mix(colors[index]!, colors[index + 1]!, position - index);
      const faded = mix(rgb, surface, scheme.leafFade);
      this.fill.push(css(rgb));
      this.faded.push(css(faded));
      this.ink.push(inkFor(rgb));
      this.fadedInk.push(inkFor(faded));
    }
  }

  level(t: number): number {
    return Math.round(Math.min(1, Math.max(0, t)) * (LEVELS - 1));
  }
}

export class Palette {
  readonly sequential: Ramp;
  readonly diverging: Ramp;
  readonly mergedInk: string;
  readonly tailInk: string;

  constructor(readonly scheme: Scheme) {
    this.sequential = new Ramp(scheme.sequential, scheme);
    this.diverging = new Ramp(scheme.diverging, scheme);
    const ink = (hex: string) => (luminance(parse(hex)) > 0.3 ? scheme.inkOnLight : scheme.inkOnDark);
    this.mergedInk = ink(scheme.merged);
    this.tailInk = ink(scheme.tail);
  }

  static current(): Palette {
    const theme = document.documentElement.dataset.theme;
    const dark = theme === "dark" || (theme !== "light" && matchMedia("(prefers-color-scheme: dark)").matches);
    return new Palette(dark ? DARK : LIGHT);
  }
}

/** Conditional weights shown in full color down to e^-SEQUENTIAL_NATS. */
export const SEQUENTIAL_NATS = 7;
/** Log ratios saturate at ±DIVERGING_NATS. */
export const DIVERGING_NATS = 3;

/** Ramp position of a slot: sequential in ln p, or diverging in ln(p / q). */
export function rampPosition(p: number, q: number, equal: boolean): number {
  if (equal) return 1 + Math.log(p) / SEQUENTIAL_NATS;
  return 0.5 + Math.log(p / q) / (2 * DIVERGING_NATS);
}
