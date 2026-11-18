// The canvas's colors, each thing its own: the background for empty space,
// quiet tiles, a gray line around each node's children, ink for labels and
// time, and one saturated color, the hover's, which nothing else uses.
// With p = q every box has one fill, and its labels one ink: width already
// shows the weight. Otherwise a box's color shows the per-slot log ratio
// ln(p / q) on a blue-gray-red diverging ramp, red where q under-weights the
// slot.

type Rgb = readonly [number, number, number];

export interface Scheme {
  surface: string;
  ink: string;
  inkMuted: string;
  inkOnDark: string;
  inkOnLight: string;
  /** Every box's fill with p = q. */
  tile: string;
  /** The line around each node's children, apart from tiles and background. */
  stroke: string;
  /** The hovered branch and the path to it. */
  hover: string;
  diverging: string[];
}

const LIGHT: Scheme = {
  surface: "#fcfcfb",
  ink: "#0b0b0b",
  inkMuted: "#52514e",
  inkOnDark: "#f0efec",
  inkOnLight: "#0b0b0b",
  tile: "#e4e8ed",
  stroke: "#a3abb6",
  hover: "#c26a00",
  diverging: ["#1c5cab", "#5598e7", "#b7d3f6", "#f0efec", "#f6c4c1", "#e34948", "#a92b2b"],
};

const DARK: Scheme = {
  surface: "#1a1a19",
  ink: "#f0efec",
  inkMuted: "#c3c2b7",
  inkOnDark: "#f0efec",
  inkOnLight: "#0b0b0b",
  tile: "#30343b",
  stroke: "#5f6773",
  hover: "#f0a040",
  diverging: ["#9ec5f4", "#5598e7", "#2f5f9e", "#383835", "#8a3535", "#e34948", "#f4a3a1"],
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

/** A box's fill and the ink its label reads in. */
export interface Tile {
  fill: string;
  ink: string;
}

function tileOf(rgb: Rgb, scheme: Scheme): Tile {
  return { fill: css(rgb), ink: luminance(rgb) > 0.3 ? scheme.inkOnLight : scheme.inkOnDark };
}

export class Palette {
  private readonly flat: Tile;
  /** The diverging ramp's tiles, precomputed at LEVELS steps. */
  private readonly ramp: Tile[] = [];

  constructor(readonly scheme: Scheme) {
    this.flat = tileOf(parse(scheme.tile), scheme);
    const colors = scheme.diverging.map(parse);
    for (let level = 0; level < LEVELS; level++) {
      const position = (level / (LEVELS - 1)) * (colors.length - 1);
      const index = Math.min(colors.length - 2, Math.floor(position));
      this.ramp.push(tileOf(mix(colors[index]!, colors[index + 1]!, position - index), scheme));
    }
  }

  /** A box's tile: the one fill with p = q, else its place on the
   * diverging ramp in ln(p / q), saturating at ±DIVERGING_NATS. */
  tile(p: number, q: number, equal: boolean): Tile {
    if (equal) return this.flat;
    const t = 0.5 + Math.log(p / q) / (2 * DIVERGING_NATS);
    return this.ramp[Math.round(Math.min(1, Math.max(0, t)) * (LEVELS - 1))]!;
  }

  static current(): Palette {
    const theme = document.documentElement.dataset.theme;
    const dark = theme === "dark" || (theme !== "light" && matchMedia("(prefers-color-scheme: dark)").matches);
    return new Palette(dark ? DARK : LIGHT);
  }
}

/** Log ratios saturate at ±DIVERGING_NATS. */
export const DIVERGING_NATS = 3;
