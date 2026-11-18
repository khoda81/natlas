// The canvas's colors, each thing its own: the background for empty space,
// one quiet fill for the inside of every node's children, a gray outline
// around them, ink for labels and time, a quieter fill under time windows,
// and one saturated color, the hover's, which nothing else uses. Color says nothing about the measures:
// height is p, width is −ln q, and their product the area.

type Rgb = readonly [number, number, number];

export interface Scheme {
  surface: string;
  ink: string;
  inkMuted: string;
  inkOnDark: string;
  inkOnLight: string;
  /** The inside of every node's children. */
  tile: string;
  /** The outline around each node's children, apart from fill and background. */
  stroke: string;
  /** Under a time window: further from the ink than the fill, so labels on
   * its shaded time still read. */
  window: string;
  /** The hovered branch and the path to it. */
  hover: string;
}

const LIGHT: Scheme = {
  surface: "#fcfcfb",
  ink: "#0b0b0b",
  inkMuted: "#52514e",
  inkOnDark: "#f0efec",
  inkOnLight: "#0b0b0b",
  tile: "#e4e8ed",
  stroke: "#a3abb6",
  window: "#f4f6f8",
  hover: "#c26a00",
};

const DARK: Scheme = {
  surface: "#1a1a19",
  ink: "#f0efec",
  inkMuted: "#c3c2b7",
  inkOnDark: "#f0efec",
  inkOnLight: "#0b0b0b",
  tile: "#30343b",
  stroke: "#5f6773",
  window: "#202328",
  hover: "#f0a040",
};

function parse(hex: string): Rgb {
  const value = Number.parseInt(hex.slice(1), 16);
  return [(value >> 16) & 255, (value >> 8) & 255, value & 255];
}

function luminance([r, g, b]: Rgb): number {
  const channel = (c: number) => {
    const s = c / 255;
    return s <= 0.04045 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4;
  };
  return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b);
}

export class Palette {
  /** The ink labels read in on the fill. */
  readonly ink: string;

  constructor(readonly scheme: Scheme) {
    this.ink = luminance(parse(scheme.tile)) > 0.3 ? scheme.inkOnLight : scheme.inkOnDark;
  }

  static current(): Palette {
    const theme = document.documentElement.dataset.theme;
    const dark = theme === "dark" || (theme !== "light" && matchMedia("(prefers-color-scheme: dark)").matches);
    return new Palette(dark ? DARK : LIGHT);
  }
}
