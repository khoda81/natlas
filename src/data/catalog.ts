// Specs: the names a server gives its models and measures. A model is
// weights loaded on a device at a precision, and its spec is each of its
// parameters in turn, as path segments: large/cuda/float16. A measure is a
// model read one way, and its spec is its model's, then name=value for each
// of its own parameters not at its default, the value percent-encoded:
// large/cuda/float16/temperature=3. The server's catalog says what
// parameters there are, and the page builds specs from them.

export type Parameter =
  | { name: string; kind: "choice"; options: string[]; default: string }
  /** A finite number, or none where the default is null. */
  | { name: string; kind: "number"; default: number | null }
  | { name: string; kind: "text" }
  /** A recording uploaded to the server, by id, or none. */
  | { name: string; kind: "recording" };

export type Choice = Extract<Parameter, { kind: "choice" }>;

export interface Catalog {
  /** A model's parameters, in order. */
  models: Choice[];
  /** A measure's own parameters. */
  parameters: Parameter[];
  /** Models offered as they are, by spec. */
  presets: string[];
}

/** Each parameter's value as a spec writes it, the default where it is left out. */
export type Values = ReadonlyMap<string, string>;

/** The value a spec leaving the parameter out has. */
export function defaultValue(parameter: Parameter): string {
  switch (parameter.kind) {
    case "choice":
      return parameter.default;
    case "number":
      return parameter.default === null ? "" : String(parameter.default);
    case "text":
    case "recording":
      return "";
  }
}

/** Whether a spec leaves ``value`` out, as the parameter's default. */
export function atDefault(parameter: Parameter, value: string): boolean {
  if (parameter.kind !== "number") return value === defaultValue(parameter);
  return value.trim() === "" || Number(value) === parameter.default;
}

/** The spec of the model a measure's spec names. */
export function modelOf(catalog: Catalog, spec: string): string {
  return spec.split("/").slice(0, catalog.models.length).join("/");
}

/** What ``spec`` says of each parameter, the model's and the measure's. A
 * part no parameter takes is left for the server to refuse. */
export function parse(catalog: Catalog, spec: string): Values {
  const segments = spec.split("/");
  const values = new Map(catalog.parameters.map((parameter) => [parameter.name, defaultValue(parameter)]));
  catalog.models.forEach((parameter, index) => values.set(parameter.name, decodeURIComponent(segments[index] ?? "")));
  for (const segment of segments.slice(catalog.models.length)) {
    const equals = segment.indexOf("=");
    if (equals < 0) continue;
    values.set(segment.slice(0, equals), decodeURIComponent(segment.slice(equals + 1)));
  }
  return values;
}

/** The spec of a measure ``values`` names, leaving out each of its own
 * parameters at its default. */
export function format(catalog: Catalog, values: Values): string {
  const segments = catalog.models.map((parameter) => encodeURIComponent(values.get(parameter.name) ?? parameter.default));
  for (const parameter of catalog.parameters) {
    const value = values.get(parameter.name) ?? "";
    if (atDefault(parameter, value)) continue;
    segments.push(`${parameter.name}=${encodeURIComponent(value)}`);
  }
  return segments.join("/");
}

/** A catalog of nothing but models, each its only measure: a recording's
 * measures, by name. */
export function named(names: string[]): Catalog {
  return { models: [{ name: "measure", kind: "choice", options: names, default: names[0] ?? "" }], parameters: [], presets: [] };
}
