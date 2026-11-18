// Specs: the names a server gives its measures. A spec is path segments: the
// first is the source, the first parameter's value, and each other is
// name=value for one of the rest, its value percent-encoded; a parameter
// left out has its default. The server's catalog says what parameters there
// are, and the page builds specs from them.

export type Parameter =
  | { name: string; kind: "choice"; options: string[]; default: string }
  /** A finite number, or none where the default is null. */
  | { name: string; kind: "number"; default: number | null }
  | { name: string; kind: "text" }
  /** A recording uploaded to the server, by id, or none. */
  | { name: string; kind: "recording" };

export interface Catalog {
  /** In spec order; the first, the source, is a choice. */
  parameters: Parameter[];
  /** The spec to start from. */
  start: string;
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

/** What ``spec`` says of each parameter. A part no parameter takes is left
 * for the server to refuse. */
export function parse(catalog: Catalog, spec: string): Values {
  const [source = "", ...segments] = spec.split("/");
  const values = new Map(catalog.parameters.map((parameter) => [parameter.name, defaultValue(parameter)]));
  values.set(catalog.parameters[0]!.name, decodeURIComponent(source));
  for (const segment of segments) {
    const equals = segment.indexOf("=");
    if (equals < 0) continue;
    values.set(segment.slice(0, equals), decodeURIComponent(segment.slice(equals + 1)));
  }
  return values;
}

/** The spec of ``values``, leaving out each parameter at its default. */
export function format(catalog: Catalog, values: Values): string {
  const [source, ...rest] = catalog.parameters;
  const segments = [encodeURIComponent(values.get(source!.name) ?? defaultValue(source!))];
  for (const parameter of rest) {
    const value = values.get(parameter.name) ?? "";
    if (atDefault(parameter, value)) continue;
    segments.push(`${parameter.name}=${encodeURIComponent(value)}`);
  }
  return segments.join("/");
}

/** Whether a spec leaves ``value`` out, as the parameter's default. */
export function atDefault(parameter: Parameter, value: string): boolean {
  if (parameter.kind !== "number") return value === defaultValue(parameter);
  return value.trim() === "" || Number(value) === parameter.default;
}

/** A catalog of nothing but sources: a recording's measures, by name. */
export function named(names: string[]): Catalog {
  return { parameters: [{ name: "measure", kind: "choice", options: names, default: names[0] ?? "" }], start: encodeURIComponent(names[0] ?? "") };
}
