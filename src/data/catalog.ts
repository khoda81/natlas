// Specs: the names a server gives its models and measures. A model spec is
// canonical k=v pairs joined by '&' (omitting defaults, keeping the first
// parameter when all are at default). A measure spec is '<model_spec>' when
// all measure parameters are at default, or '<model_spec>|<k=v&...>'.

export type Parameter =
  | { name: string; kind: "choice"; options: string[]; default: string }
  /** A finite number, or none where the default is null. */
  | { name: string; kind: "number"; default: number | null; min?: number; max?: number; step?: number; integer?: boolean }
  | { name: string; kind: "text"; default?: string; placeholder?: string; suggestions?: string[] }
  | { name: string; kind: "boolean"; default: boolean }
  /** A recording uploaded to the server, by id, or none. */
  | { name: string; kind: "recording" };

export type Choice = Extract<Parameter, { kind: "choice" }>;

export interface Catalog {
  /** A model's load-time parameters, in order. */
  models: Parameter[];
  /** A measure's read-time parameters, in order. */
  parameters: Parameter[];
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
      return parameter.default ?? "";
    case "boolean":
      return parameter.default ? "true" : "false";
    case "recording":
      return "";
  }
}

/** Whether a spec leaves ``value`` out, as the parameter's default. */
export function atDefault(parameter: Parameter, value: string): boolean {
  if (parameter.kind === "number") {
    return value.trim() === "" || (parameter.default !== null && Number(value) === parameter.default);
  }
  return value === defaultValue(parameter);
}

/** The spec of the model a measure's spec names. */
export function modelOf(_catalog: Catalog, spec: string): string {
  const sep = spec.indexOf("|");
  return sep < 0 ? spec : spec.slice(0, sep);
}

/** What ``spec`` says of each parameter, the model's and the measure's. */
export function parse(catalog: Catalog, spec: string): Values {
  const values = new Map<string, string>();
  for (const parameter of catalog.models) values.set(parameter.name, defaultValue(parameter));
  for (const parameter of catalog.parameters) values.set(parameter.name, defaultValue(parameter));
  const sep = spec.indexOf("|");
  const modelPart = sep < 0 ? spec : spec.slice(0, sep);
  const measurePart = sep < 0 ? "" : spec.slice(sep + 1);
  if (modelPart !== "") {
    for (const [name, value] of new URLSearchParams(modelPart)) values.set(name, value);
  }
  if (measurePart !== "") {
    for (const [name, value] of new URLSearchParams(measurePart)) values.set(name, value);
  }
  return values;
}

/** Format a canonical model spec from ``values``, omitting defaults unless all are at default. */
export function formatModel(catalog: Catalog, values: Values): string {
  const params = new URLSearchParams();
  for (const parameter of catalog.models) {
    const value = values.get(parameter.name) ?? defaultValue(parameter);
    if (atDefault(parameter, value)) continue;
    params.set(parameter.name, value);
  }
  if ([...params.keys()].length === 0 && catalog.models.length > 0) {
    const first = catalog.models[0]!;
    params.set(first.name, values.get(first.name) ?? defaultValue(first));
  }
  return params.toString();
}

/** The spec of a measure ``values`` names, leaving out each parameter at its default. */
export function format(catalog: Catalog, values: Values): string {
  const modelSpec = formatModel(catalog, values);
  const measureParams = new URLSearchParams();
  for (const parameter of catalog.parameters) {
    const value = values.get(parameter.name) ?? defaultValue(parameter);
    if (atDefault(parameter, value)) continue;
    measureParams.set(parameter.name, value);
  }
  const suffix = measureParams.toString();
  return suffix === "" ? modelSpec : `${modelSpec}|${suffix}`;
}

/** A catalog of nothing but models, each its only measure: a recording's measures, by name. */
export function named(names: string[]): Catalog {
  return {
    models: [{ name: "measure", kind: "choice", options: names, default: names[0] ?? "" }],
    parameters: [],
  };
}
