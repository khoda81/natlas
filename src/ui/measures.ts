// Choosing a measure: its source inline, and what else its spec says in a
// popover beside it, a field for each of the catalog's parameters. A
// recording is any the server holds, or one uploaded from here.

import { atDefault, type Catalog, format, type Parameter, parse, type Values } from "../data/catalog";
import type { Recording, Recordings } from "../data/source";

const UPLOAD = "\u0000upload";

/** ``spec`` in words: the source, then each parameter not at its default,
 * a recording by its name. */
export function describe(catalog: Catalog, spec: string, recordings: Recordings | null): string {
  const values = parse(catalog, spec);
  const [source, ...rest] = catalog.parameters;
  const parts = [values.get(source!.name)!];
  for (const parameter of rest) {
    const value = values.get(parameter.name)!;
    if (atDefault(parameter, value)) continue;
    parts.push(`${parameter.name} ${parameter.kind === "recording" ? recordingName(value, recordings) : value}`);
  }
  return parts.join(" · ");
}

/** The recording ``spec`` hears, where the server is known to hold it. */
export function heard(catalog: Catalog, spec: string, recordings: Recordings | null): Recording | null {
  const parameter = catalog.parameters.find(({ kind }) => kind === "recording");
  if (parameter === undefined || recordings === null) return null;
  return recordings.known.get(parse(catalog, spec).get(parameter.name)!) ?? null;
}

function recordingName(id: string, recordings: Recordings | null): string {
  return id === "" ? "silence" : (recordings?.known.get(id)?.name ?? id);
}

export class MeasurePicker {
  readonly element = document.createElement("span");
  private readonly source = document.createElement("select");
  private readonly options = document.createElement("button");
  private readonly panel = document.createElement("div");
  private readonly fields = new Map<string, HTMLInputElement | HTMLSelectElement>();
  /** The fields choosing a recording. */
  private readonly heard: HTMLSelectElement[] = [];
  private spec = "";

  constructor(
    private readonly label: string,
    private readonly catalog: Catalog,
    private readonly recordings: Recordings | null,
    /** Told the spec chosen. */
    private readonly chosen: (spec: string) => void,
    private readonly failed: (error: unknown) => void,
  ) {
    const [source, ...rest] = catalog.parameters;
    if (source?.kind !== "choice") throw new Error("a catalog's first parameter, the source, is a choice");
    this.element.className = "measure";
    const name = document.createElement("b");
    name.textContent = label;
    this.source.setAttribute("aria-label", `${label}: ${source.name}`);
    this.source.append(...source.options.map((option) => new Option(option, option)));
    this.source.addEventListener("change", () => this.commit());
    this.element.append(name, this.source);
    if (rest.length === 0) return;

    this.panel.popover = "auto";
    this.panel.className = "measure-panel";
    this.panel.id = `measure-${label}`;
    for (const parameter of rest) this.panel.append(this.field(parameter));
    this.options.type = "button";
    this.options.className = "measure-options";
    this.options.setAttribute("popovertarget", this.panel.id);
    this.options.title = `what else ${label} is conditioned on: ${rest.map(({ name }) => name).join(", ")}`;
    this.panel.addEventListener("toggle", (event) => {
      if ((event as ToggleEvent).newState !== "open") return;
      this.listRecordings();
      const rect = this.options.getBoundingClientRect();
      this.panel.style.left = `${Math.max(8, Math.min(rect.left, innerWidth - this.panel.offsetWidth - 8))}px`;
      this.panel.style.top = `${rect.bottom + 4}px`;
    });
    this.element.append(this.options, this.panel);
  }

  /** Show ``spec`` as chosen. */
  set(spec: string): void {
    this.spec = spec;
    const values = parse(this.catalog, spec);
    this.source.value = values.get(this.catalog.parameters[0]!.name)!;
    for (const select of this.heard) {
      // Silence, each recording the server is known to hold, the one chosen
      // though it is not, and a way to upload another.
      const ids = new Set(["", ...(this.recordings?.known.keys() ?? []), values.get(select.dataset.parameter!)!]);
      select.replaceChildren(...[...ids].map((id) => new Option(recordingName(id, this.recordings), id)), new Option("upload…", UPLOAD));
    }
    for (const [name, field] of this.fields) field.value = values.get(name)!;
    const said = describe(this.catalog, spec, this.recordings).split(" · ").slice(1).join(" · ");
    this.options.textContent = said === "" ? "…" : said;
    this.options.classList.toggle("set", said !== "");
  }

  /** Use the recording ``file``, uploading it. */
  async hear(file: File): Promise<void> {
    const parameter = this.catalog.parameters.find(({ kind }) => kind === "recording");
    if (parameter === undefined || this.recordings === null) throw new Error("this server's measures hear no recordings");
    const recording = await this.recordings.upload(file);
    const values = new Map(parse(this.catalog, this.spec));
    values.set(parameter.name, recording.id);
    this.chosen(format(this.catalog, values));
  }

  private field(parameter: Parameter): HTMLElement {
    const label = document.createElement("label");
    const name = document.createElement("span");
    name.textContent = parameter.name;
    let field: HTMLInputElement | HTMLSelectElement;
    switch (parameter.kind) {
      case "choice": {
        field = document.createElement("select");
        field.append(...parameter.options.map((option) => new Option(option, option)));
        break;
      }
      case "number": {
        field = document.createElement("input");
        field.type = "number";
        field.step = "any";
        field.placeholder = parameter.default === null ? "none" : String(parameter.default);
        break;
      }
      case "text": {
        field = document.createElement("input");
        field.type = "text";
        field.placeholder = "none";
        break;
      }
      case "recording": {
        const select = field = document.createElement("select");
        select.dataset.parameter = parameter.name;
        this.heard.push(select);
        const file = document.createElement("input");
        file.type = "file";
        file.accept = "audio/*";
        file.hidden = true;
        file.addEventListener("change", () => {
          const chosen = file.files?.[0];
          file.value = "";
          if (chosen !== undefined) this.hear(chosen).catch(this.failed);
        });
        select.addEventListener("change", () => {
          if (select.value !== UPLOAD) return;
          // The spec stays as it is until a file is uploaded.
          select.value = parse(this.catalog, this.spec).get(parameter.name)!;
          file.click();
        });
        label.append(file);
        break;
      }
    }
    field.setAttribute("aria-label", `${this.label}: ${parameter.name}`);
    field.addEventListener("change", () => {
      if (field.value !== UPLOAD) this.commit();
    });
    this.fields.set(parameter.name, field);
    label.prepend(name, field);
    return label;
  }

  /** The spec the fields say. */
  private commit(): void {
    const values = new Map<string, string>([[this.catalog.parameters[0]!.name, this.source.value]]);
    for (const [name, field] of this.fields) values.set(name, field.value);
    this.chosen(format(this.catalog, values as Values));
  }

  /** Offer every recording the server holds. */
  private listRecordings(): void {
    if (this.recordings === null) return;
    this.recordings.list().then(() => this.set(this.spec)).catch(this.failed);
  }
}
