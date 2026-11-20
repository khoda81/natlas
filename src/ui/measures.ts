// Choosing measures. p and q are each one of the measures the server holds.
// Beside them, a panel lists the models it was asked to load, loading,
// loaded or failed, and loads others: one offered as it is, or any its
// parameters make. A measure is made of a loaded model, its parameters set
// in a form, and taken as p or q. A recording is any the server holds, or
// one uploaded from here.

import { atDefault, type Catalog, format, modelOf, type Parameter, parse } from "../data/catalog";
import type { Held, Holdings, Recording, Recordings } from "../data/source";

const UPLOAD = "\u0000upload";
/** How often to ask what the server holds while a model loads. */
const POLL_MS = 1000;

/** ``spec`` in words: its model's parameters, then each of its own not at
 * its default, a recording by its name. */
export function describe(catalog: Catalog, spec: string, recordings: Recordings | null): string {
  const values = parse(catalog, spec);
  const parts = catalog.models.map(({ name }) => values.get(name)!);
  for (const parameter of catalog.parameters) {
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

function button(text: string, title: string, click: () => void): HTMLButtonElement {
  const element = document.createElement("button");
  element.type = "button";
  element.textContent = text;
  element.title = title;
  element.addEventListener("click", click);
  return element;
}

export class Measures {
  /** p's and q's choices. */
  readonly p = document.createElement("select");
  readonly q = document.createElement("select");
  /** Opens the panel; hidden where the server's models cannot change. */
  readonly toggle = document.createElement("button");
  private readonly panel = document.createElement("div");
  private readonly models = document.createElement("div");
  private readonly presets = document.createElement("div");
  /** A model's parameters, to load any. */
  private readonly loading = new Map<string, HTMLSelectElement>();
  /** The measure being made: its model, and its own parameters. */
  private readonly model = document.createElement("select");
  private readonly fields = new Map<string, HTMLInputElement | HTMLSelectElement>();
  private readonly hearing: HTMLSelectElement[] = [];
  private specs = { p: "", q: "" };
  private poll: number | null = null;
  /** What the server held when last asked. */
  held: Held = { models: [], measures: [] };
  /** Told each time the server says what it holds. */
  onHeld: (held: Held) => void = () => {};

  constructor(
    private readonly catalog: Catalog,
    private readonly holdings: Holdings,
    private readonly recordings: Recordings | null,
    /** Told the measure chosen as p or q. */
    private readonly chosen: (which: "p" | "q", spec: string) => void,
    private readonly failed: (error: unknown) => void,
  ) {
    for (const which of ["p", "q"] as const) {
      const select = this[which];
      select.setAttribute("aria-label", which === "p" ? "height measure p" : "width measure q");
      select.addEventListener("change", () => this.chosen(which, select.value));
      // What another page made since is there to choose.
      select.addEventListener("pointerdown", () => void this.refresh());
    }
    const change = holdings.change;
    this.toggle.hidden = change === undefined;
    if (change === undefined) return;

    this.toggle.type = "button";
    this.toggle.textContent = "models…";
    this.toggle.title = "load models, and make measures of them";
    this.panel.popover = "auto";
    this.panel.className = "measure-panel";
    this.panel.id = "measure-panel";
    this.toggle.setAttribute("popovertarget", this.panel.id);
    this.panel.addEventListener("toggle", (event) => {
      if ((event as ToggleEvent).newState !== "open") return;
      void this.refresh();
      this.recordings?.list().then(() => this.show(), this.failed);
      const rect = this.toggle.getBoundingClientRect();
      this.panel.style.left = `${Math.max(8, Math.min(rect.left, innerWidth - this.panel.offsetWidth - 8))}px`;
      this.panel.style.top = `${rect.bottom + 4}px`;
    });

    const heading = (text: string) => {
      const element = document.createElement("h3");
      element.textContent = text;
      return element;
    };
    this.models.className = "measure-models";
    this.presets.className = "measure-presets";
    const custom = document.createElement("div");
    custom.className = "measure-row";
    for (const parameter of catalog.models) {
      const select = document.createElement("select");
      select.setAttribute("aria-label", `model ${parameter.name}`);
      select.title = parameter.name;
      select.append(...parameter.options.map((option) => new Option(option, option)));
      select.value = parameter.default;
      this.loading.set(parameter.name, select);
      custom.append(select);
    }
    custom.append(button("load", "load this model", () => {
      const model = catalog.models.map(({ name }) => encodeURIComponent(this.loading.get(name)!.value)).join("/");
      void this.change(() => change.load(model));
    }));

    const form = document.createElement("div");
    form.className = "measure-form";
    const label = (name: string, fields: HTMLElement[]) => {
      const element = document.createElement("label");
      const text = document.createElement("span");
      text.textContent = name;
      element.append(text, ...fields);
      form.append(element);
    };
    this.model.setAttribute("aria-label", "the measure's model");
    label("model", [this.model]);
    for (const parameter of catalog.parameters) label(parameter.name, this.field(parameter));
    const take = document.createElement("div");
    take.className = "measure-row";
    for (const which of ["p", "q"] as const) {
      take.append(button(`as ${which}`, `read this measure as ${which}`, () => void this.make(which)));
    }

    this.panel.append(heading("Models"), this.models, this.presets, custom, heading("New measure"), form, take);
  }

  /** Where the panel is: placed beside the toggle. */
  get element(): HTMLElement {
    return this.panel;
  }

  /** Show ``specs`` as chosen. */
  set(specs: { p: string; q: string }): void {
    this.specs = specs;
    this.show();
  }

  /** Ask what the server holds, and show it. */
  async refresh(): Promise<void> {
    try {
      this.held = await this.holdings.held();
    } catch (error) {
      this.failed(error);
      return;
    }
    this.show();
    this.onHeld(this.held);
  }

  /** Make p a measure like it that hears the recording ``file``, uploading it. */
  async hear(file: File): Promise<void> {
    const parameter = this.catalog.parameters.find(({ kind }) => kind === "recording");
    if (parameter === undefined || this.recordings === null) throw new Error("this server's measures hear no recordings");
    if (this.specs.p === "") throw new Error("load a model and make a measure of it first");
    const recording = await this.recordings.upload(file);
    const values = new Map(parse(this.catalog, this.specs.p));
    values.set(parameter.name, recording.id);
    const spec = format(this.catalog, values);
    await this.change(() => this.holdings.change!.open(spec));
    this.chosen("p", spec);
  }

  private async change(changed: () => Promise<Held>): Promise<void> {
    try {
      this.held = await changed();
    } catch (error) {
      this.failed(error);
    }
    this.show();
    this.onHeld(this.held);
  }

  private async make(which: "p" | "q"): Promise<void> {
    if (this.model.value === "") return this.failed(new Error("load a model first"));
    const values = new Map(this.catalog.models.map(({ name }, index) => [name, decodeURIComponent(this.model.value.split("/")[index]!)]));
    for (const [name, field] of this.fields) values.set(name, field.value);
    const spec = format(this.catalog, values);
    await this.change(() => this.holdings.change!.open(spec));
    this.chosen(which, spec);
  }

  private show(): void {
    const { catalog, recordings, held } = this;
    const words = (spec: string) => describe(catalog, spec, recordings);
    for (const which of ["p", "q"] as const) {
      const select = this[which];
      const spec = this.specs[which];
      const specs = held.measures.includes(spec) || spec === "" ? held.measures : [...held.measures, spec];
      const options = specs.map((measure) => new Option(words(measure), measure));
      if (options.length === 0) options.push(new Option("no measure yet", ""));
      select.replaceChildren(...options);
      select.value = spec;
    }
    if (this.holdings.change === undefined) return;
    const change = this.holdings.change;

    this.models.replaceChildren(...held.models.map(({ spec, state, message }) => {
      const row = document.createElement("div");
      row.className = `measure-model ${state}`;
      const name = document.createElement("span");
      name.textContent = words(spec);
      const said = document.createElement("span");
      said.className = "measure-state";
      said.textContent = state === "failed" ? `failed: ${message ?? ""}` : state === "loading" ? "loading…" : "loaded";
      said.title = said.textContent;
      row.append(name, said, button(state === "failed" ? "dismiss" : "unload", "let this model go", () => void this.change(() => change.unload(spec))));
      return row;
    }));
    if (held.models.length === 0) this.models.textContent = "None loaded.";
    const asked = new Set(held.models.map(({ spec }) => spec));
    this.presets.replaceChildren(...catalog.presets.filter((preset) => !asked.has(preset)).map((preset) =>
      button(`load ${words(preset)}`, "load this model, as it is offered", () => void this.change(() => change.load(preset)))));

    const loaded = held.models.filter(({ state }) => state === "loaded").map(({ spec }) => spec);
    const chosen = this.model.value || modelOf(catalog, this.specs.p);
    this.model.replaceChildren(...loaded.map((spec) => new Option(words(spec), spec)));
    if (loaded.length === 0) this.model.append(new Option("load a model first", ""));
    if (loaded.includes(chosen)) this.model.value = chosen;

    for (const select of this.hearing) {
      // Silence, each recording the server is known to hold, and a way to
      // upload another.
      const value = select.value;
      const ids = new Set(["", ...(recordings?.known.keys() ?? []), value === UPLOAD ? "" : value]);
      select.replaceChildren(...[...ids].map((id) => new Option(recordingName(id, recordings), id)), new Option("upload…", UPLOAD));
      select.value = value === UPLOAD ? "" : value;
    }
    if (this.poll === null && held.models.some(({ state }) => state === "loading")) {
      this.poll = window.setTimeout(() => {
        this.poll = null;
        void this.refresh();
      }, POLL_MS);
    }
  }

  private field(parameter: Parameter): HTMLElement[] {
    let field: HTMLInputElement | HTMLSelectElement;
    const after: HTMLElement[] = [];
    switch (parameter.kind) {
      case "choice": {
        field = document.createElement("select");
        field.append(...parameter.options.map((option) => new Option(option, option)));
        field.value = parameter.default;
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
        select.append(new Option("silence", ""), new Option("upload…", UPLOAD));
        this.hearing.push(select);
        const file = document.createElement("input");
        file.type = "file";
        file.accept = "audio/*,.m4a";
        file.hidden = true;
        file.addEventListener("change", () => {
          const chosen = file.files?.[0];
          file.value = "";
          if (chosen === undefined || this.recordings === null) return;
          this.recordings.upload(chosen).then((recording) => {
            this.show();
            select.value = recording.id;
          }, this.failed);
        });
        select.addEventListener("change", () => {
          if (select.value !== UPLOAD) return;
          select.value = "";
          file.click();
        });
        after.push(file);
        break;
      }
    }
    field.setAttribute("aria-label", `the measure's ${parameter.name}`);
    this.fields.set(parameter.name, field);
    return [field, ...after];
  }
}
