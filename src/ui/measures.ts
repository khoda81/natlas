// Choosing measures. p and q are each one of the measures the server holds.
// Beside them, a panel lists the models it was asked to load, loading,
// loaded or failed, and loads others from the server's parameter schema.
// Quick-load presets live purely in client localStorage.

import { atDefault, type Catalog, defaultValue, format, formatModel, modelOf, type Parameter, parse } from "../data/catalog";
import type { Held, Holdings, Recording, Recordings } from "../data/source";

const UPLOAD = "\u0000upload";
/** How often to ask what the server holds while a model loads. */
const POLL_MS = 1000;

/** ``spec`` in words: each non-empty model parameter (paths shortened to basename),
 * then each measure parameter not at its default. */
export function describe(catalog: Catalog, spec: string, recordings: Recordings | null): string {
  const values = parse(catalog, spec);
  const parts: string[] = [];
  for (const parameter of catalog.models) {
    const value = values.get(parameter.name) ?? "";
    if (parts.length > 0 && atDefault(parameter, value)) continue;
    if (value === "") continue;
    const shown = parameter.kind === "text" && value.includes("/") ? value.slice(value.lastIndexOf("/") + 1) : value;
    parts.push(parts.length === 0 ? shown : `${parameter.name} ${shown}`);
  }
  for (const parameter of catalog.parameters) {
    const value = values.get(parameter.name) ?? "";
    if (atDefault(parameter, value)) continue;
    const shown = parameter.kind === "recording"
      ? recordingName(value, recordings)
      : parameter.kind === "text" && value.includes("/")
        ? value.slice(value.lastIndexOf("/") + 1)
        : value;
    parts.push(`${parameter.name} ${shown}`);
  }
  return parts.join(" · ") || spec;
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

type FieldControl = HTMLInputElement | HTMLSelectElement;

function readControl(parameter: Parameter, control: FieldControl): string {
  if (parameter.kind === "boolean") return (control as HTMLInputElement).checked ? "true" : "false";
  return control.value;
}

function presetKey(datasetId: string): string {
  return `natlas.presets:1:${encodeURIComponent(datasetId)}`;
}

function loadPresets(datasetId: string): string[] {
  try {
    const raw = localStorage.getItem(presetKey(datasetId));
    if (raw === null) return [];
    const parsed: unknown = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed.filter((item): item is string => typeof item === "string" && item !== "") : [];
  } catch {
    return [];
  }
}

function savePresets(datasetId: string, presets: string[]): void {
  try {
    localStorage.setItem(presetKey(datasetId), JSON.stringify(presets));
  } catch {
    // Remembering presets is only a convenience.
  }
}

export class Measures {
  /** p's and q's choices. */
  readonly p = document.createElement("select");
  readonly q = document.createElement("select");
  /** Opens the panel; hidden where the server's models cannot change. */
  readonly toggle = document.createElement("button");
  private readonly panel = document.createElement("div");
  private readonly models = document.createElement("div");
  private readonly presetsEl = document.createElement("div");
  /** A model's load-time parameter controls. */
  private readonly loading = new Map<string, FieldControl>();
  /** The measure being made: its model, and its own parameters. */
  private readonly model = document.createElement("select");
  private readonly fields = new Map<string, FieldControl>();
  private readonly hearing: HTMLSelectElement[] = [];
  private specs = { p: "", q: "" };
  private poll: number | null = null;
  private presets: string[];
  /** What the server held when last asked. */
  held: Held = { models: [], measures: [] };
  /** Told each time the server says what it holds. */
  onHeld: (held: Held) => void = () => {};

  constructor(
    private readonly datasetId: string,
    private readonly catalog: Catalog,
    private readonly holdings: Holdings,
    private readonly recordings: Recordings | null,
    /** Told the measure chosen as p or q. */
    private readonly chosen: (which: "p" | "q", spec: string) => void,
    private readonly failed: (error: unknown) => void,
  ) {
    this.presets = loadPresets(datasetId);
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
    this.presetsEl.className = "measure-presets";

    const addForm = (parameters: Parameter[], target: Map<string, FieldControl>, role: string) => {
      const form = document.createElement("div");
      form.className = "measure-form";
      for (const parameter of parameters) {
        const row = document.createElement("label");
        const text = document.createElement("span");
        text.textContent = parameter.name;
        row.append(text, ...this.makeField(parameter, target, role));
        form.append(row);
      }
      return form;
    };

    const modelForm = addForm(catalog.models, this.loading, "model");
    const modelActions = document.createElement("div");
    modelActions.className = "measure-row";
    const readModelSpec = () => {
      const values = new Map<string, string>();
      for (const parameter of catalog.models) {
        values.set(parameter.name, readControl(parameter, this.loading.get(parameter.name)!));
      }
      return formatModel(catalog, values);
    };
    modelActions.append(
      button("load", "load this model", () => void this.change(() => change.load(readModelSpec()))),
      button("save preset", "save this model config as a quick-load button in this browser", () => {
        this.addPreset(readModelSpec());
      }),
    );

    const measureForm = document.createElement("div");
    measureForm.className = "measure-form";
    const modelRow = document.createElement("label");
    const modelLabel = document.createElement("span");
    modelLabel.textContent = "model";
    this.model.setAttribute("aria-label", "the measure's model");
    modelRow.append(modelLabel, this.model);
    measureForm.append(modelRow);
    for (const parameter of catalog.parameters) {
      const row = document.createElement("label");
      const text = document.createElement("span");
      text.textContent = parameter.name;
      row.append(text, ...this.makeField(parameter, this.fields, "measure"));
      measureForm.append(row);
    }
    const take = document.createElement("div");
    take.className = "measure-row";
    for (const which of ["p", "q"] as const) {
      take.append(button(`as ${which}`, `read this measure as ${which}`, () => void this.make(which)));
    }

    this.panel.append(
      heading("Models"),
      this.models,
      this.presetsEl,
      heading("Load model"),
      modelForm,
      modelActions,
      heading("New measure"),
      measureForm,
      take,
    );
  }

  /** Where the panel is: placed beside the toggle. */
  get element(): HTMLElement {
    return this.panel;
  }

  dispose(): void {
    if (this.poll !== null) {
      clearTimeout(this.poll);
      this.poll = null;
    }
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

  private addPreset(spec: string): void {
    if (!spec || this.presets.includes(spec)) return;
    this.presets = [...this.presets, spec];
    savePresets(this.datasetId, this.presets);
    this.show();
  }

  private removePreset(spec: string): void {
    this.presets = this.presets.filter((item) => item !== spec);
    savePresets(this.datasetId, this.presets);
    this.show();
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
    const values = new Map(parse(this.catalog, this.model.value));
    for (const parameter of this.catalog.parameters) {
      values.set(parameter.name, readControl(parameter, this.fields.get(parameter.name)!));
    }
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
      name.title = spec;
      const said = document.createElement("span");
      said.className = "measure-state";
      said.textContent = state === "failed" ? `failed: ${message ?? ""}` : state === "loading" ? "loading…" : "loaded";
      said.title = said.textContent;
      const actions = document.createElement("span");
      actions.className = "measure-model-actions";
      if (state === "loaded" && !this.presets.includes(spec)) {
        actions.append(button("pin", "save as quick-load preset", () => this.addPreset(spec)));
      }
      actions.append(button(state === "failed" ? "dismiss" : "unload", "let this model go", () => void this.change(() => change.unload(spec))));
      row.append(name, said, actions);
      return row;
    }));
    if (held.models.length === 0) this.models.textContent = "None loaded.";

    const asked = new Set(held.models.map(({ spec }) => spec));
    const chips = this.presets.filter((preset) => !asked.has(preset)).map((preset) => {
      const wrap = document.createElement("span");
      wrap.className = "preset-chip";
      wrap.append(
        button(`load ${words(preset)}`, preset, () => void this.change(() => change.load(preset))),
        button("×", "remove saved preset", () => this.removePreset(preset)),
      );
      return wrap;
    });
    this.presetsEl.replaceChildren(...chips);
    this.presetsEl.hidden = chips.length === 0;

    const loaded = held.models.filter(({ state }) => state === "loaded").map(({ spec }) => spec);
    const chosen = this.model.value || modelOf(catalog, this.specs.p);
    this.model.replaceChildren(...loaded.map((spec) => new Option(words(spec), spec)));
    if (loaded.length === 0) this.model.append(new Option("load a model first", ""));
    if (loaded.includes(chosen)) this.model.value = chosen;

    for (const select of this.hearing) {
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

  private makeField(parameter: Parameter, target: Map<string, FieldControl>, role: string): HTMLElement[] {
    let field: FieldControl;
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
        field.step = parameter.step !== undefined ? String(parameter.step) : parameter.integer ? "1" : "any";
        if (parameter.min !== undefined) field.min = String(parameter.min);
        if (parameter.max !== undefined) field.max = String(parameter.max);
        field.placeholder = parameter.default === null ? "none" : String(parameter.default);
        break;
      }
      case "text": {
        field = document.createElement("input");
        field.type = "text";
        field.value = defaultValue(parameter);
        field.placeholder = parameter.placeholder ?? "none";
        if (parameter.suggestions && parameter.suggestions.length > 0) {
          const list = document.createElement("datalist");
          list.id = `dl-${role}-${parameter.name}-${Math.random().toString(36).slice(2, 8)}`;
          list.append(...parameter.suggestions.map((item) => new Option(item, item)));
          field.setAttribute("list", list.id);
          after.push(list);
        }
        break;
      }
      case "boolean": {
        field = document.createElement("input");
        field.type = "checkbox";
        field.checked = parameter.default;
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
    field.setAttribute("aria-label", `${role} ${parameter.name}`);
    target.set(parameter.name, field);
    return [field, ...after];
  }
}
