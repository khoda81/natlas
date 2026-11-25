// Console-backed, auto-dismissing toast banners that pause on hover.

export type ToastLevel = "error" | "warn" | "info";

interface ActiveToast {
  element: HTMLElement;
  badge: HTMLElement;
  count: number;
  remaining: number;
  startedAt: number | null;
  timer: number | null;
}

const DEFAULT_MS: Record<ToastLevel, number> = {
  error: 6000,
  warn: 4500,
  info: 3500,
};

const MAX_TOASTS = 5;

export class Toasts {
  private readonly active = new Map<string, ActiveToast>();

  constructor(private readonly container: HTMLElement) {}

  error(error: unknown, context?: string): void {
    if (context) console.error(`[natlas] ${context}:`, error);
    else console.error("[natlas]", error);
    this.show("error", formatError(error));
  }

  warn(message: string, detail?: unknown): void {
    if (detail !== undefined) console.warn(`[natlas] ${message}`, detail);
    else console.warn(`[natlas] ${message}`);
    this.show("warn", message);
  }

  info(message: string): void {
    console.info(`[natlas] ${message}`);
    this.show("info", message);
  }

  private show(level: ToastLevel, message: string, duration = DEFAULT_MS[level]): void {
    const key = `${level}:${message}`;
    const existing = this.active.get(key);
    if (existing !== undefined) {
      existing.count++;
      existing.badge.textContent = `×${existing.count}`;
      existing.badge.hidden = false;
      existing.remaining = duration;
      if (existing.startedAt !== null) this.arm(key, existing);
      return;
    }

    while (this.active.size >= MAX_TOASTS) {
      const oldest = this.active.keys().next().value;
      if (oldest === undefined) break;
      this.dismiss(oldest);
    }

    const toast = document.createElement("div");
    toast.className = `toast ${level}`;
    toast.setAttribute("role", level === "error" ? "alert" : "status");

    const body = document.createElement("span");
    body.className = "toast-message";
    body.textContent = message;

    const badge = document.createElement("span");
    badge.className = "toast-count";
    badge.hidden = true;

    const close = document.createElement("button");
    close.type = "button";
    close.className = "toast-close";
    close.title = "Dismiss";
    close.setAttribute("aria-label", "Dismiss");
    close.textContent = "×";
    close.addEventListener("click", () => this.dismiss(key));

    toast.append(body, badge, close);
    this.container.append(toast);

    const entry: ActiveToast = {
      element: toast,
      badge,
      count: 1,
      remaining: duration,
      startedAt: null,
      timer: null,
    };
    this.active.set(key, entry);

    toast.addEventListener("pointerenter", () => this.pause(entry));
    toast.addEventListener("pointerleave", () => this.arm(key, entry));
    this.arm(key, entry);
  }

  private arm(key: string, entry: ActiveToast): void {
    if (entry.timer !== null) clearTimeout(entry.timer);
    entry.startedAt = performance.now();
    entry.timer = window.setTimeout(() => this.dismiss(key), Math.max(0, entry.remaining));
  }

  private pause(entry: ActiveToast): void {
    if (entry.timer !== null) {
      clearTimeout(entry.timer);
      entry.timer = null;
    }
    if (entry.startedAt !== null) {
      entry.remaining = Math.max(0, entry.remaining - (performance.now() - entry.startedAt));
      entry.startedAt = null;
    }
  }

  private dismiss(key: string): void {
    const entry = this.active.get(key);
    if (entry === undefined) return;
    if (entry.timer !== null) clearTimeout(entry.timer);
    this.active.delete(key);
    entry.element.remove();
  }
}

function formatError(error: unknown): string {
  if (error instanceof Error) return error.message || error.name;
  return String(error);
}
