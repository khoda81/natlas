// Anchored integer-distribution geometry and best-effort picking, implemented
// by the shared Rust core. These adapters keep camera mutation at the UI edge.

import {
  clampPosition,
  layout as layoutFrame,
  normalizeBoundary,
  pick as pickBox,
  place as placeNode,
  reanchor as reanchorCamera,
  repairBoundary,
  rootExtent as measureRootExtent,
} from "./wasm";
import type { IntervalTree, Label, Lens, NodeId } from "./tree";

export type { Lens } from "./tree";

export interface Boundary {
  node: NodeId;
  slot: number;
}

export interface Camera {
  x: number;
  k: number;
  anchor: Boundary;
  a: number;
}

export interface Viewport {
  width: number;
  height: number;
}

export type BoxKind = "node" | "leaf" | "tail" | "merged";

export interface Box {
  node: NodeId;
  slot: number;
  count: number;
  kind: BoxKind;
  label: Label | null;
  p: number;
  q: number;
  lnMass: number;
  x0: number;
  x1: number;
  y0: number;
  y1: number;
}

export interface Frame {
  boxes: Box[];
  known: Box[];
  walked: { node: NodeId; lnMass: number; x: number }[];
  unexpanded: { node: NodeId; label: Label; lnMass: number; x1: number }[];
  boundaries: { boundary: Boundary; y: number }[];
  rootTop: number;
  rootBottom: number;
  /** Deepest known node covering the screen, and its top in view heights. */
  cover: { node: NodeId; y: number } | null;
}

export interface Selection {
  box: Box;
  /** False in null space: the branch is known, its exact continuation is not. */
  exact: boolean;
}

export interface LayoutOptions {
  mergePx: number;
}

export function logView(cam: Camera): number {
  return Math.min(0, -cam.x);
}

export function layout(
  tree: IntervalTree,
  lens: Lens,
  cam: Camera,
  view: Viewport,
  options: Partial<LayoutOptions> = {},
): Frame {
  return layoutFrame(tree.raw, lens.p, lens.q, cam, view, options.mergePx ?? 1) as Frame;
}

export function normalize(tree: IntervalTree, lens: Lens, boundary: Boundary): Boundary {
  return normalizeBoundary(tree.raw, lens.p, lens.q, boundary) as Boundary;
}

export function repair(tree: IntervalTree, lens: Lens, cam: Camera): void {
  cam.anchor = repairBoundary(tree.raw, lens.p, lens.q, cam) as Boundary;
}

export function place(tree: IntervalTree, lens: Lens, id: NodeId): { lnMass: number; x1: number } {
  return placeNode(tree.raw, lens.p, lens.q, id) as { lnMass: number; x1: number };
}

export function rootExtent(tree: IntervalTree, lens: Lens, cam: Camera): { top: number; bottom: number } {
  return measureRootExtent(tree.raw, lens.p, lens.q, cam) as { top: number; bottom: number };
}

export function reanchor(cam: Camera, frame: Frame): void {
  const position = reanchorCamera(cam, frame) as { anchor: Boundary; a: number } | null;
  if (position !== null) {
    cam.anchor = position.anchor;
    cam.a = position.a;
  }
}

export function clamp(tree: IntervalTree, lens: Lens, cam: Camera): void {
  cam.a = clampPosition(tree.raw, lens.p, lens.q, cam);
}

export function pick(frame: Frame, cam: Camera, view: Viewport, x: number, y: number): Selection | null {
  return pickBox(frame, cam, view, x, y) as Selection | null;
}
