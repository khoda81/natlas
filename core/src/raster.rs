//! Exact sub-pixel vertical coverage rasterizer with linear-RGB blending.
//!
//! Tall box interiors are painted opaque (`tile` with 1px `stroke` borders).
//! Boundary and thin-tail rows compute the exact vertical coverage
//! `alpha(px) = sum_{i in row, col1_i >= px} h_i` in [0, 1] and composite
//! `stroke` in IEC 61966-2-1 linear light so thin branches never flicker or
//! darken as they cross pixel boundaries.

use std::sync::OnceLock;

use crate::client::{ClientMeasure, ClientTree};
use crate::layout::{Frame, LayoutTree, Placed};
use crate::pstore::CDF_SIZE;
use crate::qlayout::QLayout;
use crate::tree::Lens;

const ON_GRID: f64 = 1e-6;
/// Above this many children in a single pixel row, switch to 64 stratified
/// mass samples so zoomed-out cached subtrees stay O(1) per row.
const EXACT_ROW_LIMIT: usize = 64;
const STRATIFIED_SAMPLES: usize = 64;

const LIN_LUT_LEN: usize = 4096;

struct SrgbLuts {
    to_lin: [f32; 256],
    to_srgb: [u8; LIN_LUT_LEN],
}

fn luts() -> &'static SrgbLuts {
    static LUTS: OnceLock<SrgbLuts> = OnceLock::new();
    LUTS.get_or_init(|| {
        let mut to_lin = [0.0f32; 256];
        for (i, slot) in to_lin.iter_mut().enumerate() {
            let s = i as f64 / 255.0;
            *slot = (if s <= 0.04045 {
                s / 12.92
            } else {
                ((s + 0.055) / 1.055).powf(2.4)
            }) as f32;
        }
        let mut to_srgb = [0u8; LIN_LUT_LEN];
        let denom = (LIN_LUT_LEN - 1) as f64;
        for (i, slot) in to_srgb.iter_mut().enumerate() {
            let l = i as f64 / denom;
            let s = if l <= 0.0031308 {
                12.92 * l
            } else {
                1.055 * l.powf(1.0 / 2.4) - 0.055
            };
            *slot = (s * 255.0).round().clamp(0.0, 255.0) as u8;
        }
        SrgbLuts { to_lin, to_srgb }
    })
}

#[inline]
fn unpack_rgb(rgb: u32) -> [u8; 3] {
    [
        ((rgb >> 16) & 255) as u8,
        ((rgb >> 8) & 255) as u8,
        (rgb & 255) as u8,
    ]
}

#[inline]
fn rgb_to_lin(rgb: [u8; 3], lut: &SrgbLuts) -> [f32; 3] {
    [
        lut.to_lin[rgb[0] as usize],
        lut.to_lin[rgb[1] as usize],
        lut.to_lin[rgb[2] as usize],
    ]
}

#[inline]
fn lin_to_srgb(v: f32, lut: &SrgbLuts) -> u8 {
    let idx = (v.clamp(0.0, 1.0) * (LIN_LUT_LEN - 1) as f32 + 0.5) as usize;
    lut.to_srgb[idx.min(LIN_LUT_LEN - 1)]
}

#[inline]
fn blend_linear(bg: [u8; 3], fg_lin: [f32; 3], alpha: f32, lut: &SrgbLuts) -> [u8; 4] {
    let a = alpha.clamp(0.0, 1.0);
    let bg_lin = rgb_to_lin(bg, lut);
    [
        lin_to_srgb(bg_lin[0] + a * (fg_lin[0] - bg_lin[0]), lut),
        lin_to_srgb(bg_lin[1] + a * (fg_lin[1] - bg_lin[1]), lut),
        lin_to_srgb(bg_lin[2] + a * (fg_lin[2] - bg_lin[2]), lut),
        255,
    ]
}

#[inline]
fn col_of(x: f64, cam_x: f64, px_per_nat: f64) -> i32 {
    ((x - cam_x) * px_per_nat + ON_GRID).floor() as i32
}

/// Rasterize `frame`'s solid box fills and exact linear-RGB sub-pixel
/// anti-aliased borders & thin tails into `out` (RGBA8, `width_px * height_px * 4`).
#[allow(clippy::too_many_arguments)]
pub fn rasterize(
    tree: &ClientTree,
    lens: Lens,
    frame: &Frame,
    cam_x: f64,
    cam_k: f64,
    width_px: usize,
    height_px: usize,
    dpr: f64,
    surface_rgb: u32,
    tile_rgb: u32,
    stroke_rgb: u32,
    out: &mut [u8],
) {
    if width_px == 0 || height_px == 0 || out.len() < width_px * height_px * 4 {
        return;
    }
    let lut = luts();
    let surface = unpack_rgb(surface_rgb);
    let tile = unpack_rgb(tile_rgb);
    let stroke = unpack_rgb(stroke_rgb);
    let surface_rgba = [surface[0], surface[1], surface[2], 255];
    let tile_rgba = [tile[0], tile[1], tile[2], 255];
    let stroke_rgba = [stroke[0], stroke[1], stroke[2], 255];
    let stroke_lin = rgb_to_lin(stroke, lut);

    // Precompute 256-level linear-RGB LUT for stroke blended over surface.
    let mut stroke_over_surface = [[0u8; 4]; 256];
    for (i, slot) in stroke_over_surface.iter_mut().enumerate() {
        *slot = blend_linear(surface, stroke_lin, i as f32 / 255.0, lut);
    }

    // 1. Fill background with surface.
    let stride = width_px * 4;
    for chunk in out[..height_px * stride].as_chunks_mut::<4>().0 {
        chunk.copy_from_slice(&surface_rgba);
    }

    let px_per_nat = cam_k * dpr;
    let height_f = height_px as f64;
    let width_i = width_px as i32;

    // 2. Paint solid tall-box interiors (and fallback strokes for nodes without a fixed QLayout).
    for span in &frame.spans {
        let y0 = (span.y0.clamp(-1.0, 2.0) * height_f).round() as i32;
        let y1 = (span.y1.clamp(-1.0, 2.0) * height_f).round() as i32;
        let r0 = y0.max(0) as usize;
        let r1 = y1.min(height_px as i32).max(0) as usize;
        if r1 <= r0 {
            continue;
        }
        let left = col_of(span.x0, cam_x, px_per_nat);
        let reach = col_of(span.reach, cam_x, px_per_nat) + 1;
        let has_fill = span.x0 < span.fill;
        let fill = if has_fill {
            col_of(span.fill, cam_x, px_per_nat)
        } else {
            left
        };

        let is_quantile = has_quantile(tree, lens, frame, span.node);
        for py in r0..r1 {
            let row_off = py * stride;
            if has_fill {
                // Crisp 1px left border.
                if 0 <= left && left < width_i {
                    let o = row_off + left as usize * 4;
                    out[o..o + 4].copy_from_slice(&stroke_rgba);
                }
                // Solid interior fill.
                let f0 = (left + 1).max(0).min(width_i) as usize;
                let f1 = fill.max(0).min(width_i) as usize;
                for px in f0..f1 {
                    let o = row_off + px * 4;
                    out[o..o + 4].copy_from_slice(&tile_rgba);
                }
                // Crisp 1px right border of tall box.
                let s0 = fill.max(0).min(width_i) as usize;
                let s1 = reach.max(0).min(width_i) as usize;
                for px in s0..s1 {
                    let o = row_off + px * 4;
                    out[o..o + 4].copy_from_slice(&stroke_rgba);
                }
            } else if !is_quantile {
                // Ultra-deep fallback (beyond 2^44 QLayout range): crisp span stroke.
                let s0 = left.max(0).min(width_i) as usize;
                let s1 = reach.max(0).min(width_i) as usize;
                for px in s0..s1 {
                    let o = row_off + px * 4;
                    out[o..o + 4].copy_from_slice(&stroke_rgba);
                }
            }
        }
    }

    // 3. Exact linear-RGB sub-pixel coverage pass for all boundary & thin-tail rows.
    let mut cov = vec![0.0f32; width_px];
    let row_h = 1.0 / height_f;

    for (&node, placed) in &frame.placed {
        let Some(d) = tree.client_measure(node, lens.p) else {
            continue;
        };
        let Some(fixed) = d.fixed() else {
            continue;
        };
        let Some(geometry) = placed.quantile() else {
            continue;
        };
        let col0 = col_of(placed.x, cam_x, px_per_nat);
        if col0 >= width_i {
            continue;
        }
        let col0_u = col0.max(0) as usize;

        let q = tree.client_measure(node, lens.q);
        let key = (lens.q, q.map_or(0, ClientMeasure::revision));
        let mut cached = fixed.widths.borrow_mut();
        if cached.as_ref().is_none_or(|(prior, _)| *prior != key) {
            *cached = Some((key, d.fixed_widths(fixed, q, lens.p == lens.q)));
        }
        let (_, widths) = cached.as_ref().expect("width index");

        let row_geo = geometry.in_rows(row_h);
        let p = &fixed.p;
        let len = p.ends().len();
        if len == 0 {
            continue;
        }
        let top_row = (row_geo.y(0) + ON_GRID).floor() as i64;
        let end_row = ((row_geo.y(CDF_SIZE) + ON_GRID).floor() as i64 + 1).min(height_px as i64);
        let start_py = top_row.max(0);
        if end_row <= start_py {
            continue;
        }
        let mut cursor = if row_geo.y(0) >= 0.0 {
            0
        } else {
            row_geo.position(p, start_py as f64)
        };
        let limit = if row_geo.y(CDF_SIZE) <= height_f {
            len
        } else {
            row_geo.tops_before(p, height_f, cursor)
        };

        let mut py = start_py;
        while py < end_row {
            let py_f = py as f64;
            let first = row_geo.bottom_at(p, py_f + ON_GRID, cursor);
            let end = row_geo
                .tops_before(p, py_f + 1.0 - ON_GRID, first)
                .min(limit);
            if first < end {
                cursor = end - 1;
                // Skip pure interior of a single tall child (already painted opaque in step 2).
                if first + 1 == end {
                    let top = row_geo.y(QLayout::start(p, first));
                    let bot = row_geo.end_y(p.ends()[first]);
                    if top <= py_f + ON_GRID && py_f + 1.0 - ON_GRID <= bot {
                        let skip_to = ((bot - 2.0 * ON_GRID).floor() as i64).min(end_row);
                        py = (py + 1).max(skip_to);
                        continue;
                    }
                }

                let mut max_col: Option<usize> = None;
                let count = end - first;
                if count <= EXACT_ROW_LIMIT {
                    for pos in first..end {
                        let w = widths.at(pos);
                        if w <= 0.0 {
                            continue;
                        }
                        let top = row_geo.y(QLayout::start(p, pos));
                        let bot = row_geo.end_y(p.ends()[pos]);
                        let h = (bot.min(py_f + 1.0) - top.max(py_f)).clamp(0.0, 1.0) as f32;
                        if h <= 0.0 {
                            continue;
                        }
                        let col1 = col_of(placed.x + w, cam_x, px_per_nat);
                        if col1 < 0 {
                            continue;
                        }
                        let c = (col1.min(width_i - 1) as usize).max(col0_u);
                        cov[c] += h;
                        max_col = Some(max_col.map_or(c, |m| m.max(c)));
                    }
                } else {
                    let weight = 1.0f32 / STRATIFIED_SAMPLES as f32;
                    let mut sc = first;
                    for s in 0..STRATIFIED_SAMPLES {
                        let y_s = py_f + (s as f64 + 0.5) / STRATIFIED_SAMPLES as f64;
                        sc = row_geo.bottom_at(p, y_s, sc);
                        if sc >= end {
                            break;
                        }
                        let w = widths.at(sc);
                        if w <= 0.0 {
                            continue;
                        }
                        let col1 = col_of(placed.x + w, cam_x, px_per_nat);
                        if col1 < 0 {
                            continue;
                        }
                        let c = (col1.min(width_i - 1) as usize).max(col0_u);
                        cov[c] += weight;
                        max_col = Some(max_col.map_or(c, |m| m.max(c)));
                    }
                }

                if let Some(right_col) = max_col {
                    let row_off = py as usize * stride;
                    let mut acc = 0.0f32;
                    for px in (col0_u..=right_col).rev() {
                        acc = (acc + cov[px]).min(1.0);
                        cov[px] = 0.0;
                        if acc <= 0.0 {
                            continue;
                        }
                        let o = row_off + px * 4;
                        let bg = [out[o], out[o + 1], out[o + 2]];
                        let rgba = if bg == surface {
                            stroke_over_surface[(acc * 255.0 + 0.5) as usize]
                        } else {
                            blend_linear(bg, stroke_lin, acc, lut)
                        };
                        out[o..o + 4].copy_from_slice(&rgba);
                    }
                }
            }
            py += 1;
        }
    }
}

fn has_quantile(tree: &ClientTree, lens: Lens, frame: &Frame, node: u64) -> bool {
    let Some(placed): Option<&Placed> = frame.placed.get(&node) else {
        return false;
    };
    placed.quantile().is_some()
        && tree
            .client_measure(node, lens.p)
            .and_then(ClientMeasure::fixed)
            .is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn srgb_linear_round_trip_is_exact_for_all_256_values() {
        let lut = luts();
        for v in 0u8..=255 {
            let lin = lut.to_lin[v as usize];
            assert_eq!(lin_to_srgb(lin, lut), v, "mismatch at sRGB {v}");
        }
    }

    #[test]
    fn linear_blend_half_coverage_is_gamma_correct() {
        let lut = luts();
        let white_lin = rgb_to_lin([255, 255, 255], lut);
        let mid = blend_linear([0, 0, 0], white_lin, 0.5, lut);
        // 0.5^(1/2.4) * 1.055 - 0.055 ≈ 0.7354 -> 188 (not sRGB 128).
        assert!((186..=189).contains(&mid[0]), "got {mid:?}");
        assert_eq!(mid[0], mid[1]);
        assert_eq!(mid[1], mid[2]);
        assert_eq!(mid[3], 255);
    }
}
