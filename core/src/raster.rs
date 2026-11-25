//! Boundary coverage as horizontal rectangles for Canvas2D.
//!
//! Fractional rows use 16 coverage levels blended in linear light. Dense
//! unordered widths use a maximum per block to approximate coverage.
//! Grid-aligned and exact-layout edges retain the layout span's outline.

use std::sync::OnceLock;

use crate::client::{ClientMeasure, ClientTree};
use crate::layout::{Frame, LayoutTree, Span};
use crate::qlayout::QLayout;
use crate::tree::Lens;

const ON_GRID: f64 = 1e-6;
const EXACT_ROW_LIMIT: usize = 64;
pub const AA_BUCKETS: usize = 16;

struct SrgbLuts {
    to_lin: [f32; 256],
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
        SrgbLuts { to_lin }
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
fn lin_to_srgb_byte(l: f32) -> u8 {
    let l = l.clamp(0.0, 1.0) as f64;
    let s = if l <= 0.0031308 {
        12.92 * l
    } else {
        1.055 * l.powf(1.0 / 2.4) - 0.055
    };
    (s * 255.0).round().clamp(0.0, 255.0) as u8
}

/// Precompute `AA_BUCKETS` packed `0xRRGGBB` colors for `stroke` blended over
/// `surface` at coverage `1/N .. N/N` in IEC 61966-2-1 linear light.
pub fn linear_palette(surface_rgb: u32, stroke_rgb: u32) -> [u32; AA_BUCKETS] {
    let lut = luts();
    let bg = unpack_rgb(surface_rgb);
    let fg = unpack_rgb(stroke_rgb);
    let bg_lin = [
        lut.to_lin[bg[0] as usize],
        lut.to_lin[bg[1] as usize],
        lut.to_lin[bg[2] as usize],
    ];
    let fg_lin = [
        lut.to_lin[fg[0] as usize],
        lut.to_lin[fg[1] as usize],
        lut.to_lin[fg[2] as usize],
    ];
    let mut out = [0u32; AA_BUCKETS];
    for (i, slot) in out.iter_mut().enumerate() {
        let a = (i + 1) as f32 / AA_BUCKETS as f32;
        let r = lin_to_srgb_byte(bg_lin[0] + a * (fg_lin[0] - bg_lin[0])) as u32;
        let g = lin_to_srgb_byte(bg_lin[1] + a * (fg_lin[1] - bg_lin[1])) as u32;
        let b = lin_to_srgb_byte(bg_lin[2] + a * (fg_lin[2] - bg_lin[2])) as u32;
        *slot = (r << 16) | (g << 8) | b;
    }
    out
}

#[inline]
fn col_of(x: f64, cam_x: f64, px_per_nat: f64) -> i32 {
    ((x - cam_x) * px_per_nat + ON_GRID).floor() as i32
}

/// Emit merged AA span quads `[bucket, px0, py, width]` (4 `i32`s per rect)
/// for all boundary and thin-tail rows of `frame`.
#[allow(clippy::too_many_arguments)]
pub fn aa_spans(
    tree: &ClientTree,
    lens: Lens,
    frame: &Frame,
    cam_x: f64,
    cam_k: f64,
    width_px: usize,
    height_px: usize,
    dpr: f64,
    out: &mut Vec<i32>,
) {
    out.clear();
    if width_px == 0 || height_px == 0 {
        return;
    }
    let px_per_nat = cam_k * dpr;
    let inv_px_per_nat = px_per_nat.recip();
    let height = height_px as f64;
    let width = width_px as i32;
    let columns_of = |span: &Span| {
        let left = col_of(span.x0, cam_x, px_per_nat).max(0) as usize;
        let right = col_of(span.reach, cam_x, px_per_nat).saturating_add(1);
        left..right.clamp(0, width) as usize
    };
    let rows_of = |span: &Span| {
        let row = |y: f64| (y * height).round().clamp(0.0, height) as usize;
        row(span.y0)..row(span.y1)
    };
    let emit = |out: &mut Vec<i32>, bucket: u8, x: usize, y: usize, len: usize| {
        out.extend_from_slice(&[i32::from(bucket - 1), x as i32, y as i32, len as i32]);
    };
    let full = AA_BUCKETS as u8;
    let bucket = |coverage: f32| {
        if 0.0 < coverage {
            ((coverage * f32::from(full)).round() as u8).clamp(1, full)
        } else {
            0
        }
    };
    let mut coverage = vec![0.0f32; width_px];
    let mut row_buckets = vec![0u8; width_px];

    // Layout already identifies which rows need an outline across the span.
    for spans in frame.spans.chunk_by(|left, right| left.node == right.node) {
        let mut boundaries = spans.iter().filter(|span| span.x0 == span.fill).peekable();
        let Some(first_boundary) = boundaries.peek() else {
            continue;
        };
        let node = first_boundary.node;
        let placed = &frame.placed[&node];
        let fixed_geometry = placed.quantile().and_then(|geometry| {
            let d = tree.client_measure(node, lens.p)?;
            Some((d, d.fixed()?, geometry))
        });
        let Some((d, fixed, geometry)) = fixed_geometry else {
            for span in boundaries {
                let columns = columns_of(span);
                if !columns.is_empty() {
                    for y in rows_of(span) {
                        emit(out, full, columns.start, y, columns.len());
                    }
                }
            }
            continue;
        };

        let q = tree.client_measure(node, lens.q);
        let key = (lens.q, q.map_or(0, ClientMeasure::revision));
        let mut cached = fixed.widths.borrow_mut();
        if cached.as_ref().is_none_or(|(prior, _)| *prior != key) {
            *cached = Some((key, d.fixed_widths(fixed, q, lens.p == lens.q)));
        }
        let (_, widths) = cached.as_ref().expect("width index");
        let rows = geometry.in_rows(1.0 / height);
        let p = &fixed.p;
        let increasing = widths.monotonic() == Some(true);
        let mut cursor = 0;
        for span in boundaries {
            let columns = columns_of(span);
            if columns.is_empty() {
                continue;
            }
            for y in rows_of(span) {
                let row = y as f64;
                let first = rows.bottom_at(p, row + ON_GRID, cursor);
                let end = rows.tops_before(p, row + 1.0 - ON_GRID, first);
                // Layout's reach includes both neighbors at a grid line.
                if end <= first || (rows.y(QLayout::start(p, first)) - row).abs() <= ON_GRID {
                    emit(out, full, columns.start, y, columns.len());
                    continue;
                }
                cursor = end - 1;
                let count = end - first;
                if EXACT_ROW_LIMIT < count && increasing {
                    let row_top = rows.y(QLayout::start(p, first)).max(row);
                    let row_bottom = rows.end_y(p.ends()[end - 1]).min(row + 1.0);
                    let ordered = widths.slice(first, end);
                    for x in columns.clone() {
                        let target = cam_x + (x as f64 - ON_GRID) * inv_px_per_nat - placed.x;
                        let offset = ordered.partition_point(|&w| w <= 0.0 || w < target);
                        row_buckets[x] = if offset < ordered.len() {
                            let top = rows.y(QLayout::start(p, first + offset)).max(row_top);
                            bucket((row_bottom - top).clamp(0.0, 1.0) as f32)
                        } else {
                            0
                        };
                    }
                } else {
                    let step = if count <= EXACT_ROW_LIMIT {
                        1
                    } else {
                        count.div_ceil(32)
                    };
                    for start in (first..end).step_by(step) {
                        let stop = (start + step).min(end);
                        let w = widths.maximum(start, stop);
                        if w <= 0.0 {
                            continue;
                        }
                        let right = col_of(placed.x + w, cam_x, px_per_nat);
                        if right < columns.start as i32 {
                            continue;
                        }
                        let x = (right as usize).min(columns.end - 1);
                        let top = rows.y(QLayout::start(p, start));
                        let bottom = rows.end_y(p.ends()[stop - 1]);
                        coverage[x] +=
                            (bottom.min(row + 1.0) - top.max(row)).clamp(0.0, 1.0) as f32;
                    }
                    let mut mass = 0.0;
                    for x in columns.clone().rev() {
                        mass = (mass + coverage[x]).min(1.0);
                        coverage[x] = 0.0;
                        row_buckets[x] = bucket(mass);
                    }
                }

                let mut x = columns.start;
                for run in row_buckets[columns.clone()].chunk_by(|left, right| left == right) {
                    if run[0] != 0 {
                        emit(out, run[0], x, y, run.len());
                    }
                    x += run.len();
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn srgb_linear_round_trip_is_exact_for_all_256_values() {
        let lut = luts();
        for v in 0u8..=255 {
            assert_eq!(
                lin_to_srgb_byte(lut.to_lin[v as usize]),
                v,
                "mismatch at sRGB {v}"
            );
        }
    }

    #[test]
    fn linear_blend_half_coverage_is_gamma_correct() {
        let pal = linear_palette(0x000000, 0xffffff);
        let [r, g, b] = unpack_rgb(pal[AA_BUCKETS / 2 - 1]);
        assert!((186..=189).contains(&r), "got {r}, {g}, {b}");
        assert_eq!(r, g);
        assert_eq!(g, b);
        assert_eq!(pal[AA_BUCKETS - 1], 0xffffff);
    }
}
