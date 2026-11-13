//! Points inside a node, exact at any depth: fractions of the node's p mass
//! from its top, written in base 256.
//!
//! An edge is known as ln of the mass below it, so its own fraction,
//! 1 − e^ln, is whatever [`Dyadic::exp`] rounds it to. Every comparison,
//! distance and change of node here goes through that one rounding, so a
//! point keeps its side of an edge and its distance from it however deep
//! the view is, and going up a node and down again returns the same point.

use std::cmp::Ordering;
use std::f64::consts::LN_2;

use num_bigint::BigUint;

/// ln 2 in two parts, the first with enough trailing zeros that a multiple
/// of it is exact: fdlibm's argument reduction.
const LN2_HIGH: f64 = 0.693_147_180_369_123_8;
const LN2_LOW: f64 = 1.908_214_929_270_587_7e-10;

/// n / 2^bits.
#[derive(Clone, Debug)]
struct Dyadic {
    n: BigUint,
    bits: u64,
}

impl Dyadic {
    fn zero() -> Self {
        Self::scaled(0, 0)
    }

    fn one() -> Self {
        Self::scaled(1, 0)
    }

    /// mantissa · 2^exponent.
    fn scaled(mantissa: u64, exponent: i64) -> Self {
        if exponent < 0 {
            Self {
                n: mantissa.into(),
                bits: exponent.unsigned_abs(),
            }
        } else {
            Self {
                n: BigUint::from(mantissa) << exponent.unsigned_abs(),
                bits: 0,
            }
        }
    }

    /// A finite float that is not negative, exactly.
    fn from_f64(value: f64) -> Self {
        assert!(
            value.is_finite() && 0.0 <= value,
            "{value} is no finite distance"
        );
        let raw = value.to_bits();
        let exponent = ((raw >> 52) & 0x7ff) as i64;
        let fraction = raw & ((1 << 52) - 1);
        if exponent == 0 {
            Self::scaled(fraction, -1074)
        } else {
            Self::scaled(fraction | 1 << 52, exponent - 1075)
        }
    }

    /// e^ln to 53 bits, for any ln a float holds: ln −∞ is zero. The one
    /// rounding that says where an edge is.
    fn exp(ln: f64) -> Self {
        if ln == f64::NEG_INFINITY {
            return Self::zero();
        }
        assert!(ln.is_finite(), "{ln} is no log mass");
        let halvings = (ln / LN_2).floor();
        let rest = (ln - halvings * LN2_HIGH) - halvings * LN2_LOW;
        let mantissa = (rest.exp() * 2f64.powi(52)).round();
        Self::scaled(mantissa as u64, halvings as i64 - 52)
    }

    /// The numerator over 2^bits, for bits no fewer than its own.
    fn raised(&self, bits: u64) -> BigUint {
        &self.n << (bits - self.bits)
    }

    fn cmp(&self, other: &Self) -> Ordering {
        let bits = self.bits.max(other.bits);
        self.raised(bits).cmp(&other.raised(bits))
    }

    fn add(&self, other: &Self) -> Self {
        let bits = self.bits.max(other.bits);
        Self {
            n: self.raised(bits) + other.raised(bits),
            bits,
        }
    }

    /// How this compares with `other`, and the distance between them.
    fn distance(&self, other: &Self) -> (Ordering, Self) {
        let bits = self.bits.max(other.bits);
        let (a, b) = (self.raised(bits), other.raised(bits));
        let side = a.cmp(&b);
        let n = if side == Ordering::Less { b - a } else { a - b };
        (side, Self { n, bits })
    }

    fn mul(&self, other: &Self) -> Self {
        Self {
            n: &self.n * &other.n,
            bits: self.bits + other.bits,
        }
    }

    /// self / other, rounded down to `bits` bits.
    fn div(&self, other: &Self, bits: u64) -> Self {
        Self {
            n: (&self.n << (other.bits + bits)) / (&other.n << self.bits),
            bits,
        }
    }

    /// Rounded down to at most `bits` bits.
    fn floor(&self, bits: u64) -> Self {
        if self.bits <= bits {
            self.clone()
        } else {
            Self {
                n: &self.n >> (self.bits - bits),
                bits,
            }
        }
    }

    fn ln(&self) -> f64 {
        let length = self.n.bits();
        if length == 0 {
            return f64::NEG_INFINITY;
        }
        // A mantissa in [1, 2) and a whole exponent: no digit is lost to
        // the logs of two large numbers cancelling.
        let shift = length.saturating_sub(64);
        let top = u64::try_from(&self.n >> shift).expect("at most 64 bits remain");
        let mantissa = top as f64 / 2f64.powi((length - 1 - shift) as i32);
        mantissa.ln() + (length as f64 - 1.0 - self.bits as f64) * LN_2
    }
}

/// The fraction of a node's mass above an edge.
fn above(ln_below: f64) -> Dyadic {
    match Dyadic::one().distance(&Dyadic::exp(ln_below)) {
        // A mass rounded a bit past the whole node.
        (Ordering::Less, _) => Dyadic::zero(),
        (_, above) => above,
    }
}

/// A point of a node: the fraction of its mass above it, in [0, 1).
#[derive(Clone, Debug)]
pub struct Point(Dyadic);

impl Point {
    /// The point whose fraction has these base-256 digits.
    pub fn new(digits: &[u8]) -> Self {
        Self(Dyadic {
            n: BigUint::from_bytes_be(digits),
            bits: 8 * digits.len() as u64,
        })
    }

    /// Its digits, ending in no zero: a node's top has none.
    pub fn digits(&self) -> Vec<u8> {
        let bits = self.0.bits.next_multiple_of(8);
        let bytes = self.0.raised(bits).to_bytes_be();
        let mut digits = vec![0; (bits / 8) as usize - bytes.len().min((bits / 8) as usize)];
        digits.extend(bytes);
        while digits.last() == Some(&0) {
            digits.pop();
        }
        digits
    }

    /// Whether the point is at or below the edge.
    pub fn below(&self, ln_below: f64) -> bool {
        above(ln_below).cmp(&self.0) != Ordering::Greater
    }

    /// Whether a child between these edges holds the point: its top is its
    /// own, its bottom the next child's.
    pub fn within(&self, top: f64, bottom: f64) -> bool {
        self.below(top) && !self.below(bottom)
    }

    /// Whether the edge is under the point, and ln of the mass between them.
    pub fn to(&self, ln_below: f64) -> (bool, f64) {
        let (side, distance) = above(ln_below).distance(&self.0);
        (side == Ordering::Greater, distance.ln())
    }

    /// The same point as one of the child holding it, between these edges,
    /// rounded down to `bits` bits or its own, whichever is more: a point
    /// that came [`out_of_child`](Self::out_of_child) of the child goes back exactly.
    pub fn into_child(&self, top: f64, bottom: f64, bits: u64) -> Point {
        let top = above(top);
        let (_, width) = above(bottom).distance(&top);
        let (_, offset) = self.0.distance(&top);
        Point(offset.div(&width, bits.max(self.0.bits)))
    }

    /// The same point as one of the parent of the node it is in, a child
    /// between these edges; exact.
    pub fn out_of_child(&self, top: f64, bottom: f64) -> Point {
        let top = above(top);
        let (_, width) = above(bottom).distance(&top);
        Point(top.add(&width.mul(&self.0)))
    }

    /// The point `distance` · e^`ln_unit` of the node's mass further down,
    /// or up; exact. None where that leaves the node.
    pub fn moved(&self, down: bool, distance: f64, ln_unit: f64) -> Option<Point> {
        let by = Dyadic::from_f64(distance).mul(&Dyadic::exp(ln_unit));
        let moved = if down {
            self.0.add(&by)
        } else {
            match self.0.distance(&by) {
                (Ordering::Less, _) => return None,
                (_, moved) => moved,
            }
        };
        (moved.cmp(&Dyadic::one()) == Ordering::Less).then_some(Point(moved))
    }

    /// Rounded down to `bits` bits.
    pub fn floor(&self, bits: u64) -> Point {
        Point(self.0.floor(bits))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn digits_are_a_fraction_in_base_256_without_trailing_zeros() {
        assert_eq!(Point::new(&[]).digits(), [] as [u8; 0]);
        assert_eq!(Point::new(&[0, 0]).digits(), [] as [u8; 0]);
        assert_eq!(Point::new(&[0x80, 0]).digits(), [0x80]);
        assert_eq!(Point::new(&[0, 1]).digits(), [0, 1]);
        let moved = Point::new(&[0x80])
            .moved(true, 2f64.powi(-10), 0.0)
            .unwrap();
        assert_eq!(moved.digits(), [0x80, 0x40]);
        assert_eq!(moved.floor(8).digits(), [0x80]);
        assert!(Point::new(&[0x80]).moved(true, 0.5, 0.0).is_none());
        assert!(Point::new(&[0x80]).moved(false, 0.75, 0.0).is_none());
        assert_eq!(
            Point::new(&[0x80]).moved(false, 0.5, 0.0).unwrap().digits(),
            [] as [u8; 0]
        );
    }

    #[test]
    fn a_point_keeps_its_side_of_an_edge_at_any_depth() {
        // The edge at half the node, and points 2^-800 either side of it.
        let half = 0.5f64.ln();
        let mut over = vec![0x7f];
        over.extend([0xff; 99]);
        let mut under = vec![0x80];
        under.extend([0; 98]);
        under.push(1);
        assert!(!Point::new(&over).below(half));
        assert!(Point::new(&[0x80]).below(half));
        assert!(Point::new(&under).below(half));
        let (edge_under, ln) = Point::new(&over).to(half);
        assert!(edge_under);
        assert!((ln - -800.0 * LN_2).abs() < 1e-9);
        let (edge_under, ln) = Point::new(&under).to(half);
        assert!(!edge_under);
        assert!((ln - -800.0 * LN_2).abs() < 1e-9);
        // The node's own edges: all of it is under its top.
        assert_eq!(Point::new(&[0x40]).to(0.0), (false, 0.25f64.ln()));
        assert_eq!(
            Point::new(&[0x40]).to(f64::NEG_INFINITY),
            (true, 0.75f64.ln())
        );
        assert!(Point::new(&[]).within(0.0, f64::NEG_INFINITY));
    }

    #[test]
    fn a_point_goes_out_of_a_child_and_back_exactly() {
        // A child deep in a tail, between edges no fraction of a float tells apart.
        let tiny = -3000.0 * LN_2;
        let (top, bottom) = (tiny + 3f64.ln(), tiny + 2f64.ln());
        let inside = Point::new(&[0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde, 0xf0, 0x11]);
        let outside = inside.out_of_child(top, bottom);
        assert!(outside.within(top, bottom));
        assert!(380 < outside.digits().len());
        assert_eq!(outside.into_child(top, bottom, 0).digits(), inside.digits());
        // Any other point comes back within the bits asked for.
        let near = outside.moved(true, 1.0, -3100.0 * LN_2).unwrap();
        let back = near.into_child(top, bottom, 64).out_of_child(top, bottom);
        let (_, apart) = back.0.distance(&near.0);
        assert!(apart.ln() < tiny - 64.0 * LN_2);
    }
}
