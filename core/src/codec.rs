use serde::{Deserialize, Serialize};

use crate::tree::Listing;

pub const SATURATED: f64 = 0x7fff_ffff as f64;
const LOWEST: f64 = -0x8000_0000i64 as f64;

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct WireEdge {
    pub label: u32,
    pub odds: f64,
}

/// Nats per odds step at `horizon` bits.
pub fn step(horizon: f64) -> f64 {
    horizon * std::f64::consts::LN_2 / 2f64.powi(31)
}

/// ln σ(λ), without overflow at either end.
pub fn ln_sigmoid(lambda: f64) -> f64 {
    if lambda == f64::INFINITY {
        0.0
    } else if lambda >= 0.0 {
        -(-lambda).exp().ln_1p()
    } else {
        lambda - lambda.exp().ln_1p()
    }
}

fn lambda(odds: f64, unit: f64) -> f64 {
    if odds == SATURATED {
        f64::INFINITY
    } else {
        odds * unit
    }
}

#[derive(Debug)]
struct Decoder {
    labels: Vec<u32>,
    ln_weights: Vec<f64>,
    unit: f64,
    leftover: f64,
    slack: f64,
    stopped: bool,
    complete: f64,
}

impl Decoder {
    fn new(horizon: f64, complete: f64) -> Self {
        Self {
            labels: Vec::new(),
            ln_weights: Vec::new(),
            unit: step(horizon),
            leftover: 1.0,
            slack: 0.0,
            stopped: false,
            complete,
        }
    }

    fn push(&mut self, edge: WireEdge) -> bool {
        if self.stopped {
            return false;
        }
        let l = lambda(edge.odds, self.unit);
        let previous = self.ln_weights.last().copied().unwrap_or(0.0);
        let ln_weight = previous + ln_sigmoid(l);
        let weight = ln_weight.exp();
        if self.ln_weights.is_empty() {
            self.leftover = ln_sigmoid(-l).exp();
        } else if weight > self.leftover + self.slack {
            self.stopped = true;
            return false;
        } else {
            self.leftover -= weight;
            self.slack += self.unit * weight;
        }
        self.labels.push(edge.label);
        self.ln_weights.push(ln_weight);
        true
    }

    fn listing(&self, horizon: f64) -> Listing {
        let tail = self.leftover > self.slack;
        // A leftover within rounding goes to the listed children, which then
        // partition the interval exactly.
        let shift = if tail { 0.0 } else { -(-self.leftover).ln_1p() };
        Listing {
            horizon,
            labels: self.labels.clone(),
            ln_weights: self
                .ln_weights
                .iter()
                .map(|weight| weight + shift)
                .collect(),
            leftover: if tail { self.leftover } else { 0.0 },
            complete: self.complete,
        }
    }
}

pub fn decode(edges: &[WireEdge], horizon: f64, complete: f64) -> Listing {
    let mut decoder = Decoder::new(horizon, complete);
    for &edge in edges {
        if !decoder.push(edge) {
            break;
        }
    }
    decoder.listing(horizon)
}

/// JavaScript's Math.round: nearest integer, with halves toward +infinity.
fn js_round(value: f64) -> f64 {
    let floor = value.floor();
    if value - floor < 0.5 {
        floor
    } else {
        floor + 1.0
    }
}

/// Edges for children sorted by decreasing weight, down to 2^-horizon.
pub fn encode(labels: &[u32], ln_weights: &[f64], horizon: f64) -> Vec<WireEdge> {
    if horizon <= 0.0 {
        return Vec::new();
    }
    let unit = step(horizon);
    let floor = -horizon * std::f64::consts::LN_2;
    let mut edges = Vec::new();
    let mut previous = 0.0;
    for i in 0..labels.len() {
        let ln_weight = ln_weights[i];
        // Like the TypeScript guard, this also stops on NaN.
        if ln_weight
            .partial_cmp(&floor)
            .is_none_or(|order| order == std::cmp::Ordering::Less)
        {
            break;
        }
        let ln_g = (ln_weight - previous).min(0.0);
        let mut odds = SATURATED;
        if ln_g < 0.0 && unit > 0.0 {
            let l = ln_g - (-ln_g.exp_m1()).ln();
            let rounded = js_round(l / unit);
            odds = rounded.clamp(LOWEST, SATURATED);
        }
        edges.push(WireEdge {
            label: labels[i],
            odds,
        });
        previous += ln_sigmoid(lambda(odds, unit));
    }
    edges
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ln(weights: &[f64]) -> Vec<f64> {
        weights.iter().map(|weight| weight.ln()).collect()
    }

    #[test]
    fn encoding_matches_the_exporter_and_stops_at_the_horizon() {
        let weights = [0.5, 0.3, 0.3, 0.15, 0.001, 1e-6];
        let edges = encode(&[7, 3, 9, 1, 4, 2], &ln(&weights), 12.0);
        assert_eq!(
            edges,
            vec![
                WireEdge {
                    label: 7,
                    odds: 0.0
                },
                WireEdge {
                    label: 3,
                    odds: 104_683_117.0,
                },
                WireEdge {
                    label: 9,
                    odds: SATURATED,
                },
                WireEdge {
                    label: 1,
                    odds: 0.0
                },
                WireEdge {
                    label: 4,
                    odds: -1_291_920_529.0,
                },
            ]
        );
        assert_eq!(
            encode(&[0], &[(-1e-12f64).ln_1p()], 48.0),
            vec![WireEdge {
                label: 0,
                odds: 1_783_446_566.0,
            }]
        );
        let stopped = encode(&[0, 1, 2], &ln(&[0.5, 0.25, 0.2]), 2.0);
        assert_eq!(
            stopped.iter().map(|edge| edge.label).collect::<Vec<_>>(),
            [0, 1]
        );
    }

    #[test]
    fn round_trip_recovers_weights_and_keeps_ties_exact() {
        let weights = [0.41, 0.2, 0.13, 0.1, 0.07, 0.05, 0.03, 0.01];
        let horizon = 20.0;
        let labels = (0..weights.len() as u32).collect::<Vec<_>>();
        let listing = decode(&encode(&labels, &ln(&weights), horizon), horizon, horizon);
        for (actual, expected) in listing.ln_weights.iter().zip(weights) {
            assert!((actual - expected.ln()).abs() <= step(horizon));
        }
        assert_eq!(listing.leftover, 0.0);

        let ties = decode(&encode(&[0, 1, 2, 3], &ln(&[0.25; 4]), 8.0), 8.0, 8.0);
        assert!(ties.ln_weights[1..]
            .iter()
            .all(|weight| *weight == ties.ln_weights[0]));
        assert_eq!(ties.leftover, 0.0);
    }

    #[test]
    fn decoding_preserves_tail_and_stops_on_overflow() {
        let near_certain = decode(&encode(&[0], &[(-1e-12f64).ln_1p()], 48.0), 48.0, 48.0);
        assert!((near_certain.leftover / 1e-12 - 1.0).abs() < 1e-6);

        let horizon = 12.0;
        let listing = decode(
            &encode(&[7, 3, 9, 1], &ln(&[0.5, 0.3, 0.3, 0.15]), horizon),
            horizon,
            horizon,
        );
        assert_eq!(listing.labels, [7, 3]);
        assert!((listing.leftover - 0.2).abs() < 1e-6);

        let empty = decode(&[], 10.0, 10.0);
        assert!(empty.labels.is_empty());
        assert!(empty.ln_weights.is_empty());
        assert_eq!(empty.leftover, 1.0);
        assert_eq!(empty.complete, 10.0);
    }

    #[test]
    fn arbitrary_odds_remain_non_increasing() {
        let mut seed = 7i64;
        let edges = (0..50)
            .map(|label| {
                seed = (seed * 48_271) % 0x7fff_ffff;
                WireEdge {
                    label,
                    odds: (seed - 0x4000_0000) as f64,
                }
            })
            .collect::<Vec<_>>();
        let listing = decode(&edges, 30.0, 30.0);
        assert!(listing.ln_weights.windows(2).all(|pair| pair[1] <= pair[0]));
    }
}
