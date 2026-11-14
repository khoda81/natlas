//! Integer-sequence distributions, independent of the renderer and transport.
pub mod distribution;
pub mod layout;
mod packed;
pub mod point;
pub mod session;
pub mod tree;

pub type Result<T> = std::result::Result<T, String>;

#[cfg(feature = "python")]
mod python;
#[cfg(feature = "wasm")]
mod wasm;
