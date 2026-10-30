//! Integer-sequence distributions, independent of the renderer and transport.
pub mod codec;
pub mod layout;
pub mod tree;

pub type Result<T> = std::result::Result<T, String>;

#[cfg(feature = "python")]
mod python;
#[cfg(feature = "wasm")]
mod wasm;
