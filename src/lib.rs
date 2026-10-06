//! A deterministic tensor engine with reverse-mode autodiff ([`tensor`]) and a neural-network
//! library on it ([`nn`]), on the CPU and, behind features, Apple Metal and NVIDIA CUDA.
//!
//! The reference CPU backend is bit-exact: the same inputs give the same bits on every run. The
//! fast CPU backend and the GPU backends are checked against it within stated tolerances.

pub mod tensor;
pub mod nn;
pub mod rng;

mod error;
mod hash;

pub use error::{Error, NnError, Result};

#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctests;
