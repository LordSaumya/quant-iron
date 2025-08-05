//! # Components
//! 
//! This module contains the core components for quantum operations, including gates, measurements, and state management.
//! It provides the necessary structures and traits to define quantum circuits and operations.
//! - [State]: Represents the quantum state of a system, and allows for operations on that state.
//! - [Gate](gate): Defines quantum gates that can be applied to qubits.
//! - [Measurement](measurement): Provides functionality for quantum measurements in the X, Y, Z, and custom bases.
//! - [Operator](operator): Represents quantum operators that can be applied to states.
//! - [pauli_string]: Represents a string of Pauli operators with a coefficient, useful for defining Hamiltonians and other operations.

pub mod gate;
#[cfg(feature = "gpu")]
pub(crate) mod gpu_context;
pub mod measurement;
pub mod operator;
pub mod state;
pub mod pauli_string;

// Re-export important types for easier use within the crate
pub use measurement::{MeasurementBasis, MeasurementResult};
pub use state::{ChainableState, State};
