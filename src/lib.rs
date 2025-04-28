pub mod components;
pub mod errors;
pub mod circuit;

// Re-export important types for easier imports
pub use crate::components::state::{State, ChainableState};
pub use crate::components::measurement::{MeasurementResult, MeasurementBasis};
pub use crate::components::operator::{
    CNOT, Hadamard, Identity, Operator, Pauli, PhaseS, PhaseSdag, PhaseShift, PhaseT,
    PhaseTdag, RotateX, RotateY, RotateZ, SWAP, Toffoli,
};
pub use crate::components::gate::Gate;

#[cfg(test)]
pub mod tests;