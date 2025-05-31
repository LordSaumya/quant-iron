pub mod circuit;
pub mod components;
pub mod errors;
pub mod subroutine;
pub mod models;
pub mod algorithms;

// Re-export important types for easier imports
pub use crate::components::measurement::{MeasurementBasis, MeasurementResult};
pub use crate::components::operator::{
    CNOT, Hadamard, Identity, Operator, Pauli, PhaseS, PhaseSdag, PhaseShift, PhaseT, PhaseTdag,
    RotateX, RotateY, RotateZ, SWAP, Toffoli, Unitary2
};
pub use crate::components::state::{ChainableState, State};
pub use crate::components::pauli_string::{PauliString, SumOp};
pub use crate::circuit::{Circuit, CircuitBuilder};
pub use crate::subroutine::Subroutine;
pub use crate::models::{heisenberg, ising};
pub use crate::algorithms::time_evolution;

#[cfg(test)]
pub mod tests;
