//! A high-performance, hardware-accelerated modular quantum computing library with a focus on physical applications.
//!
//! Quant-Iron provides tools to represent quantum states, apply standard quantum gates, perform measurements, build quantum circuits, and implement quantum algorithms.
//! 
//! ## Features
//! - **Quantum State Representation**: Create and manipulate predefined or custom quantum states of arbitrary qubit count.
//! - **Standard Operations**: Hadamard (H), Pauli (X, Y, Z), CNOT, SWAP, Toffoli, Phase shifts, Rotations, and custom unitary operations.
//! - **Hardware Acceleration**: Optimised for parallel execution (CPU and GPU) and low memory overhead, with OpenCL-accelerated operations for enhanced performance on compatible hardware. (Requires `gpu` feature flag).
//! - **Circuit Builder**: High-level interface for constructing quantum circuits with a fluent API and the `circuit!` macro with support for subroutines.
//! - **Measurement**: Collapse wavefunction in the measurement basis with single or repeated measurements in the `Computational`, `X`, `Y`, and custom bases.
//! - **Pauli String Algebra**:
//!   - Represent products of Pauli operators with complex coefficients (`PauliString`).
//!   - Construct sums of Pauli strings (`SumOp`) to define Hamiltonians and other observables.
//!   - Apply Pauli strings and their sums to quantum states.
//!   - Calculate expectation values of `SumOp` with respect to a quantum state.
//!   - Apply exponentials of `PauliString` instances to states.
//! - **Predefined Quantum Models**:
//!   - **Heisenberg Model**: Generate Hamiltonians for 1D and 2D anisotropic Heisenberg models using `SumOp`.
//!   - **Ising Model**: Generate Hamiltonians for 1D and 2D Ising models with configurable site-specific or uniform interactions and fields using `SumOp`.
//! - **Predefined Quantum Algorithms**:
//!   - **Quantum Fourier Transform (QFT)**: Efficiently compute the QFT for a given number of qubits.
//!   - **Inverse Quantum Fourier Transform (IQFT)**: Efficiently compute the inverse QFT for a given number of qubits.
//! - **OpenQASM 3.0 Support**: Convert circuits to OpenQASM 3.0 for interoperability with other quantum computing platforms.
//! - **Extensibility**: Easily extensible for custom gates and measurement bases.
//! - **Error Handling**: Comprehensive error handling for invalid operations and state manipulations.
//! - **Quality of Life**: Implementation of `std` and arithmetic traits for easy, intuitive usage.
//! 
//! ## Important Components
//! 
//! - [Circuit](mod@circuit): Provides the `Circuit` and `CircuitBuilder` for constructing quantum circuits.
//! - [Components](components): Contains the core components of the library:
//!   - [Operator]: Contains standard quantum gates and operators like Hadamard
//!   - [State]: Represents quantum states and provides methods for state manipulation.
//!   - [PauliString]: Represents products of Pauli operators with complex coefficients.
//!   - [SumOp]: Represents sums of Pauli strings for defining Hamiltonians and other observables.
//!   - [MeasurementResult]: Represents the result of a measurement operation, including indices, basis, outcomes, and new state.
//! - [Subroutine](subroutine): Contains the `Subroutine` for defining reusable quantum subroutines.
//! - [Models](models): Contains predefined quantum models like the Heisenberg and Ising models.
//! - [Algorithms](algorithms): Contains predefined quantum algorithms like time evolution.
//! 
//! Refer to the examples in the `examples` directory for practical usage of the library.
//! 
//! ## Feature Flags
//! - `gpu`: Enables OpenCL support for hardware acceleration for operations on compatible hardware.

#[macro_use]
mod macros;
pub mod circuit;
pub mod components;
pub mod errors;
pub mod subroutine;
pub mod models;
pub mod algorithms;
pub(crate) mod compiler;

// Re-export important types for easier imports
pub use crate::components::measurement::{MeasurementBasis, MeasurementResult};
pub use crate::components::operator::{
    CNOT, Hadamard, Identity, Operator, Pauli, PhaseS, PhaseSdag, PhaseShift, PhaseT, PhaseTdag,
    RotateX, RotateY, RotateZ, SWAP, Toffoli, Unitary2, Matchgate
};
pub use crate::components::state::{ChainableState, State};
pub use crate::components::pauli_string::{PauliString, SumOp};
pub use crate::circuit::{Circuit, CircuitBuilder};
pub use crate::subroutine::Subroutine;
pub use crate::models::{heisenberg, ising};
pub use crate::algorithms::time_evolution;

#[cfg(test)]
pub mod tests;
