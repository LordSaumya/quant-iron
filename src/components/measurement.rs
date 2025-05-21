use crate::components::state::State;
use std::ops::Deref;

#[derive(Debug, Clone, PartialEq)]
/// Represents the result of a measurement on a quantum state.
///
/// # Fields
///
/// * `basis` - The basis of measurement (e.g., computational basis).
/// * `indices` - The indices of the measured qubits.
/// * `outcomes` - The measurement outcomes for the qubits.
/// * `new_state` - The new state vector after the measurement.
pub struct MeasurementResult {
    /// The basis of measurement
    pub basis: MeasurementBasis,
    /// The indices of the measured qubits.
    pub indices: Vec<usize>,
    /// The measurement outcomes for the qubits.
    /// Represented as a vector of bits (0 or 1).
    pub outcomes: Vec<u8>,
    /// The new state vector after the measurement.
    pub new_state: State,
}

// Allow dereferencing to the new state vector for method chaining.
impl Deref for MeasurementResult {
    type Target = State;

    fn deref(&self) -> &Self::Target {
        &self.new_state
    }
}

impl MeasurementResult {
    /// Gets the measured indices of the qubits.
    ///
    /// # Returns
    ///
    /// * `indices` - A vector of indices of the measured qubits.
    pub fn get_indices(&self) -> &Vec<usize> {
        &self.indices
    }

    /// Gets the basis of measurement.
    ///
    /// # Returns
    ///
    /// * `basis` - The basis of measurement.
    pub fn get_basis(&self) -> &MeasurementBasis {
        &self.basis
    }

    /// Gets the measurement outcomes for the qubits.
    ///
    /// # Returns
    ///
    /// * `outcomes` - A vector of measurement outcomes for the qubits.
    pub fn get_outcomes(&self) -> &Vec<u8> {
        &self.outcomes
    }

    /// Gets the new state vector after the measurement.
    ///
    /// # Returns
    ///
    /// * `new_state` - The new state vector after the measurement.
    pub fn get_new_state(&self) -> &State {
        &self.new_state
    }
}

/// Represents the basis of measurement for qubits.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MeasurementBasis {
    /// The computational basis |0> and |1>.
    /// Also known as the standard basis or Z basis.
    Computational,
    /// The X basis (|+> and |->).
    X,
    /// The Y basis (|i+> and |i->).
    Y,
}
