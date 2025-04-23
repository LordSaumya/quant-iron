#[derive(Debug, thiserror::Error, Clone, PartialEq)]
pub enum Error {
    /// There is an invalid number of measurements
    /// 
    /// # Arguments:
    /// 
    /// * `0` - The invalid number of measurements
    #[error("Invalid number of measurements: {0}")]
    InvalidNumberOfMeasurements(usize),

    /// There is an invalid number of qubits
    /// 
    /// # Arguments:
    /// 
    /// * `0` - The invalid number of qubits
    #[error("Invalid number of qubits: {0}")]
    InvalidNumberOfQubits(usize),

    /// A qubit index is invalid for the number of qubits
    /// 
    /// # Arguments:
    /// 
    /// * `0` - The invalid qubit index
    /// * `1` - The number of qubits
    #[error("Invalid qubit index: {0} for {1} qubits")]
    InvalidQubitIndex(usize, usize),

    /// The state vector is not normalised
    #[error("State vector is not normalised")]
    StateVectorNotNormalised,

    /// Unexpected error occurred
    #[error("An unknown error occurred")]
    UnknownError,
}