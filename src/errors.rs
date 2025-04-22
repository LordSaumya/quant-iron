#[derive(Debug, thiserror::Error, Clone, PartialEq)]
pub enum Error {
    #[error("Invalid number of qubits: {0}")]
    InvalidNumberOfQubits(usize),
    #[error("Invalid qubit index: {0} for {1} qubits")]
    InvalidQubitIndex(usize, usize),
    #[error("State vector is not normalised")]
    StateVectorNotNormalised,
    #[error("An unknown error occurred")]
    UnknownError,
}