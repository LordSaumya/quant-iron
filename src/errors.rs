#[derive(Debug, thiserror::Error, Clone, PartialEq)]
pub enum Error {
    /// There is an invalid number of measurements
    /// 
    /// # Arguments:
    /// 
    /// * `0` - The invalid number of measurements
    #[error("Invalid number of measurements: {0}")]
    InvalidNumberOfMeasurements(usize),

    /// The control qubits and target qubit indices are overlapping
    /// 
    /// # Arguments:
    /// 
    /// * `0` - The control qubit index
    /// 
    /// * `1` - The target qubit index
    #[error("Control qubit index {0} overlaps with target qubit index {1}")]
    OverlappingControlAndTargetQubits(usize, usize),

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

    /// Input matrix for arbitrary unitary operator was not unitary
    #[error("Non-unitary matrix")]
    NonUnitaryMatrix,

    /// Unexpected number of inputs
    /// 
    /// # Arguments:
    /// 
    /// * `0` - The actual number of inputs
    /// * `1` - The expected number of inputs
    #[error("Unexpected number of inputs: expected {1}, got {0}")]
    InvalidNumberOfInputs(usize, usize),

    /// The number of parameters does not match the number of target qubits.
    ///
    /// # Arguments:
    ///
    /// * `expected` - The expected number of parameters.
    /// * `actual` - The actual number of parameters.
    #[error("Mismatched number of parameters: expected {expected}, got {actual}")]
    MismatchedNumberOfParameters { expected: usize, actual: usize },

    /// Unexpected error occurred
    #[error("An unknown error occurred")]
    UnknownError,

    /// An error occurred during OpenCL operation
    ///
    /// # Arguments:
    ///
    /// * `0` - The OpenCL error message
    #[error("OpenCL error: {0}")]
    OpenCLError(String),

    /// Failed to lock the global GPU context
    #[error("Failed to lock GPU context")]
    GpuContextLockError,

    /// Failed to create circuit from macro
    #[error("Failed to create circuit from macro: {0}")]
    CircuitMacroError(String),

    /// Invalid input value for operation
    #[error("Invalid input value for operation: {0}")]
    InvalidInputValue(usize),

    /// A Pauli string contains duplicate operators for the same qubit
    ///
    /// # Arguments:
    ///
    /// * `0` - The qubit index
    #[error("Duplicate Pauli string operator for qubit: {0}")]
    DuplicatePauliStringOperator(usize),
}

#[derive(Debug, thiserror::Error, Clone, PartialEq)]
pub enum CompilerError {
    #[error("I/O error: {0}")]
    IOError(String),

    #[error("An unsupported operation was encountered: {0}")]
    UnsupportedOperator(String),

    #[error("Invalid operands ({0}) for operator {1}")]
    InvalidOperands(String, String),
}

#[cfg(feature = "gpu")]
impl From<ocl::Error> for Error {
    fn from(err: ocl::Error) -> Self {
        Error::OpenCLError(err.to_string())
    }
}
