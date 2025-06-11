use num_complex::Complex;
use crate::components::measurement::MeasurementBasis;

pub(crate) struct CircuitIR {
    pub(crate) num_qubits: usize,
    pub(crate) instructions: Vec<InstructionIR>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum InstructionIR {
    // GATES

    /// IR representation for a Hadamard gate on qubit index with controls
    Hadamard(usize, Vec<usize>),
    /// IR representation for a Pauli-X gate on qubit index with controls 
    PauliX(usize, Vec<usize>),
    /// IR representation for a Pauli-Y gate on qubit index with controls
    PauliY(usize, Vec<usize>),
    /// IR representation for a Pauli-Z gate on qubit index with controls
    PauliZ(usize, Vec<usize>),
    /// IR representation for a Phase S gate on qubit index with controls
    S(usize, Vec<usize>),
    /// IR representation for a Phase T gate on qubit index with controls
    T(usize, Vec<usize>),
    /// IR representation for a Phase S-dagger gate on qubit index with controls
    Sdg(usize, Vec<usize>),
    /// IR representation for a Phase T-dagger gate on qubit index with controls
    Tdg(usize, Vec<usize>),
    /// IR representation for a Phase Shift (P) gate with an angle, on qubit index with controls
    Phase(f64, usize, Vec<usize>),
    /// IR representation for a RotateX gate with an angle, on qubit index with controls
    Rx(f64, usize, Vec<usize>),
    /// IR representation for a RotateY gate with an angle, on qubit index with controls
    Ry(f64, usize, Vec<usize>),
    /// IR representation for a RotateZ gate with an angle, on qubit index with controls
    Rz(f64, usize, Vec<usize>),
    /// IR representation for an Identity gate on qubit index with controls
    Id(usize, Vec<usize>),
    /// IR representation for a SWAP gate on two target qubits, with controls
    Swap(usize, usize, Vec<usize>),
    /// IR representation for a custom 2x2 Unitary gate, on qubit index with controls
    Unitary([[Complex<f64>; 2]; 2], usize, Vec<usize>),

    // MEASUREMENT

    /// IR representation for a measurement operation on qubit index with basis
    Measurement(usize, MeasurementBasis),

    // Allow Raw QASM Strings (without validation) for custom instructions
    /// IR representation for a raw QASM string
    /// 
    /// This allows for custom instructions that may not be directly supported by the existing IR format.
    /// 
    /// WARNING: This does not validate the QASM string, so it should be used with caution.
    RawQASM(String),
}