use crate::{
    circuit::Circuit,
    compiler::ir::InstructionIR,
    components::gate::Gate,
    components::{
        measurement::MeasurementOperation,
        operator::{
            CNOT, Hadamard, Identity, Pauli, PhaseS, PhaseSdag, PhaseShift, PhaseT, PhaseTdag,
            RotateX, RotateY, RotateZ, SWAP, Toffoli, Unitary2,
        },
    },
    errors::CompilerError,
};
use dyn_clone::DynClone;
use rayon::prelude::*;
use std::convert::TryFrom;

/// Trait for operators or measurements that can be compiled into an IR representation
///
/// This trait defines the `to_ir` method, which converts an operator into an
/// IR representation. The IR representation is a structured format that can be
/// used for further processing, such as optimisation or execution on a quantum
/// circuit simulator or hardware.
pub trait Compilable: DynClone + Send + Sync + 'static {
    /// Converts the operation into an IR representation
    ///
    /// # Returns
    /// A vector of `InstructionIR` representing the operation in IR format.
    fn to_ir(&self, targets: Vec<usize>, controls: Vec<usize>) -> Vec<InstructionIR>;
}

dyn_clone::clone_trait_object!(Compilable);

/// Represents a quantum circuit that can be compiled into an IR representation
///
/// This is an internal struct used to convert a quantum circuit into an IR representation
pub(crate) struct CompilableCircuit {
    /// The number of qubits in the circuit
    pub num_qubits: usize,

    /// The list of gates in the circuit
    ///
    /// All gates in the circuit must implement the `Compilable` trait.
    pub gates: Vec<CompilableGate>,
}

impl CompilableCircuit {
    /// Converts the `CompilableCircuit` into an IR representation
    ///
    /// # Returns
    /// A vector of `InstructionIR` representing the circuit in IR format.
    pub(crate) fn to_ir(&self) -> Vec<InstructionIR> {
        self.gates
            .par_iter()
            .flat_map(|gate| {
                gate.operator
                    .to_ir(gate.targets.clone(), gate.controls.clone())
            })
            .collect()
    }
}

/// Internal struct representing a gate that can be compiled into an IR representation
///
/// This struct is used to hold the operator, target qubits, and control qubits of a gate.
pub(crate) struct CompilableGate {
    /// The operator of the gate
    pub operator: Box<dyn Compilable>,
    /// The target qubits of the gate
    pub targets: Vec<usize>,
    /// The control qubits of the gate
    pub controls: Vec<usize>,
}

impl TryFrom<&Circuit> for CompilableCircuit {
    type Error = CompilerError;

    /// Converts a `Circuit` into a `CompilableCircuit`
    ///
    /// # Arguments
    /// * `circuit` - The circuit to convert
    ///
    /// # Returns
    /// A `Result` containing the `CompilableCircuit` or an error message if conversion fails.
    fn try_from(circuit: &Circuit) -> Result<Self, CompilerError> {
        let num_qubits: usize = circuit.get_num_qubits();
        let mut compilable_gates: Vec<CompilableGate> = Vec::with_capacity(circuit.gates.len() * 2); // Preallocate space for operators

        for gate in &circuit.gates {
            match gate {
                Gate::Operator(op, _targets, _controls) => {
                    if let Some(compilable_op) = op.to_compilable() {
                        // Convert the operator to a compilable form
                        let targets = _targets.clone();
                        let controls = _controls.clone();
                        let operator = dyn_clone::clone_box(compilable_op);

                        // Create a CompilableGate for the operator
                        let gate: CompilableGate = CompilableGate {
                            operator,
                            targets,
                            controls,
                        };
                        compilable_gates.push(gate);
                    } else {
                        return Err(CompilerError::UnsupportedOperator(
                            "Operator does not implement Compilable trait".to_string(),
                        ));
                    }
                }
                Gate::Measurement(measurement_basis, targets) => {
                    // All measurement operations are compilable, no trait check needed
                    let measurement_op = MeasurementOperation {
                        basis: *measurement_basis,
                    };

                    // Create a CompilableGate for the measurement operation
                    let gate = CompilableGate {
                        operator: Box::new(measurement_op),
                        targets: targets.clone(),
                        controls: vec![],
                    };

                    compilable_gates.push(gate);
                },
                Gate::Parametric(_, _, _) => {
                    unreachable!("All parametric gates should be converted to concrete gates before compilation")
                },
                Gate::PauliString(_) => {
                    unreachable!("All Pauli strings should be converted to Pauli gates before compilation")
                },
                Gate::PauliTimeEvolution(_, _) => {
                    unimplemented!("Compilation for Pauli time evolution gates is not yet implemented")
                }
            }
        }
        Ok(CompilableCircuit {
            num_qubits,
            gates: compilable_gates,
        })
    }
}

impl Compilable for Hadamard {
    fn to_ir(&self, targets: Vec<usize>, controls: Vec<usize>) -> Vec<InstructionIR> {
        targets
            .par_iter()
            .map(|&target| InstructionIR::Hadamard(target, controls.clone()))
            .collect()
    }
}

impl Compilable for Pauli {
    fn to_ir(&self, targets: Vec<usize>, controls: Vec<usize>) -> Vec<InstructionIR> {
        targets
            .par_iter()
            .map(|&target| match self {
                Pauli::X => InstructionIR::PauliX(target, controls.clone()),
                Pauli::Y => InstructionIR::PauliY(target, controls.clone()),
                Pauli::Z => InstructionIR::PauliZ(target, controls.clone()),
            })
            .collect()
    }
}

impl Compilable for CNOT {
    fn to_ir(&self, targets: Vec<usize>, controls: Vec<usize>) -> Vec<InstructionIR> {
        targets
            .par_iter()
            .map(|&target| {
                if controls.is_empty() {
                    // If no controls specified, treat as unconditional X gate
                    InstructionIR::PauliX(target, vec![])
                } else {
                    // For CNOT, only use the first control
                    InstructionIR::PauliX(target, vec![controls[0]])
                }
            })
            .collect()
    }
}

impl Compilable for SWAP {
    fn to_ir(&self, targets: Vec<usize>, controls: Vec<usize>) -> Vec<InstructionIR> {
        // SWAP requires exactly 2 target qubits
        if targets.len() >= 2 {
            // Create SWAP instructions for pairs of qubits
            targets
                .par_chunks(2)
                .filter_map(|chunk| {
                    if chunk.len() == 2 {
                        Some(InstructionIR::Swap(chunk[0], chunk[1], controls.clone()))
                    } else {
                        None
                    }
                })
                .collect()
        } else {
            vec![]
        }
    }
}

impl Compilable for Toffoli {
    fn to_ir(&self, targets: Vec<usize>, controls: Vec<usize>) -> Vec<InstructionIR> {
        targets
            .par_iter()
            .map(|&target| {
                // Toffoli is a controlled X gate with additional controls
                InstructionIR::PauliX(target, controls.clone())
            })
            .collect()
    }
}

impl Compilable for Identity {
    fn to_ir(&self, targets: Vec<usize>, controls: Vec<usize>) -> Vec<InstructionIR> {
        targets
            .par_iter()
            .map(|&target| InstructionIR::Id(target, controls.clone()))
            .collect()
    }
}

impl Compilable for PhaseS {
    fn to_ir(&self, targets: Vec<usize>, controls: Vec<usize>) -> Vec<InstructionIR> {
        targets
            .par_iter()
            .map(|&target| InstructionIR::S(target, controls.clone()))
            .collect()
    }
}

impl Compilable for PhaseT {
    fn to_ir(&self, targets: Vec<usize>, controls: Vec<usize>) -> Vec<InstructionIR> {
        targets
            .par_iter()
            .map(|&target| InstructionIR::T(target, controls.clone()))
            .collect()
    }
}

impl Compilable for PhaseSdag {
    fn to_ir(&self, targets: Vec<usize>, controls: Vec<usize>) -> Vec<InstructionIR> {
        targets
            .par_iter()
            .map(|&target| InstructionIR::Sdg(target, controls.clone()))
            .collect()
    }
}

impl Compilable for PhaseTdag {
    fn to_ir(&self, targets: Vec<usize>, controls: Vec<usize>) -> Vec<InstructionIR> {
        targets
            .par_iter()
            .map(|&target| InstructionIR::Tdg(target, controls.clone()))
            .collect()
    }
}

impl Compilable for PhaseShift {
    fn to_ir(&self, targets: Vec<usize>, controls: Vec<usize>) -> Vec<InstructionIR> {
        targets
            .par_iter()
            .map(|&target| InstructionIR::Phase(self.angle, target, controls.clone()))
            .collect()
    }
}

impl Compilable for RotateX {
    fn to_ir(&self, targets: Vec<usize>, controls: Vec<usize>) -> Vec<InstructionIR> {
        targets
            .par_iter()
            .map(|&target| InstructionIR::Rx(self.angle, target, controls.clone()))
            .collect()
    }
}

impl Compilable for RotateY {
    fn to_ir(&self, targets: Vec<usize>, controls: Vec<usize>) -> Vec<InstructionIR> {
        targets
            .par_iter()
            .map(|&target| InstructionIR::Ry(self.angle, target, controls.clone()))
            .collect()
    }
}

impl Compilable for RotateZ {
    fn to_ir(&self, targets: Vec<usize>, controls: Vec<usize>) -> Vec<InstructionIR> {
        targets
            .par_iter()
            .map(|&target| InstructionIR::Rz(self.angle, target, controls.clone()))
            .collect()
    }
}

impl Compilable for Unitary2 {
    fn to_ir(&self, targets: Vec<usize>, controls: Vec<usize>) -> Vec<InstructionIR> {
        targets
            .par_iter()
            .map(|&target| InstructionIR::Unitary(self.matrix, target, controls.clone()))
            .collect()
    }
}
