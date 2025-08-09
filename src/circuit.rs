use crate::{
    compiler::{compilable::CompilableCircuit, qasm::QasmCircuit},
    components::{
        gate::Gate,
        measurement::MeasurementBasis,
        operator::Operator,
        parametric::{
            parameter::Parameter,
            parametric_gate::{ParametricMatchgate, ParametricRyPhase},
        },
        state::State,
    },
    errors::{CompilerError, Error},
    subroutine::Subroutine,
};

use std::fs;
use std::io::Write;
use std::path::Path;

use num_complex::Complex;

/// Represents a quantum circuit as a vector of gates.
///
/// # Fields
///
/// * `gates` - A vector of gates in the circuit.
/// * `num_qubits` - The number of qubits in the circuit.
#[derive(Debug)]
pub struct Circuit {
    /// A vector of gates in the circuit.
    pub gates: Vec<Gate>,
    /// The number of qubits in the circuit.
    pub num_qubits: usize,
}

impl Circuit {
    // Helper function to validate gate qubits
    fn _validate_gate_qubits(gate: &Gate, circuit_num_qubits: usize) -> Result<(), Error> {
        // Check if the gate's target qubits are within the circuit's qubit range
        for &qubit in gate.get_target_qubits() {
            if qubit >= circuit_num_qubits {
                return Err(Error::InvalidQubitIndex(qubit, circuit_num_qubits));
            }
        }

        // Check if the gate's control qubits are within the circuit's qubit range
        if let Some(control_qubits) = gate.get_control_qubits() {
            for &qubit in control_qubits {
                if qubit >= circuit_num_qubits {
                    return Err(Error::InvalidQubitIndex(qubit, circuit_num_qubits));
                }
            }
        }
        Ok(())
    }

    /// Creates a new circuit with the specified number of qubits.
    ///
    /// # Arguments
    ///
    /// * `num_qubits` - The number of qubits in the circuit.
    ///
    /// # Returns
    ///
    /// * `Circuit` - A new instance of the Circuit struct.
    pub fn new(num_qubits: usize) -> Self {
        Circuit {
            gates: Vec::new(),
            num_qubits,
        }
    }

    /// Creates a new circuit with the specified gates and number of qubits.
    ///
    /// # Arguments
    ///
    /// * `gates` - A vector of gates in the circuit.
    /// * `num_qubits` - The number of qubits in the circuit.
    ///
    /// # Returns
    ///
    /// * `Result<Circuit, Error>` - A new instance of the Circuit struct or an error if the circuit cannot be created.
    pub fn with_gates(gates: Vec<Gate>, num_qubits: usize) -> Result<Circuit, Error> {
        for gate in &gates {
            Self::_validate_gate_qubits(gate, num_qubits)?;
        }

        Ok(Circuit { gates, num_qubits })
    }

    /// Adds a gate to the circuit.
    ///
    /// # Arguments
    ///
    /// * `gate` - The gate to be added to the circuit.
    ///
    /// # Returns
    ///
    /// * `Result<(), Error>` - An empty result if the gate is added successfully, or an error if the gate cannot be added.
    pub fn add_gate(&mut self, gate: Gate) -> Result<(), Error> {
        Self::_validate_gate_qubits(&gate, self.num_qubits)?;
        self.gates.push(gate);
        Ok(())
    }

    /// Adds multiple gates to the circuit.
    ///
    /// # Arguments
    ///
    /// * `gates` - A vector of gates to be added to the circuit.
    ///
    /// # Returns
    ///
    /// * `Result<(), Error>` - An empty result if the gates are added successfully, or an error if any gate cannot be added.
    pub fn add_gates(&mut self, gates: Vec<Gate>) -> Result<(), Error> {
        for gate in &gates {
            Self::_validate_gate_qubits(gate, self.num_qubits)?;
        }
        self.gates.extend(gates);
        Ok(())
    }

    /// Gets the number of qubits in the circuit.
    ///
    /// # Returns
    ///
    /// * `usize` - The number of qubits in the circuit.
    pub fn get_num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Gets the gates in the circuit.
    ///
    /// # Returns
    ///
    /// * `&Vec<Gate>` - A reference to the vector of gates in the circuit.
    pub fn get_gates(&self) -> &Vec<Gate> {
        &self.gates
    }

    /// Executes the circuit with the given initial state, and returns the final state.
    ///
    /// # Arguments
    ///
    /// * `initial_state` - The initial state of the qubits in the circuit.
    ///
    /// # Returns
    ///
    /// * `Result<State, Error>` - The final state of the qubits after executing the circuit.
    ///
    /// # Errors
    ///
    /// * Returns an error if the number of qubits in the initial state does not match the number of qubits in the circuit.
    /// * Returns an error if the circuit cannot be executed due to invalid gate operations.
    pub fn execute(&self, initial_state: &State) -> Result<State, Error> {
        if initial_state.num_qubits() != self.num_qubits {
            return Err(Error::InvalidNumberOfQubits(initial_state.num_qubits()));
        }

        let mut current_state = initial_state.clone();

        for gate in &self.gates {
            current_state = gate.apply(&current_state)?;
        }

        Ok(current_state)
    }

    /// Executes the circuit with the given initial state, and returns all the intermediate states and the final state.
    ///
    /// # Arguments
    ///
    /// * `initial_state` - The initial state of the qubits in the circuit.
    ///
    /// # Returns
    ///
    /// * `Result<Vec<State>, Error>` - A vector of intermediate states and the final state after executing the circuit.
    ///
    /// # Errors
    ///
    /// * Returns an error if the number of qubits in the initial state does not match the number of qubits in the circuit.
    /// * Returns an error if the circuit cannot be executed due to invalid gate operations.
    pub fn trace_execution(&self, initial_state: &State) -> Result<Vec<State>, Error> {
        if initial_state.num_qubits() != self.num_qubits {
            return Err(Error::InvalidNumberOfQubits(initial_state.num_qubits()));
        }

        let mut current_state = initial_state.clone();
        let mut states = vec![current_state.clone()];

        for gate in &self.gates {
            current_state = gate.apply(&current_state)?;
            states.push(current_state.clone());
        }

        Ok(states)
    }

    /// Converts a parametric circuit into a circuit with concrete gates.
    pub(crate) fn to_concrete_circuit(&self) -> Self {
        let concrete_gates = self.gates.iter().flat_map(|g| {
            match g {
                Gate::Parametric(p_gate, targets, controls) => {
                    p_gate.to_concrete_gates(targets, controls)
                }
                _ => vec![g.clone()],
            }
        }).collect();

        Circuit {
            gates: concrete_gates,
            num_qubits: self.num_qubits,
        }
    }

    /// Converts the circuit to its internal QASM circuit if the circuit is compilable, and return an error if it is not, or if the conversion fails.
    pub(crate) fn to_qasm_circuit(&self) -> Result<QasmCircuit, CompilerError> {
        let concrete_circuit = self.to_concrete_circuit();
        let compilable_circuit = CompilableCircuit::try_from(&concrete_circuit)?;
        let qasm_instructions =
            compilable_circuit
                .to_ir()
                .iter()
                .try_fold(Vec::new(), |mut acc, instr| {
                    acc.extend(instr.to_qasm()?);
                    Ok(acc)
                })?;

        Ok(QasmCircuit::new(
            qasm_instructions,
            compilable_circuit.num_qubits,
        ))
    }

    /// Converts the circuit to its OpenQASM 3.0 (Quantum Assembly 3.0) representation.
    pub fn to_qasm(&self, to_dir: Option<impl AsRef<Path>>) -> Result<String, CompilerError> {
        let qasm_circuit: QasmCircuit = self.to_qasm_circuit()?;
        let qasm_string = qasm_circuit.to_qasm_string();

        if let Some(path) = to_dir {
            let dir_path = path.as_ref();

            if !dir_path.is_dir() {
                return Err(CompilerError::IOError(format!(
                    "Provided path is not a directory: {}",
                    dir_path.display()
                )));
            }

            let output_path = dir_path.join("circuit.qasm");

            let mut file = fs::File::create(&output_path).map_err(|e| {
                CompilerError::IOError(format!(
                    "Error creating file '{}': {}",
                    output_path.display(),
                    e
                ))
            })?;

            file.write_all(qasm_string.as_bytes()).map_err(|e| {
                CompilerError::IOError(format!(
                    "Error writing to file '{}': {}",
                    output_path.display(),
                    e
                ))
            })?;
        }

        Ok(qasm_string)
    }
}

/// A builder for creating a quantum circuit.
///
/// # Fields
///
/// * `gates` - A vector of gates in the circuit.
///
/// * `num_qubits` - The number of qubits in the circuit.
pub struct CircuitBuilder {
    /// A vector of gates in the circuit builder.
    /// A temporary vector to hold gates before building the circuit.
    pub gates: Vec<Gate>,
    /// The number of qubits in the circuit builder.
    pub num_qubits: usize,
}

impl CircuitBuilder {
    /// Creates a new circuit builder with the specified number of qubits.
    ///
    /// # Arguments
    ///
    /// * `num_qubits` - The number of qubits in the circuit builder.
    ///
    /// # Returns
    ///
    /// * `CircuitBuilder` - A new instance of the CircuitBuilder struct.
    pub fn new(num_qubits: usize) -> Self {
        CircuitBuilder {
            gates: Vec::new(),
            num_qubits,
        }
    }

    /// Adds a gate to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `gate` - The gate to be added to the circuit builder.
    pub fn add_gate(&mut self, gate: Gate) -> &mut Self {
        self.gates.push(gate);
        self
    }

    /// Adds multiple gates to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `gates` - A vector of gates to be added to the circuit builder.
    pub fn add_gates(&mut self, gates: Vec<Gate>) -> &mut Self {
        self.gates.extend(gates);
        self
    }

    /// Builds the circuit from the gates in the circuit builder.
    /// The builder's internal gate list is not cleared, allowing the builder to be reused.
    /// If this is the final circuit, use `build_final` instead.
    ///
    /// # Returns
    ///
    /// * `Result<Circuit, Error>` - A new instance of the Circuit struct or an error if the circuit cannot be built.
    pub fn build(&mut self) -> Result<Circuit, Error> {
        let gates_cloned = self.gates.clone();
        Circuit::with_gates(gates_cloned, self.num_qubits)
    }

    /// Builds the circuit from the gates in the circuit builder.
    /// The builder's internal gate list is cleared, allowing the builder to be reused.
    /// If this is an intermediate circuit, use `build` instead to retain the gates for further modifications.
    ///
    /// # Returns
    ///
    /// * `Result<Circuit, Error>` - A new instance of the Circuit struct or an error if the circuit cannot be built.
    pub fn build_final(&mut self) -> Result<Circuit, Error> {
        let gates = std::mem::take(&mut self.gates);
        Circuit::with_gates(gates, self.num_qubits)
    }

    /// Builds a subroutine from the gates in the circuit builder.
    /// The builder's internal gate list is cleared, allowing the builder to be reused.
    ///
    /// # Returns
    ///
    /// * `Subroutine` - A new instance of the Subroutine struct.
    pub fn build_subroutine(&mut self) -> Subroutine {
        let gates = std::mem::take(&mut self.gates);
        Subroutine::with_gates(gates, self.num_qubits)
    }

    /// Adds a subroutine to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `subroutine` - The subroutine to be added to the circuit builder.
    pub fn add_subroutine(&mut self, subroutine: Subroutine) -> &mut Self {
        self.gates.extend(subroutine.gates);
        self
    }

    // -- SINGLE QUBIT GATES --

    /// Adds a Hadamard gate to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `qubit` - The index of the qubit to which the Hadamard gate will be applied.
    pub fn h_gate(&mut self, qubit: usize) -> &mut Self {
        let gate: Gate = Gate::h_gate(qubit);
        self.add_gate(gate);
        self
    }

    /// Adds multiple Hadamard gates to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `qubits` - A vector of indices of the qubits to which the Hadamard gates will be applied.
    pub fn h_gates(&mut self, qubits: Vec<usize>) -> &mut Self {
        let gates: Vec<Gate> = Gate::h_multi_gate(qubits);
        self.add_gates(gates);
        self
    }

    /// Adds controlled Hadamard gates to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `target_qubits` - A vector of indices of the target qubits.
    /// * `control_qubits` - A vector of indices of the control qubits.
    pub fn ch_gates(&mut self, target_qubits: Vec<usize>, control_qubits: Vec<usize>) -> &mut Self {
        let gates: Vec<Gate> = Gate::h_controlled_gates(target_qubits, control_qubits);
        self.add_gates(gates);
        self
    }

    /// Adds a Pauli-X gate to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `qubit` - The index of the qubit to which the Pauli-X gate will be applied.
    pub fn x_gate(&mut self, qubit: usize) -> &mut Self {
        let gate: Gate = Gate::x_gate(qubit);
        self.add_gate(gate);
        self
    }

    /// Adds multiple Pauli-X gates to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `qubits` - A vector of indices of the qubits to which the Pauli-X gates will be applied.
    pub fn x_gates(&mut self, qubits: Vec<usize>) -> &mut Self {
        let gates: Vec<Gate> = Gate::x_multi_gate(qubits);
        self.add_gates(gates);
        self
    }

    /// Adds controlled Pauli-X gates to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `target_qubits` - A vector of indices of the target qubits.
    /// * `control_qubits` - A vector of indices of the control qubits.
    pub fn cx_gates(&mut self, target_qubits: Vec<usize>, control_qubits: Vec<usize>) -> &mut Self {
        let gates: Vec<Gate> = Gate::x_controlled_gates(target_qubits, control_qubits);
        self.add_gates(gates);
        self
    }

    /// Adds a Pauli-Y gate to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `qubit` - The index of the qubit to which the Pauli-Y gate will be applied.
    pub fn y_gate(&mut self, qubit: usize) -> &mut Self {
        let gate: Gate = Gate::y_gate(qubit);
        self.add_gate(gate);
        self
    }

    /// Adds multiple Pauli-Y gates to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `qubits` - A vector of indices of the qubits to which the Pauli-Y gates will be applied.
    pub fn y_gates(&mut self, qubits: Vec<usize>) -> &mut Self {
        let gates: Vec<Gate> = Gate::y_multi_gate(qubits);
        self.add_gates(gates);
        self
    }

    /// Adds controlled Pauli-Y gates to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `target_qubits` - A vector of indices of the target qubits.
    /// * `control_qubits` - A vector of indices of the control qubits.
    pub fn cy_gates(&mut self, target_qubits: Vec<usize>, control_qubits: Vec<usize>) -> &mut Self {
        let gates: Vec<Gate> = Gate::y_controlled_gates(target_qubits, control_qubits);
        self.add_gates(gates);
        self
    }

    /// Adds a Pauli-Z gate to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `qubit` - The index of the qubit to which the Pauli-Z gate will be applied.
    pub fn z_gate(&mut self, qubit: usize) -> &mut Self {
        let gate: Gate = Gate::z_gate(qubit);
        self.add_gate(gate);
        self
    }

    /// Adds multiple Pauli-Z gates to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `qubits` - A vector of indices of the qubits to which the Pauli-Z gates will be applied.
    pub fn z_gates(&mut self, qubits: Vec<usize>) -> &mut Self {
        let gates: Vec<Gate> = Gate::z_multi_gate(qubits);
        self.add_gates(gates);
        self
    }

    /// Adds controlled Pauli-Z gates to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `target_qubits` - A vector of indices of the target qubits.
    /// * `control_qubits` - A vector of indices of the control qubits.
    pub fn cz_gates(&mut self, target_qubits: Vec<usize>, control_qubits: Vec<usize>) -> &mut Self {
        let gates: Vec<Gate> = Gate::z_controlled_gates(target_qubits, control_qubits);
        self.add_gates(gates);
        self
    }

    /// Adds an Identity gate to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `qubit` - The index of the qubit to which the Identity gate will be applied.
    pub fn id_gate(&mut self, qubit: usize) -> &mut Self {
        let gate: Gate = Gate::i_gate(qubit);
        self.add_gate(gate);
        self
    }

    /// Adds multiple Identity gates to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `qubits` - A vector of indices of the qubits to which the Identity gates will be applied.
    pub fn id_gates(&mut self, qubits: Vec<usize>) -> &mut Self {
        let gates: Vec<Gate> = Gate::i_multi_gate(qubits);
        self.add_gates(gates);
        self
    }

    /// Adds controlled Identity gates to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `target_qubits` - A vector of indices of the target qubits.
    /// * `control_qubits` - A vector of indices of the control qubits.
    pub fn ci_gates(&mut self, target_qubits: Vec<usize>, control_qubits: Vec<usize>) -> &mut Self {
        let gates: Vec<Gate> = Gate::i_controlled_gates(target_qubits, control_qubits);
        self.add_gates(gates);
        self
    }

    /// Adds a Phase S gate to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `qubit` - The index of the qubit to which the Phase S gate will be applied.
    pub fn s_gate(&mut self, qubit: usize) -> &mut Self {
        let gate: Gate = Gate::s_gate(qubit);
        self.add_gate(gate);
        self
    }

    /// Adds multiple Phase S gates to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `qubits` - A vector of indices of the qubits to which the Phase S gates will be applied.
    pub fn s_gates(&mut self, qubits: Vec<usize>) -> &mut Self {
        let gates: Vec<Gate> = Gate::s_multi_gate(qubits);
        self.add_gates(gates);
        self
    }

    /// Adds controlled Phase S gates to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `target_qubits` - A vector of indices of the target qubits.
    /// * `control_qubits` - A vector of indices of the control qubits.
    pub fn cs_gates(&mut self, target_qubits: Vec<usize>, control_qubits: Vec<usize>) -> &mut Self {
        let gates: Vec<Gate> = Gate::s_controlled_gates(target_qubits, control_qubits);
        self.add_gates(gates);
        self
    }

    /// Adds a Phase S Dagger gate to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `qubit` - The index of the qubit to which the Phase S Dagger gate will be applied.
    pub fn sdag_gate(&mut self, qubit: usize) -> &mut Self {
        let gate: Gate = Gate::s_dag_gate(qubit);
        self.add_gate(gate);
        self
    }

    /// Adds multiple Phase S Dagger gates to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `qubits` - A vector of indices of the qubits to which the Phase S Dagger gates will be applied.
    pub fn sdag_gates(&mut self, qubits: Vec<usize>) -> &mut Self {
        let gates: Vec<Gate> = Gate::s_dag_multi_gate(qubits);
        self.add_gates(gates);
        self
    }

    /// Adds controlled Phase S Dagger gates to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `target_qubits` - A vector of indices of the target qubits.
    /// * `control_qubits` - A vector of indices of the control qubits.
    pub fn csdag_gates(
        &mut self,
        target_qubits: Vec<usize>,
        control_qubits: Vec<usize>,
    ) -> &mut Self {
        let gates: Vec<Gate> = Gate::s_dag_controlled_gates(target_qubits, control_qubits);
        self.add_gates(gates);
        self
    }

    /// Adds a Phase T gate to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `qubit` - The index of the qubit to which the Phase T gate will be applied.
    pub fn t_gate(&mut self, qubit: usize) -> &mut Self {
        let gate: Gate = Gate::t_gate(qubit);
        self.add_gate(gate);
        self
    }

    /// Adds multiple Phase T gates to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `qubits` - A vector of indices of the qubits to which the Phase T gates will be applied.
    pub fn t_gates(&mut self, qubits: Vec<usize>) -> &mut Self {
        let gates: Vec<Gate> = Gate::t_multi_gate(qubits);
        self.add_gates(gates);
        self
    }

    /// Adds controlled Phase T gates to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `target_qubits` - A vector of indices of the target qubits.
    /// * `control_qubits` - A vector of indices of the control qubits.
    pub fn ct_gates(&mut self, target_qubits: Vec<usize>, control_qubits: Vec<usize>) -> &mut Self {
        let gates: Vec<Gate> = Gate::t_controlled_gates(target_qubits, control_qubits);
        self.add_gates(gates);
        self
    }

    /// Adds a Phase T Dagger gate to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `qubit` - The index of the qubit to which the Phase T Dagger gate will be applied.
    pub fn tdag_gate(&mut self, qubit: usize) -> &mut Self {
        let gate: Gate = Gate::t_dag_gate(qubit);
        self.add_gate(gate);
        self
    }

    /// Adds multiple Phase T Dagger gates to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `qubits` - A vector of indices of the qubits to which the Phase T Dagger gates will be applied.
    pub fn tdag_gates(&mut self, qubits: Vec<usize>) -> &mut Self {
        let gates: Vec<Gate> = Gate::t_dag_multi_gate(qubits);
        self.add_gates(gates);
        self
    }

    /// Adds controlled Phase T Dagger gates to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `target_qubits` - A vector of indices of the target qubits.
    /// * `control_qubits` - A vector of indices of the control qubits.
    pub fn ctdag_gates(
        &mut self,
        target_qubits: Vec<usize>,
        control_qubits: Vec<usize>,
    ) -> &mut Self {
        let gates: Vec<Gate> = Gate::t_dag_controlled_gates(target_qubits, control_qubits);
        self.add_gates(gates);
        self
    }

    /// Adds a Phase Shift gate to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `qubit` - The index of the qubit to which the Phase Shift gate will be applied.
    /// * `angle` - The phase shift angle in radians.
    pub fn p_gate(&mut self, qubit: usize, angle: f64) -> &mut Self {
        let gate: Gate = Gate::p_gate(qubit, angle);
        self.add_gate(gate);
        self
    }

    /// Adds multiple Phase Shift gates to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `qubits` - A vector of indices of the qubits to which the Phase Shift gates will be applied.
    /// * `angle` - The phase shift angle in radians for all gates.
    pub fn p_gates(&mut self, qubits: Vec<usize>, angle: f64) -> &mut Self {
        let gates: Vec<Gate> = Gate::p_multi_gate(qubits, angle);
        self.add_gates(gates);
        self
    }

    /// Adds controlled Phase Shift gates to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `target_qubits` - A vector of indices of the target qubits.
    /// * `control_qubits` - A vector of indices of the control qubits.
    /// * `angle` - The phase shift angle in radians for all gates.
    pub fn cp_gates(
        &mut self,
        target_qubits: Vec<usize>,
        control_qubits: Vec<usize>,
        angle: f64,
    ) -> &mut Self {
        let gates: Vec<Gate> = Gate::p_controlled_gates(target_qubits, control_qubits, angle);
        self.add_gates(gates);
        self
    }

    /// Adds a Rotate X gate to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `qubit` - The index of the qubit to which the Rotate X gate will be applied.
    /// * `angle` - The rotation angle in radians.
    pub fn rx_gate(&mut self, qubit: usize, angle: f64) -> &mut Self {
        let gate: Gate = Gate::rx_gate(qubit, angle);
        self.add_gate(gate);
        self
    }

    /// Adds multiple Rotate X gates to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `qubits` - A vector of indices of the qubits to which the Rotate X gates will be applied.
    /// * `angle` - The rotation angle in radians for all gates.
    pub fn rx_gates(&mut self, qubits: Vec<usize>, angle: f64) -> &mut Self {
        let gates: Vec<Gate> = Gate::rx_multi_gate(qubits, angle);
        self.add_gates(gates);
        self
    }

    /// Adds controlled Rotate X gates to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `target_qubits` - A vector of indices of the target qubits.
    /// * `control_qubits` - A vector of indices of the control qubits.
    /// * `angle` - The rotation angle in radians for all gates.
    pub fn crx_gates(
        &mut self,
        target_qubits: Vec<usize>,
        control_qubits: Vec<usize>,
        angle: f64,
    ) -> &mut Self {
        let gates: Vec<Gate> = Gate::rx_controlled_gates(target_qubits, control_qubits, angle);
        self.add_gates(gates);
        self
    }

    /// Adds a Rotate Y gate to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `qubit` - The index of the qubit to which the Rotate Y gate will be applied.
    /// * `angle` - The rotation angle in radians.
    pub fn ry_gate(&mut self, qubit: usize, angle: f64) -> &mut Self {
        let gate: Gate = Gate::ry_gate(qubit, angle);
        self.add_gate(gate);
        self
    }

    /// Adds multiple Rotate Y gates to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `qubits` - A vector of indices of the qubits to which the Rotate Y gates will be applied.
    /// * `angle` - The rotation angle in radians for all gates.
    pub fn ry_gates(&mut self, qubits: Vec<usize>, angle: f64) -> &mut Self {
        let gates: Vec<Gate> = Gate::ry_multi_gate(qubits, angle);
        self.add_gates(gates);
        self
    }

    /// Adds controlled Rotate Y gates to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `target_qubits` - A vector of indices of the target qubits.
    /// * `control_qubits` - A vector of indices of the control qubits.
    /// * `angle` - The rotation angle in radians for all gates.
    pub fn cry_gates(
        &mut self,
        target_qubits: Vec<usize>,
        control_qubits: Vec<usize>,
        angle: f64,
    ) -> &mut Self {
        let gates: Vec<Gate> = Gate::ry_controlled_gates(target_qubits, control_qubits, angle);
        self.add_gates(gates);
        self
    }

    /// Adds a Rotate Z gate to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `qubit` - The index of the qubit to which the Rotate Z gate will be applied.
    /// * `angle` - The rotation angle in radians.
    pub fn rz_gate(&mut self, qubit: usize, angle: f64) -> &mut Self {
        let gate: Gate = Gate::rz_gate(qubit, angle);
        self.add_gate(gate);
        self
    }

    /// Adds multiple Rotate Z gates to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `qubits` - A vector of indices of the qubits to which the Rotate Z gates will be applied.
    /// * `angle` - The rotation angle in radians for all gates.
    pub fn rz_gates(&mut self, qubits: Vec<usize>, angle: f64) -> &mut Self {
        let gates: Vec<Gate> = Gate::rz_multi_gate(qubits, angle);
        self.add_gates(gates);
        self
    }

    /// Adds controlled Rotate Z gates to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `target_qubits` - A vector of indices of the target qubits.
    /// * `control_qubits` - A vector of indices of the control qubits.
    /// * `angle` - The rotation angle in radians for all gates.
    pub fn crz_gates(
        &mut self,
        target_qubits: Vec<usize>,
        control_qubits: Vec<usize>,
        angle: f64,
    ) -> &mut Self {
        let gates: Vec<Gate> = Gate::rz_controlled_gates(target_qubits, control_qubits, angle);
        self.add_gates(gates);
        self
    }

    /// Adds an arbitrary unitary operator gate to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `qubit` - The index of the qubit to which the operator will be applied.
    ///
    /// * `unitary` - Matrix representing the unitary operator.
    ///
    /// # Warning
    ///
    /// This method is fallible due to the potential for invalid unitary matrices.
    /// If the unitary matrix is not valid, it will return an error.
    /// Therefore, the `Result` must be handled appropriately before chaining further operations.
    pub fn unitary_gate(
        &mut self,
        qubit: usize,
        unitary: [[Complex<f64>; 2]; 2],
    ) -> Result<&mut Self, Error> {
        let gate: Gate = Gate::unitary2_gate(qubit, unitary)?;
        self.add_gate(gate);
        Ok(self)
    }

    /// Adds multiple arbitrary unitary operator gates to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `qubits` - A vector of indices of the qubits to which the operator will be applied.
    ///
    /// * `unitary` - Matrix representing the unitary operator.
    ///
    /// # Warning
    ///
    /// This method is fallible due to the potential for invalid unitary matrices.
    /// If the unitary matrix is not valid, it will return an error.
    /// Therefore, the `Result` must be handled appropriately before chaining further operations.
    pub fn unitary_gates(
        &mut self,
        qubits: Vec<usize>,
        unitary: [[Complex<f64>; 2]; 2],
    ) -> Result<&mut Self, Error> {
        let gates: Vec<Gate> = Gate::unitary2_multi_gate(qubits, unitary)?;
        self.add_gates(gates);
        Ok(self)
    }

    /// Adds controlled arbitrary unitary operator gates to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `target_qubits` - A vector of indices of the target qubits.
    ///
    /// * `control_qubits` - A vector of indices of the control qubits.
    ///
    /// * `unitary` - Matrix representing the unitary operator.
    ///
    /// # Warning
    ///
    /// This method is fallible due to the potential for invalid unitary matrices.
    /// If the unitary matrix is not valid, it will return an error.
    /// Therefore, the `Result` must be handled appropriately before chaining further operations.
    pub fn cunitary_gates(
        &mut self,
        target_qubits: Vec<usize>,
        control_qubits: Vec<usize>,
        unitary: [[Complex<f64>; 2]; 2],
    ) -> Result<&mut Self, Error> {
        let gates = Gate::unitary2_controlled_gates(target_qubits, control_qubits, unitary)?;
        self.add_gates(gates);
        Ok(self)
    }

    /// Creates and adds a new Unitary2 operator gate from a rotation angle theta and phase shift angle phi.
    ///
    /// This operator can be decomposed into a rotation around the Y axis followed by a phase shift.
    /// The enclosed unitary matrix is guaranteed to be unitary.
    ///
    /// Special cases include:
    ///
    /// * U(theta, 0) = RY(theta)
    /// * U(0, phi) = PhaseShift(phi)
    /// * U(Pi/2, Pi) = Hadamard
    /// * U(Pi, Pi) = Pauli-X
    ///
    /// # Arguments
    ///
    /// * `qubit` - The index of the qubit to which the operator will be applied.
    ///
    /// * `theta` - The rotation angle in radians.
    ///
    /// * `phi` - The phase shift angle in radians.
    pub fn ry_phase_gate(&mut self, qubit: usize, theta: f64, phi: f64) -> &mut Self {
        let gate: Gate = Gate::ry_phase_gate(qubit, theta, phi);
        self.add_gate(gate);
        self
    }

    /// Creates and adds multiple new Unitary2 operator gates from a rotation angle theta and phase shift angle phi.
    ///
    /// This operator can be decomposed into a rotation around the Y axis followed by a phase shift.
    /// The enclosed unitary matrix is guaranteed to be unitary.
    ///
    /// Special cases include:
    /// * U(theta, 0) = RY(theta)
    /// * U(0, phi) = PhaseShift(phi)
    /// * U(Pi/2, Pi) = Hadamard
    /// * U(Pi, Pi) = Pauli-X
    ///
    /// # Arguments
    /// * `qubits` - A vector of indices of the qubits to which the operator will be applied.
    /// * `theta` - The rotation angle in radians for all gates.
    /// * `phi` - The phase shift angle in radians for all gates.
    pub fn ry_phase_gates(&mut self, qubits: Vec<usize>, theta: f64, phi: f64) -> &mut Self {
        let gates: Vec<Gate> = Gate::ry_phase_multi_gate(qubits, theta, phi);
        self.add_gates(gates);
        self
    }

    /// Creates and adds controlled Unitary2 operator gates from a rotation angle theta and phase shift angle phi.
    ///
    /// This operator can be decomposed into a rotation around the Y axis followed by a phase shift.
    /// The enclosed unitary matrix is guaranteed to be unitary.
    ///
    /// Special cases include:
    /// * U(theta, 0) = RY(theta)
    /// * U(0, phi) = PhaseShift(phi)
    /// * U(Pi/2, Pi) = Hadamard
    /// * U(Pi, Pi) = Pauli-X
    ///
    /// # Arguments
    ///
    /// * `target_qubits` - A vector of indices of the target qubits.
    ///
    /// * `control_qubits` - A vector of indices of the control qubits.
    ///
    /// * `theta` - The rotation angle in radians for all gates.
    ///
    /// * `phi` - The phase shift angle in radians for all gates.
    pub fn cry_phase_gates(
        &mut self,
        target_qubits: Vec<usize>,
        control_qubits: Vec<usize>,
        theta: f64,
        phi: f64,
    ) -> &mut Self {
        let gates: Vec<Gate> =
            Gate::ry_phase_controlled_gates(target_qubits, control_qubits, theta, phi);
        self.add_gates(gates);
        self
    }

    // -- MULTI-QUBIT GATES --

    /// Adds a CNOT gate to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `target_qubit` - The index of the target qubit.
    /// * `control_qubit` - The index of the control qubit.
    pub fn cnot_gate(&mut self, target_qubit: usize, control_qubit: usize) -> &mut Self {
        let gate: Gate = Gate::cnot_gate(target_qubit, control_qubit);
        self.add_gate(gate);
        self
    }

    /// Adds a SWAP gate to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `qubit1` - The index of the first qubit to swap.
    /// * `qubit2` - The index of the second qubit to swap.
    pub fn swap_gate(&mut self, qubit1: usize, qubit2: usize) -> &mut Self {
        let gate: Gate = Gate::swap_gate(qubit1, qubit2);
        self.add_gate(gate);
        self
    }

    /// Adds a controlled SWAP gate to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `target_qubit1` - The index of the first qubit to swap.
    /// * `target_qubit2` - The index of the second qubit to swap.
    /// * `control_qubits` - A vector of indices of the control qubits.
    pub fn cswap_gate(
        &mut self,
        target_qubit1: usize,
        target_qubit2: usize,
        control_qubits: Vec<usize>,
    ) -> &mut Self {
        let gate: Gate = Gate::Operator(
            Box::new(crate::components::operator::SWAP),
            vec![target_qubit1, target_qubit2],
            control_qubits,
        );
        self.add_gate(gate);
        self
    }

    /// Adds a Toffoli (CCNOT) gate to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `target_qubit` - The index of the target qubit.
    /// * `control_qubit1` - The index of the first control qubit.
    /// * `control_qubit2` - The index of the second control qubit.
    pub fn toffoli_gate(
        &mut self,
        control_qubit1: usize,
        control_qubit2: usize,
        target_qubit: usize,
    ) -> &mut Self {
        let gate: Gate = Gate::toffoli_gate(target_qubit, vec![control_qubit1, control_qubit2]);
        self.add_gate(gate);
        self
    }

    /// Adds a Matchgate to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `target_qubit` - The index of the first target qubit. The second target qubit is assumed to be the next qubit.
    /// * `theta` - The angle of rotation in radians.
    /// * `phi1` - The first phase shift in radians.
    /// * `phi2` - The second phase shift in radians.
    ///
    /// # Warning
    ///
    /// This gate is not yet compilable to OpenQASM, since it requires advanced decomposition techniques.
    pub fn matchgate(
        &mut self,
        target_qubit: usize,
        theta: f64,
        phi1: f64,
        phi2: f64,
    ) -> &mut Self {
        let gate: Gate = Gate::matchgate(target_qubit, theta, phi1, phi2);
        self.add_gate(gate);
        self
    }

    /// Adds a controlled Matchgate to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `target_qubit` - The index of the first target qubit. The second target qubit is assumed to be the next qubit.
    /// * `control_qubits` - A vector of indices of the control qubits.
    /// * `theta` - The angle of rotation in radians.
    /// * `phi1` - The first phase shift in radians.
    /// * `phi2` - The second phase shift in radians.
    /// 
    /// # Warning
    ///
    /// This gate is not yet compilable to OpenQASM, since it requires advanced decomposition techniques.
    pub fn cmatchgate(
        &mut self,
        target_qubit: usize,
        control_qubits: Vec<usize>,
        theta: f64,
        phi1: f64,
        phi2: f64,
    ) -> &mut Self {
        let gate: Gate =
            Gate::controlled_matchgate(target_qubit, control_qubits, theta, phi1, phi2);
        self.add_gate(gate);
        self
    }

    /// Adds a custom operator gate to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `operator` - The operator to be added to the circuit builder.
    /// * `target_qubits` - A vector of indices of the target qubits.
    /// * `control_qubits` - An vector of indices of the control qubits.
    pub fn add_operator_gate(
        &mut self,
        operator: Box<dyn Operator>,
        target_qubits: Vec<usize>,
        control_qubits: Vec<usize>,
    ) -> &mut Self {
        let gate: Gate = Gate::Operator(operator, target_qubits, control_qubits);
        self.add_gate(gate);
        self
    }

    /// Adds a measurement gate to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `basis` - The measurement basis (e.g., computational).
    ///
    /// * `qubits` - A vector of indices of the qubits to be measured.
    pub fn measure_gate(&mut self, basis: MeasurementBasis, qubits: Vec<usize>) -> &mut Self {
        let gate: Gate = Gate::Measurement(basis, qubits);
        self.add_gate(gate);
        self
    }
}
