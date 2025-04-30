use crate::{
    components::{gate::Gate, measurement::MeasurementBasis, operator::Operator, state::State},
    errors::Error,
};

use num_complex::Complex;

/// Represents a quantum circuit as a vector of gates.
///
/// # Fields
///
/// * `gates` - A vector of gates in the circuit.
/// * `num_qubits` - The number of qubits in the circuit.
pub struct Circuit {
    /// A vector of gates in the circuit.
    pub gates: Vec<Gate>,
    /// The number of qubits in the circuit.
    pub num_qubits: usize,
}

impl Circuit {
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
            // Check if the gates' target qubits are within the circuit's qubit range
            for &qubit in gate.get_target_qubits() {
                if qubit >= num_qubits {
                    return Err(Error::InvalidQubitIndex(qubit, num_qubits));
                }
            }

            // Check if the gate's control qubits are within the circuit's qubit range
            if let Some(control_qubits) = gate.get_control_qubits() {
                for &qubit in control_qubits {
                    if qubit >= num_qubits {
                        return Err(Error::InvalidQubitIndex(qubit, num_qubits));
                    }
                }
            }
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
        // Check if the gate's target qubits are within the circuit's qubit range
        for &qubit in gate.get_target_qubits() {
            if qubit >= self.num_qubits {
                return Err(Error::InvalidQubitIndex(qubit, self.num_qubits));
            }
        }

        // Check if the gate's control qubits are within the circuit's qubit range
        if let Some(control_qubits) = gate.get_control_qubits() {
            for &qubit in control_qubits {
                if qubit >= self.num_qubits {
                    return Err(Error::InvalidQubitIndex(qubit, self.num_qubits));
                }
            }
        }

        Ok(self.gates.push(gate))
    }

    /// Adds multiple gates to the circuit.
    ///
    /// # Arguments
    ///
    /// * `gates` - A vector of gates to be added to the circuit.
    pub fn add_gates(&mut self, gates: Vec<Gate>) {
        self.gates.extend(gates);
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
    pub fn add_gate(&mut self, gate: Gate) {
        self.gates.push(gate);
    }

    /// Adds multiple gates to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `gates` - A vector of gates to be added to the circuit builder.
    pub fn add_gates(&mut self, gates: Vec<Gate>) {
        self.gates.extend(gates);
    }

    /// Builds the circuit from the gates in the circuit builder.
    ///
    /// # Returns
    ///
    /// * `Result<Circuit, Error>` - A new instance of the Circuit struct or an error if the circuit cannot be built.
    pub fn build(self) -> Result<Circuit, Error> {
        Circuit::with_gates(self.gates, self.num_qubits)
    }

    /// Builds a subroutine from the gates in the circuit builder.
    ///
    /// # Returns
    ///
    /// * `Subroutine` - A new instance of the Subroutine struct.
    pub fn build_subroutine(self) -> Subroutine {
        Subroutine::with_gates(self.gates, self.num_qubits)
    }

    /// Adds a subroutine to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `subroutine` - The subroutine to be added to the circuit builder.
    pub fn add_subroutine(&mut self, subroutine: Subroutine) {
        self.gates.extend(subroutine.gates);
    }

    // -- SINGLE QUBIT GATES --

    /// Adds a Hadamard gate to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `qubit` - The index of the qubit to which the Hadamard gate will be applied.
    pub fn h_gate(&mut self, qubit: usize) {
        let gate: Gate = Gate::h_gate(qubit);
        self.add_gate(gate);
    }

    /// Adds multiple Hadamard gates to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `qubits` - A vector of indices of the qubits to which the Hadamard gates will be applied.
    pub fn h_gates(&mut self, qubits: Vec<usize>) {
        let gates: Vec<Gate> = Gate::h_multi_gate(qubits);
        self.add_gates(gates);
    }

    /// Adds controlled Hadamard gates to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `target_qubits` - A vector of indices of the target qubits.
    /// * `control_qubits` - A vector of indices of the control qubits.
    pub fn ch_gates(&mut self, target_qubits: Vec<usize>, control_qubits: Vec<usize>) {
        let gates: Vec<Gate> = Gate::h_controlled_gates(target_qubits, control_qubits);
        self.add_gates(gates);
    }

    /// Adds a Pauli-X gate to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `qubit` - The index of the qubit to which the Pauli-X gate will be applied.
    pub fn x_gate(&mut self, qubit: usize) {
        let gate: Gate = Gate::x_gate(qubit);
        self.add_gate(gate);
    }

    /// Adds multiple Pauli-X gates to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `qubits` - A vector of indices of the qubits to which the Pauli-X gates will be applied.
    pub fn x_gates(&mut self, qubits: Vec<usize>) {
        let gates: Vec<Gate> = Gate::x_multi_gate(qubits);
        self.add_gates(gates);
    }

    /// Adds controlled Pauli-X gates to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `target_qubits` - A vector of indices of the target qubits.
    /// * `control_qubits` - A vector of indices of the control qubits.
    pub fn cx_gates(&mut self, target_qubits: Vec<usize>, control_qubits: Vec<usize>) {
        let gates: Vec<Gate> = Gate::x_controlled_gates(target_qubits, control_qubits);
        self.add_gates(gates);
    }

    /// Adds a Pauli-Y gate to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `qubit` - The index of the qubit to which the Pauli-Y gate will be applied.
    pub fn y_gate(&mut self, qubit: usize) {
        let gate: Gate = Gate::y_gate(qubit);
        self.add_gate(gate);
    }

    /// Adds multiple Pauli-Y gates to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `qubits` - A vector of indices of the qubits to which the Pauli-Y gates will be applied.
    pub fn y_gates(&mut self, qubits: Vec<usize>) {
        let gates: Vec<Gate> = Gate::y_multi_gate(qubits);
        self.add_gates(gates);
    }

    /// Adds controlled Pauli-Y gates to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `target_qubits` - A vector of indices of the target qubits.
    /// * `control_qubits` - A vector of indices of the control qubits.
    pub fn cy_gates(&mut self, target_qubits: Vec<usize>, control_qubits: Vec<usize>) {
        let gates: Vec<Gate> = Gate::y_controlled_gates(target_qubits, control_qubits);
        self.add_gates(gates);
    }

    /// Adds a Pauli-Z gate to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `qubit` - The index of the qubit to which the Pauli-Z gate will be applied.
    pub fn z_gate(&mut self, qubit: usize) {
        let gate: Gate = Gate::z_gate(qubit);
        self.add_gate(gate);
    }

    /// Adds multiple Pauli-Z gates to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `qubits` - A vector of indices of the qubits to which the Pauli-Z gates will be applied.
    pub fn z_gates(&mut self, qubits: Vec<usize>) {
        let gates: Vec<Gate> = Gate::z_multi_gate(qubits);
        self.add_gates(gates);
    }

    /// Adds controlled Pauli-Z gates to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `target_qubits` - A vector of indices of the target qubits.
    /// * `control_qubits` - A vector of indices of the control qubits.
    pub fn cz_gates(&mut self, target_qubits: Vec<usize>, control_qubits: Vec<usize>) {
        let gates: Vec<Gate> = Gate::z_controlled_gates(target_qubits, control_qubits);
        self.add_gates(gates);
    }

    /// Adds an Identity gate to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `qubit` - The index of the qubit to which the Identity gate will be applied.
    pub fn id_gate(&mut self, qubit: usize) {
        let gate: Gate = Gate::i_gate(qubit);
        self.add_gate(gate);
    }

    /// Adds multiple Identity gates to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `qubits` - A vector of indices of the qubits to which the Identity gates will be applied.
    pub fn id_gates(&mut self, qubits: Vec<usize>) {
        let gates: Vec<Gate> = Gate::i_multi_gate(qubits);
        self.add_gates(gates);
    }

    /// Adds controlled Identity gates to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `target_qubits` - A vector of indices of the target qubits.
    /// * `control_qubits` - A vector of indices of the control qubits.
    pub fn ci_gates(&mut self, target_qubits: Vec<usize>, control_qubits: Vec<usize>) {
        let gates: Vec<Gate> = Gate::i_controlled_gates(target_qubits, control_qubits);
        self.add_gates(gates);
    }

    /// Adds a Phase S gate to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `qubit` - The index of the qubit to which the Phase S gate will be applied.
    pub fn s_gate(&mut self, qubit: usize) {
        let gate: Gate = Gate::s_gate(qubit);
        self.add_gate(gate);
    }

    /// Adds multiple Phase S gates to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `qubits` - A vector of indices of the qubits to which the Phase S gates will be applied.
    pub fn s_gates(&mut self, qubits: Vec<usize>) {
        let gates: Vec<Gate> = Gate::s_multi_gate(qubits);
        self.add_gates(gates);
    }

    /// Adds controlled Phase S gates to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `target_qubits` - A vector of indices of the target qubits.
    /// * `control_qubits` - A vector of indices of the control qubits.
    pub fn cs_gates(&mut self, target_qubits: Vec<usize>, control_qubits: Vec<usize>) {
        let gates: Vec<Gate> = Gate::s_controlled_gates(target_qubits, control_qubits);
        self.add_gates(gates);
    }

    /// Adds a Phase S Dagger gate to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `qubit` - The index of the qubit to which the Phase S Dagger gate will be applied.
    pub fn sdag_gate(&mut self, qubit: usize) {
        let gate: Gate = Gate::s_dag_gate(qubit);
        self.add_gate(gate);
    }

    /// Adds multiple Phase S Dagger gates to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `qubits` - A vector of indices of the qubits to which the Phase S Dagger gates will be applied.
    pub fn sdag_gates(&mut self, qubits: Vec<usize>) {
        let gates: Vec<Gate> = Gate::s_dag_multi_gate(qubits);
        self.add_gates(gates);
    }

    /// Adds controlled Phase S Dagger gates to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `target_qubits` - A vector of indices of the target qubits.
    /// * `control_qubits` - A vector of indices of the control qubits.
    pub fn csdag_gates(&mut self, target_qubits: Vec<usize>, control_qubits: Vec<usize>) {
        let gates: Vec<Gate> = Gate::s_dag_controlled_gates(target_qubits, control_qubits);
        self.add_gates(gates);
    }

    /// Adds a Phase T gate to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `qubit` - The index of the qubit to which the Phase T gate will be applied.
    pub fn t_gate(&mut self, qubit: usize) {
        let gate: Gate = Gate::t_gate(qubit);
        self.add_gate(gate);
    }

    /// Adds multiple Phase T gates to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `qubits` - A vector of indices of the qubits to which the Phase T gates will be applied.
    pub fn t_gates(&mut self, qubits: Vec<usize>) {
        let gates: Vec<Gate> = Gate::t_multi_gate(qubits);
        self.add_gates(gates);
    }

    /// Adds controlled Phase T gates to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `target_qubits` - A vector of indices of the target qubits.
    /// * `control_qubits` - A vector of indices of the control qubits.
    pub fn ct_gates(&mut self, target_qubits: Vec<usize>, control_qubits: Vec<usize>) {
        let gates: Vec<Gate> = Gate::t_controlled_gates(target_qubits, control_qubits);
        self.add_gates(gates);
    }

    /// Adds a Phase T Dagger gate to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `qubit` - The index of the qubit to which the Phase T Dagger gate will be applied.
    pub fn tdag_gate(&mut self, qubit: usize) {
        let gate: Gate = Gate::t_dag_gate(qubit);
        self.add_gate(gate);
    }

    /// Adds multiple Phase T Dagger gates to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `qubits` - A vector of indices of the qubits to which the Phase T Dagger gates will be applied.
    pub fn tdag_gates(&mut self, qubits: Vec<usize>) {
        let gates: Vec<Gate> = Gate::t_dag_multi_gate(qubits);
        self.add_gates(gates);
    }

    /// Adds controlled Phase T Dagger gates to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `target_qubits` - A vector of indices of the target qubits.
    /// * `control_qubits` - A vector of indices of the control qubits.
    pub fn ctdag_gates(&mut self, target_qubits: Vec<usize>, control_qubits: Vec<usize>) {
        let gates: Vec<Gate> = Gate::t_dag_controlled_gates(target_qubits, control_qubits);
        self.add_gates(gates);
    }

    /// Adds a Phase Shift gate to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `qubit` - The index of the qubit to which the Phase Shift gate will be applied.
    /// * `angle` - The phase shift angle in radians.
    pub fn p_gate(&mut self, qubit: usize, angle: f64) {
        let gate: Gate = Gate::p_gate(qubit, angle);
        self.add_gate(gate);
    }

    /// Adds multiple Phase Shift gates to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `qubits` - A vector of indices of the qubits to which the Phase Shift gates will be applied.
    /// * `angle` - The phase shift angle in radians for all gates.
    pub fn p_gates(&mut self, qubits: Vec<usize>, angle: f64) {
        let gates: Vec<Gate> = Gate::p_multi_gate(qubits, angle);
        self.add_gates(gates);
    }

    /// Adds controlled Phase Shift gates to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `target_qubits` - A vector of indices of the target qubits.
    /// * `control_qubits` - A vector of indices of the control qubits.
    /// * `angle` - The phase shift angle in radians for all gates.
    pub fn cp_gates(&mut self, target_qubits: Vec<usize>, control_qubits: Vec<usize>, angle: f64) {
        let gates: Vec<Gate> = Gate::p_controlled_gates(target_qubits, control_qubits, angle);
        self.add_gates(gates);
    }

    /// Adds a Rotate X gate to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `qubit` - The index of the qubit to which the Rotate X gate will be applied.
    /// * `angle` - The rotation angle in radians.
    pub fn rx_gate(&mut self, qubit: usize, angle: f64) {
        let gate: Gate = Gate::rx_gate(qubit, angle);
        self.add_gate(gate);
    }

    /// Adds multiple Rotate X gates to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `qubits` - A vector of indices of the qubits to which the Rotate X gates will be applied.
    /// * `angle` - The rotation angle in radians for all gates.
    pub fn rx_gates(&mut self, qubits: Vec<usize>, angle: f64) {
        let gates: Vec<Gate> = Gate::rx_multi_gate(qubits, angle);
        self.add_gates(gates);
    }

    /// Adds controlled Rotate X gates to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `target_qubits` - A vector of indices of the target qubits.
    /// * `control_qubits` - A vector of indices of the control qubits.
    /// * `angle` - The rotation angle in radians for all gates.
    pub fn crx_gates(&mut self, target_qubits: Vec<usize>, control_qubits: Vec<usize>, angle: f64) {
        let gates: Vec<Gate> = Gate::rx_controlled_gates(target_qubits, control_qubits, angle);
        self.add_gates(gates);
    }

    /// Adds a Rotate Y gate to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `qubit` - The index of the qubit to which the Rotate Y gate will be applied.
    /// * `angle` - The rotation angle in radians.
    pub fn ry_gate(&mut self, qubit: usize, angle: f64) {
        let gate: Gate = Gate::ry_gate(qubit, angle);
        self.add_gate(gate);
    }

    /// Adds multiple Rotate Y gates to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `qubits` - A vector of indices of the qubits to which the Rotate Y gates will be applied.
    /// * `angle` - The rotation angle in radians for all gates.
    pub fn ry_gates(&mut self, qubits: Vec<usize>, angle: f64) {
        let gates: Vec<Gate> = Gate::ry_multi_gate(qubits, angle);
        self.add_gates(gates);
    }

    /// Adds controlled Rotate Y gates to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `target_qubits` - A vector of indices of the target qubits.
    /// * `control_qubits` - A vector of indices of the control qubits.
    /// * `angle` - The rotation angle in radians for all gates.
    pub fn cry_gates(&mut self, target_qubits: Vec<usize>, control_qubits: Vec<usize>, angle: f64) {
        let gates: Vec<Gate> = Gate::ry_controlled_gates(target_qubits, control_qubits, angle);
        self.add_gates(gates);
    }

    /// Adds a Rotate Z gate to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `qubit` - The index of the qubit to which the Rotate Z gate will be applied.
    /// * `angle` - The rotation angle in radians.
    pub fn rz_gate(&mut self, qubit: usize, angle: f64) {
        let gate: Gate = Gate::rz_gate(qubit, angle);
        self.add_gate(gate);
    }

    /// Adds multiple Rotate Z gates to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `qubits` - A vector of indices of the qubits to which the Rotate Z gates will be applied.
    /// * `angle` - The rotation angle in radians for all gates.
    pub fn rz_gates(&mut self, qubits: Vec<usize>, angle: f64) {
        let gates: Vec<Gate> = Gate::rz_multi_gate(qubits, angle);
        self.add_gates(gates);
    }

    /// Adds controlled Rotate Z gates to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `target_qubits` - A vector of indices of the target qubits.
    /// * `control_qubits` - A vector of indices of the control qubits.
    /// * `angle` - The rotation angle in radians for all gates.
    pub fn crz_gates(&mut self, target_qubits: Vec<usize>, control_qubits: Vec<usize>, angle: f64) {
        let gates: Vec<Gate> = Gate::rz_controlled_gates(target_qubits, control_qubits, angle);
        self.add_gates(gates);
    }

    /// Adds an arbitrary unitary operator gate to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `qubit` - The index of the qubit to which the operator will be applied.
    ///
    /// * `unitary` - Matrix representing the unitary operator.
    pub fn unitary_gate(&mut self, qubit: usize, unitary: [[Complex<f64>; 2]; 2]) {
        let gate: Gate = Gate::unitary2_gate(qubit, unitary);
        self.add_gate(gate);
    }

    /// Adds multiple arbitrary unitary operator gates to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `qubits` - A vector of indices of the qubits to which the operator will be applied.
    ///
    /// * `unitary` - Matrix representing the unitary operator.
    pub fn unitary_gates(&mut self, qubits: Vec<usize>, unitary: [[Complex<f64>; 2]; 2]) {
        let gates: Vec<Gate> = Gate::unitary2_multi_gate(qubits, unitary);
        self.add_gates(gates);
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
    pub fn cunitary_gates(
        &mut self,
        target_qubits: Vec<usize>,
        control_qubits: Vec<usize>,
        unitary: [[Complex<f64>; 2]; 2],
    ) {
        let gates: Vec<Gate> =
            Gate::unitary2_controlled_gates(target_qubits, control_qubits, unitary);
        self.add_gates(gates);
    }

    // -- MULTI-QUBIT GATES --

    /// Adds a CNOT gate to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `target_qubit` - The index of the target qubit.
    /// * `control_qubit` - The index of the control qubit.
    pub fn cnot_gate(&mut self, target_qubit: usize, control_qubit: usize) {
        let gate: Gate = Gate::cnot_gate(target_qubit, control_qubit);
        self.add_gate(gate);
    }

    /// Adds a SWAP gate to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `qubit1` - The index of the first qubit to swap.
    /// * `qubit2` - The index of the second qubit to swap.
    pub fn swap_gate(&mut self, qubit1: usize, qubit2: usize) {
        let gate: Gate = Gate::swap_gate(qubit1, qubit2);
        self.add_gate(gate);
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
    ) {
        let gate: Gate = Gate::Operator(
            Box::new(crate::components::operator::SWAP),
            vec![target_qubit1, target_qubit2],
            control_qubits,
        );
        self.add_gate(gate);
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
    ) {
        let gate: Gate = Gate::toffoli_gate(target_qubit, vec![control_qubit1, control_qubit2]);
        self.add_gate(gate);
    }

    /// Adds a custom operator gate to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `operator` - The operator to be added to the circuit builder.
    /// * `target_qubits` - A vector of indices of the target qubits.
    /// * `control_qubits` - An vector of indices of the control qubits.
    ///
    /// # Returns
    pub fn add_operator_gate(
        &mut self,
        operator: Box<dyn Operator>,
        target_qubits: Vec<usize>,
        control_qubits: Vec<usize>,
    ) {
        let gate: Gate = Gate::Operator(operator, target_qubits, control_qubits);
        self.add_gate(gate);
    }

    /// Adds a measurement gate to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `basis` - The measurement basis (e.g., computational).
    ///
    /// * `qubits` - A vector of indices of the qubits to be measured.
    pub fn measure_gate(&mut self, basis: MeasurementBasis, qubits: Vec<usize>) {
        let gate: Gate = Gate::Measurement(basis, qubits);
        self.add_gate(gate);
    }
}

/// A subroutine for a quantum circuit.
///
/// # Fields
///
/// * `gates` - A vector of gates in the subroutine.
///
/// * `num_qubits` - The number of qubits in the subroutine.
pub struct Subroutine {
    /// The gates in the subroutine.
    pub gates: Vec<Gate>,
    /// The number of qubits in the subroutine.
    pub num_qubits: usize,
}

impl Subroutine {
    /// Creates a new subroutine with the specified number of qubits.
    ///
    /// # Arguments
    ///
    /// * `num_qubits` - The number of qubits in the subroutine.
    pub fn new(num_qubits: usize) -> Self {
        Subroutine {
            gates: Vec::new(),
            num_qubits,
        }
    }

    /// Creates a new subroutine with the specified gates and number of qubits.
    ///
    /// # Arguments
    ///
    /// * `gates` - A vector of gates in the subroutine.
    /// * `num_qubits` - The number of qubits in the subroutine.
    pub fn with_gates(gates: Vec<Gate>, num_qubits: usize) -> Subroutine {
        Subroutine { gates, num_qubits }
    }

    /// Gets the gates in the subroutine.
    ///
    /// # Returns
    ///
    /// * `&Vec<Gate>` - A reference to the vector of gates in the subroutine.
    pub fn get_gates(&self) -> &Vec<Gate> {
        &self.gates
    }

    /// Adds a gate to the subroutine.
    ///
    /// # Arguments
    ///
    /// * `gate` - The gate to be added to the subroutine.
    pub fn add_gate(&mut self, gate: Gate) {
        self.gates.push(gate);
    }

    /// Adds multiple gates to the subroutine.
    ///
    /// # Arguments
    ///
    /// * `gates` - A vector of gates to be added to the subroutine.
    pub fn add_gates(&mut self, gates: Vec<Gate>) {
        for gate in gates {
            self.add_gate(gate);
        }
    }

    /// Gets the number of qubits in the subroutine.
    ///
    /// # Returns
    ///
    /// * `usize` - The number of qubits in the subroutine.
    pub fn get_num_qubits(&self) -> usize {
        self.num_qubits
    }

    // -- COMMON SUBROUTINES --

    /// Creates a quantum fourier transform subroutine for the specified qubits.
    ///
    /// # Arguments
    ///
    /// * `qubits` - A vector of indices of the qubits to be transformed.
    ///
    /// * `num_qubits` - The number of qubits in the subroutine.
    pub fn qft(qubits: Vec<usize>, num_qubits: usize) -> Subroutine {
        let mut builder: CircuitBuilder = CircuitBuilder::new(num_qubits);
        let n: usize = qubits.len();
        for i in 0..n {
            // Apply Hadamard gate
            builder.h_gate(qubits[i]);
            // Apply controlled phase rotations
            for j in (i + 1)..n {
                // The angle for the controlled phase gate R_k is pi / 2^(j-i)
                let angle = std::f64::consts::PI / (2_f64.powi((j - i) as i32));
                // Control qubit is j, target qubit is i
                builder.cp_gates(vec![qubits[i]], vec![qubits[j]], angle);
            }
        }
        // Swap qubits at the end
        for i in 0..(n / 2) {
            builder.swap_gate(qubits[i], qubits[n - 1 - i]);
        }
        builder.build_subroutine()
    }

    /// Creates a quantum inverse fourier transform subroutine for the specified qubits.
    /// 
    /// # Arguments
    /// 
    /// * `qubits` - A vector of indices of the qubits to be transformed.
    /// 
    /// * `num_qubits` - The number of qubits in the subroutine.
    /// 
    /// # Returns
    /// 
    /// * `Subroutine` - A new instance of the Subroutine struct.
    pub fn iqft(qubits: Vec<usize>, num_qubits: usize) -> Subroutine {
        let mut builder: CircuitBuilder = CircuitBuilder::new(num_qubits);
        let n: usize = qubits.len();

        // Apply swaps first
        for i in 0..(n / 2) {
            builder.swap_gate(qubits[i], qubits[n - 1 - i]);
        }

        // Apply inverse controlled rotations and Hadamards
        for i in (0..n).rev() {
            // Apply controlled phase rotations (inverse)
            for j in ((i + 1)..n).rev() {
                // The angle for the inverse controlled phase gate R_k dagger is -pi / 2^(j-i)
                let angle: f64 = -std::f64::consts::PI / (2_f64.powi((j - i) as i32));
                // Control qubit is j, target qubit is i
                builder.cp_gates(vec![qubits[i]], vec![qubits[j]], angle);
            }
            // Apply Hadamard gate
            builder.h_gate(qubits[i]);
        }
        builder.build_subroutine()
    }
}

// Allow conversion from Subroutine to Circuit
impl TryFrom<Subroutine> for Circuit {
    fn try_from(value: Subroutine) -> Result<Self, Self::Error> {
        let mut circuit = Circuit::new(value.num_qubits);
        for gate in value.gates {
            circuit.add_gate(gate)?;
        }
        Ok(circuit)
    }

    type Error = Error;
}
