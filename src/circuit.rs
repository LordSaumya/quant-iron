use crate::{
    components::{
        gate::Gate,
        operator::{
            CNOT, Hadamard, Identity, Operator, Pauli, PhaseS, PhaseSdag, PhaseShift, PhaseT,
            PhaseTdag, RotateX, RotateY, RotateZ, SWAP, Toffoli,
        },
        state::State,
        measurement::MeasurementBasis,
    },
    errors::Error,
};

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

    // -- SINGLE QUBIT GATES --

    /// Adds a Hadamard gate to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `qubit` - The index of the qubit to which the Hadamard gate will be applied.
    pub fn h_gate(&mut self, qubit: usize) {
        let gate: Gate = Gate::Operator(Box::new(Hadamard), vec![qubit], None);
        self.add_gate(gate);
    }

    /// Adds multiple Hadamard gates to the circuit builder.
    /// 
    /// # Arguments
    /// 
    /// * `qubits` - A vector of indices of the qubits to which the Hadamard gates will be applied.
    pub fn h_gates(&mut self, qubits: Vec<usize>) {
        for qubit in qubits {
            self.h_gate(qubit);
        }
    }

    /// Adds a Pauli-X gate to the circuit builder.
    /// 
    /// # Arguments
    /// 
    /// * `qubit` - The index of the qubit to which the Pauli-X gate will be applied.
    pub fn x_gate(&mut self, qubit: usize) {
        let gate: Gate = Gate::Operator(Box::new(Pauli::X), vec![qubit], None);
        self.add_gate(gate);
    }

    /// Adds multiple Pauli-X gates to the circuit builder.
    /// 
    /// # Arguments
    /// 
    /// * `qubits` - A vector of indices of the qubits to which the Pauli-X gates will be applied.
    pub fn x_gates(&mut self, qubits: Vec<usize>) {
        for qubit in qubits {
            self.x_gate(qubit);
        }
    }

    /// Adds a Pauli-Y gate to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `qubit` - The index of the qubit to which the Pauli-Y gate will be applied.
    pub fn y_gate(&mut self, qubit: usize) {
        let gate: Gate = Gate::Operator(Box::new(Pauli::Y), vec![qubit], None);
        self.add_gate(gate);
    }

    /// Adds multiple Pauli-Y gates to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `qubits` - A vector of indices of the qubits to which the Pauli-Y gates will be applied.
    pub fn y_gates(&mut self, qubits: Vec<usize>) {
        for qubit in qubits {
            self.y_gate(qubit);
        }
    }

    /// Adds a Pauli-Z gate to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `qubit` - The index of the qubit to which the Pauli-Z gate will be applied.
    pub fn z_gate(&mut self, qubit: usize) {
        let gate: Gate = Gate::Operator(Box::new(Pauli::Z), vec![qubit], None);
        self.add_gate(gate);
    }

    /// Adds multiple Pauli-Z gates to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `qubits` - A vector of indices of the qubits to which the Pauli-Z gates will be applied.
    pub fn z_gates(&mut self, qubits: Vec<usize>) {
        for qubit in qubits {
            self.z_gate(qubit);
        }
    }

    /// Adds an Identity gate to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `qubit` - The index of the qubit to which the Identity gate will be applied.
    pub fn id_gate(&mut self, qubit: usize) {
        let gate: Gate = Gate::Operator(Box::new(Identity), vec![qubit], None);
        self.add_gate(gate);
    }

    /// Adds multiple Identity gates to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `qubits` - A vector of indices of the qubits to which the Identity gates will be applied.
    pub fn id_gates(&mut self, qubits: Vec<usize>) {
        for qubit in qubits {
            self.id_gate(qubit);
        }
    }

    /// Adds a Phase S gate to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `qubit` - The index of the qubit to which the Phase S gate will be applied.
    pub fn s_gate(&mut self, qubit: usize) {
        let gate: Gate = Gate::Operator(Box::new(PhaseS), vec![qubit], None);
        self.add_gate(gate);
    }

    /// Adds multiple Phase S gates to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `qubits` - A vector of indices of the qubits to which the Phase S gates will be applied.
    pub fn s_gates(&mut self, qubits: Vec<usize>) {
        for qubit in qubits {
            self.s_gate(qubit);
        }
    }

    /// Adds a Phase S Dagger gate to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `qubit` - The index of the qubit to which the Phase S Dagger gate will be applied.
    pub fn sdag_gate(&mut self, qubit: usize) {
        let gate: Gate = Gate::Operator(Box::new(PhaseSdag), vec![qubit], None);
        self.add_gate(gate);
    }

    /// Adds multiple Phase S Dagger gates to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `qubits` - A vector of indices of the qubits to which the Phase S Dagger gates will be applied.
    pub fn sdag_gates(&mut self, qubits: Vec<usize>) {
        for qubit in qubits {
            self.sdag_gate(qubit);
        }
    }

    /// Adds a Phase T gate to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `qubit` - The index of the qubit to which the Phase T gate will be applied.
    pub fn t_gate(&mut self, qubit: usize) {
        let gate: Gate = Gate::Operator(Box::new(PhaseT), vec![qubit], None);
        self.add_gate(gate);
    }

    /// Adds multiple Phase T gates to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `qubits` - A vector of indices of the qubits to which the Phase T gates will be applied.
    pub fn t_gates(&mut self, qubits: Vec<usize>) {
        for qubit in qubits {
            self.t_gate(qubit);
        }
    }

    /// Adds a Phase T Dagger gate to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `qubit` - The index of the qubit to which the Phase T Dagger gate will be applied.
    pub fn tdag_gate(&mut self, qubit: usize) {
        let gate: Gate = Gate::Operator(Box::new(PhaseTdag), vec![qubit], None);
        self.add_gate(gate);
    }

    /// Adds multiple Phase T Dagger gates to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `qubits` - A vector of indices of the qubits to which the Phase T Dagger gates will be applied.
    pub fn tdag_gates(&mut self, qubits: Vec<usize>) {
        for qubit in qubits {
            self.tdag_gate(qubit);
        }
    }

    /// Adds a Phase Shift gate to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `qubit` - The index of the qubit to which the Phase Shift gate will be applied.
    /// * `angle` - The phase shift angle in radians.
    pub fn phase_shift_gate(&mut self, qubit: usize, angle: f64) {
        let gate: Gate = Gate::Operator(Box::new(PhaseShift::new(angle)), vec![qubit], None);
        self.add_gate(gate);
    }

    /// Adds multiple Phase Shift gates to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `qubits` - A vector of indices of the qubits to which the Phase Shift gates will be applied.
    /// * `angle` - The phase shift angle in radians for all gates.
    pub fn phase_shift_gates(&mut self, qubits: Vec<usize>, angle: f64) {
        for qubit in qubits {
            self.phase_shift_gate(qubit, angle);
        }
    }

    /// Adds a Rotate X gate to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `qubit` - The index of the qubit to which the Rotate X gate will be applied.
    /// * `angle` - The rotation angle in radians.
    pub fn rx_gate(&mut self, qubit: usize, angle: f64) {
        let gate: Gate = Gate::Operator(Box::new(RotateX::new(angle)), vec![qubit], None);
        self.add_gate(gate);
    }

    /// Adds multiple Rotate X gates to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `qubits` - A vector of indices of the qubits to which the Rotate X gates will be applied.
    /// * `angle` - The rotation angle in radians for all gates.
    pub fn rx_gates(&mut self, qubits: Vec<usize>, angle: f64) {
        for qubit in qubits {
            self.rx_gate(qubit, angle);
        }
    }

    /// Adds a Rotate Y gate to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `qubit` - The index of the qubit to which the Rotate Y gate will be applied.
    /// * `angle` - The rotation angle in radians.
    pub fn ry_gate(&mut self, qubit: usize, angle: f64) {
        let gate: Gate = Gate::Operator(Box::new(RotateY::new(angle)), vec![qubit], None);
        self.add_gate(gate);
    }

    /// Adds multiple Rotate Y gates to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `qubits` - A vector of indices of the qubits to which the Rotate Y gates will be applied.
    /// * `angle` - The rotation angle in radians for all gates.
    pub fn ry_gates(&mut self, qubits: Vec<usize>, angle: f64) {
        for qubit in qubits {
            self.ry_gate(qubit, angle);
        }
    }

    /// Adds a Rotate Z gate to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `qubit` - The index of the qubit to which the Rotate Z gate will be applied.
    /// * `angle` - The rotation angle in radians.
    pub fn rz_gate(&mut self, qubit: usize, angle: f64) {
        let gate: Gate = Gate::Operator(Box::new(RotateZ::new(angle)), vec![qubit], None);
        self.add_gate(gate);
    }

    /// Adds multiple Rotate Z gates to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `qubits` - A vector of indices of the qubits to which the Rotate Z gates will be applied.
    /// * `angle` - The rotation angle in radians for all gates.
    pub fn rz_gates(&mut self, qubits: Vec<usize>, angle: f64) {
        for qubit in qubits {
            self.rz_gate(qubit, angle);
        }
    }

    // -- MULTI-QUBIT GATES --

    /// Adds a CNOT gate to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `control_qubit` - The index of the control qubit.
    /// * `target_qubit` - The index of the target qubit.
    pub fn cnot_gate(&mut self, control_qubit: usize, target_qubit: usize) {
        let gate: Gate = Gate::Operator(
            Box::new(CNOT),
            vec![target_qubit],
            Some(vec![control_qubit]),
        );
        self.add_gate(gate);
    }

    /// Adds a SWAP gate to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `qubit1` - The index of the first qubit to swap.
    /// * `qubit2` - The index of the second qubit to swap.
    pub fn swap_gate(&mut self, qubit1: usize, qubit2: usize) {
        let gate: Gate = Gate::Operator(Box::new(SWAP), vec![qubit1, qubit2], None);
        self.add_gate(gate);
    }

    /// Adds a Toffoli (CCNOT) gate to the circuit builder.
    ///
    /// # Arguments
    ///
    /// * `control_qubit1` - The index of the first control qubit.
    /// * `control_qubit2` - The index of the second control qubit.
    /// * `target_qubit` - The index of the target qubit.
    pub fn toffoli_gate(
        &mut self,
        control_qubit1: usize,
        control_qubit2: usize,
        target_qubit: usize,
    ) {
        let gate: Gate = Gate::Operator(
            Box::new(Toffoli),
            vec![target_qubit],
            Some(vec![control_qubit1, control_qubit2]),
        );
        self.add_gate(gate);
    }

    /// Adds a custom operator gate to the circuit builder.
    /// 
    /// # Arguments
    /// 
    /// * `operator` - The operator to be added to the circuit builder.
    /// * `target_qubits` - A vector of indices of the target qubits.
    /// * `control_qubits` - An optional vector of indices of the control qubits.
    /// 
    /// # Returns
    pub fn add_operator_gate(
        &mut self,
        operator: Box<dyn Operator>,
        target_qubits: Vec<usize>,
        control_qubits: Option<Vec<usize>>,
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
    pub fn measure_gate(
        &mut self,
        basis: MeasurementBasis,
        qubits: Vec<usize>,
    ) {
        let gate: Gate = Gate::Measurement(basis, qubits);
        self.add_gate(gate);
    }
}
