use crate::{
    components::{
        measurement::MeasurementBasis,
        operator::{
            CNOT, Hadamard, Identity, Operator, Pauli, PhaseS, PhaseSdag, PhaseShift, PhaseT,
            PhaseTdag, RotateX, RotateY, RotateZ, SWAP, Toffoli, Unitary2, Matchgate
        },
        state::State,
        parametric::parametric_gate::ParametricGate,
    },
    errors::Error,
};
use num_complex::Complex;

/// Represents a quantum gate as part of a quantum circuit.
#[derive(Debug, Clone)]
pub enum Gate {
    /// Represents an operator gate.
    ///
    /// # Fields
    ///
    /// * `operator` - A boxed dynamic operator trait object.
    ///
    /// * `target_indices` - The indices of the qubits on which the operator acts.
    ///
    /// * `control_indices` - Optional control qubit indices for controlled gates. If empty, the gate is applied unconditionally.
    Operator(Box<dyn Operator>, Vec<usize>, Vec<usize>),

    /// Represents a measurement gate.
    ///
    /// # Fields
    ///
    /// * `MeasurementBasis` - The basis of measurement (e.g., computational basis).
    /// * `indices` - The indices of the measured qubits.
    Measurement(MeasurementBasis, Vec<usize>),

    /// Represents a parametric gate.
    Parametric(Box<dyn ParametricGate>, Vec<usize>, Vec<usize>),
}

impl Gate {
    /// Creates a new measurement gate for the specified qubit indices.
    ///
    /// # Arguments
    ///
    /// * `qubit_indices` - The indices of the qubits to be measured.
    /// * `basis` - The basis of measurement (e.g., computational basis).
    ///
    /// # Returns
    ///
    /// * `Gate` - A new instance of the Gate struct representing a measurement gate.
    pub fn new_measurement(qubit_indices: Vec<usize>, basis: MeasurementBasis) -> Self {
        Gate::Measurement(basis, qubit_indices)
    }

    /// Creates a new operator gate for the specified qubit indices.
    ///
    /// # Arguments
    ///
    /// * `operator` - A boxed dynamic operator trait object.
    ///
    /// * `target_indices` - The indices of the qubits on which the operator acts.
    ///
    /// * `control_indices` - Optional control qubit indices for controlled gates.
    ///
    /// # Returns
    ///
    /// * `Gate` - A new instance of the Gate struct representing an operator gate.
    pub fn new_operator(
        operator: Box<dyn Operator>,
        target_indices: Vec<usize>,
        control_indices: Vec<usize>,
    ) -> Self {
        Gate::Operator(operator, target_indices, control_indices)
    }

    /// Applies the gate to the given state and returns the new state.
    ///
    /// # Arguments
    ///
    /// * `state` - The quantum state to which the gate will be applied.
    ///
    /// # Returns
    ///
    /// * `State` - The new quantum state after applying the gate.
    pub fn apply(&self, state: &State) -> Result<State, Error> {
        match self {
            Gate::Operator(operator, target_indices, control_indices) => {
                operator.apply(state, target_indices, control_indices.as_slice())
            }
            Gate::Measurement(basis, indices) => state
                .measure(*basis, indices.as_slice())
                .map(|measurementresult| measurementresult.get_new_state().clone()),
            Gate::Parametric(p_gate, target_indices, control_indices) => {
                let concrete_gates = p_gate.to_concrete_gates(target_indices, control_indices);
                concrete_gates
                    .into_iter()
                    .try_fold(state.clone(), |current_state, gate| {
                        gate.apply(&current_state)
                    })
            }
        }
    }

    /// Returns the indices of the qubits on which the gate acts.
    ///
    /// # Returns
    ///
    /// * `&Vec<usize>` - A vector of indices of the qubits on which the gate acts.
    pub fn get_target_qubits(&self) -> &Vec<usize> {
        match self {
            Gate::Operator(_, target_indices, _) => target_indices,
            Gate::Measurement(_, indices) => indices,
            Gate::Parametric(_, target_indices, _) => target_indices,
        }
    }

    /// Returns the control indices of the gate if it has any.
    ///
    /// # Returns
    ///
    /// * `Option<&Vec<usize>>` - An optional vector of control indices.
    pub fn get_control_qubits(&self) -> Option<&Vec<usize>> {
        match self {
            Gate::Operator(_, _, control_indices) => Some(control_indices),
            Gate::Measurement(_, _) => None,
            Gate::Parametric(_, _, control_indices) => Some(control_indices),
        }
    }

    // -- SINGLE-QUBIT GATES --

    /// Creates a new Hadamard gate for the specified qubit index.
    ///
    /// # Arguments
    ///
    /// * `qubit_index` - The index of the qubit on which the Hadamard gate acts.
    ///
    /// # Returns
    ///
    /// * `Gate` - A new instance of the Gate struct representing a Hadamard gate.
    pub fn h_gate(qubit_index: usize) -> Self {
        Gate::Operator(Box::new(Hadamard), vec![qubit_index], vec![])
    }

    /// Creates new Hadamard gates for the specified qubit indices.
    ///
    /// # Arguments
    ///
    /// * `qubit_indices` - The indices of the qubits on which the Hadamard gate acts.
    ///
    /// # Returns
    ///
    /// * `Gates` - A vector of Gate structs representing Hadamard gates for each qubit index.
    pub fn h_multi_gate(qubit_indices: Vec<usize>) -> Vec<Self> {
        qubit_indices
            .into_iter()
            .map(Gate::h_gate)
            .collect()
    }

    /// Creates new controlled Hadamard gates for the specified qubit indices.
    /// 
    /// # Arguments
    /// 
    /// * `target_indices` - The indices of the target qubits.
    /// * `control_indices` - The indices of the control qubits.
    /// 
    /// # Returns
    /// 
    /// * `Gates` - A vector of Gate structs representing controlled Hadamard gates for each target qubit index.
    pub fn h_controlled_gates(
        target_indices: Vec<usize>,
        control_indices: Vec<usize>,
    ) -> Vec<Self> {
        target_indices
            .into_iter()
            .map(|target_index| Gate::Operator(
                Box::new(Hadamard),
                vec![target_index],
                control_indices.clone(),
            ))
            .collect()
    }

    /// Creates a new Pauli-X gate for the specified qubit index.
    ///
    /// # Arguments
    ///
    /// * `qubit_index` - The index of the qubit on which the Pauli-X gate acts.
    ///
    /// # Returns
    ///
    /// * `Gate` - A new instance of the Gate struct representing a Pauli-X gate.
    pub fn x_gate(qubit_index: usize) -> Self {
        Gate::Operator(Box::new(Pauli::X), vec![qubit_index], vec![])
    }

    /// Creates new controlled Pauli-X gates for the specified qubit indices.
    ///
    /// # Arguments
    ///
    /// * `qubit_indices` - The indices of the qubits on which the Pauli-X gates act.
    ///
    /// # Returns
    ///
    /// * `Vec<Gate>` - A vector of Gate structs representing Pauli-X gates for each qubit index.
    pub fn x_multi_gate(qubit_indices: Vec<usize>) -> Vec<Self> {
        qubit_indices
            .into_iter()
            .map(Gate::x_gate)
            .collect()
    }

    /// Creates new Pauli-X gates for the specified qubit indices with control qubits.
    /// 
    /// # Arguments
    /// 
    /// * `target_indices` - The indices of the target qubits.
    /// 
    /// * `control_indices` - The indices of the control qubits.
    /// 
    /// # Returns
    /// 
    /// * `Gates` - A vector of Gate structs representing controlled Pauli-X gates for each target qubit index.
    pub fn x_controlled_gates(
        target_indices: Vec<usize>,
        control_indices: Vec<usize>,
    ) -> Vec<Self> {
        target_indices
            .into_iter()
            .map(|target_index| Gate::Operator(
                Box::new(Pauli::X),
                vec![target_index],
                control_indices.clone(),
            ))
            .collect()
    }

    /// Creates a new Pauli-Y gate for the specified qubit index.
    ///
    /// # Arguments
    ///
    /// * `qubit_index` - The index of the qubit on which the Pauli-Y gate acts.
    ///
    /// # Returns
    ///
    /// * `Gate` - A new instance of the Gate struct representing a Pauli-Y gate.
    pub fn y_gate(qubit_index: usize) -> Self {
        Gate::Operator(Box::new(Pauli::Y), vec![qubit_index], vec![])
    }

    /// Creates new Pauli-Y gates for the specified qubit indices.
    ///
    /// # Arguments
    ///
    /// * `qubit_indices` - The indices of the qubits on which the Pauli-Y gates act.
    ///
    /// # Returns
    ///
    /// * `Vec<Gate>` - A vector of Gate structs representing Pauli-Y gates for each qubit index.
    pub fn y_multi_gate(qubit_indices: Vec<usize>) -> Vec<Self> {
        qubit_indices
            .into_iter()
            .map(Gate::y_gate)
            .collect()
    }

    /// Creates new controlled Pauli-Y gates for the specified qubit indices.
    /// 
    /// # Arguments
    /// 
    /// * `target_indices` - The indices of the target qubits.
    /// 
    /// * `control_indices` - The indices of the control qubits.
    /// 
    /// # Returns
    /// 
    /// * `Gates` - A vector of Gate structs representing controlled Pauli-Y gates for each target qubit index.
    pub fn y_controlled_gates(
        target_indices: Vec<usize>,
        control_indices: Vec<usize>,
    ) -> Vec<Self> {
        target_indices
            .into_iter()
            .map(|target_index| Gate::Operator(
                Box::new(Pauli::Y),
                vec![target_index],
                control_indices.clone(),
            ))
            .collect()
    }

    /// Creates a new Pauli-Z gate for the specified qubit index.
    ///
    /// # Arguments
    ///
    /// * `qubit_index` - The index of the qubit on which the Pauli-Z gate acts.
    ///
    /// # Returns
    ///
    /// * `Gate` - A new instance of the Gate struct representing a Pauli-Z gate.
    pub fn z_gate(qubit_index: usize) -> Self {
        Gate::Operator(Box::new(Pauli::Z), vec![qubit_index], vec![])
    }

    /// Creates new Pauli-Z gates for the specified qubit indices.
    ///
    /// # Arguments
    ///
    /// * `qubit_indices` - The indices of the qubits on which the Pauli-Z gates act.
    ///
    /// # Returns
    ///
    /// * `Vec<Gate>` - A vector of Gate structs representing Pauli-Z gates for each qubit index.
    pub fn z_multi_gate(qubit_indices: Vec<usize>) -> Vec<Self> {
        qubit_indices
            .into_iter()
            .map(Gate::z_gate)
            .collect()
    }

    /// Creates new controlled Pauli-Z gates for the specified qubit indices.
    /// 
    /// # Arguments
    /// 
    /// * `target_indices` - The indices of the target qubits.
    /// 
    /// * `control_indices` - The indices of the control qubits.
    /// 
    /// # Returns
    /// 
    /// * `Gates` - A vector of Gate structs representing controlled Pauli-Z gates for each target qubit index.
    pub fn z_controlled_gates(
        target_indices: Vec<usize>,
        control_indices: Vec<usize>,
    ) -> Vec<Self> {
        target_indices
            .into_iter()
            .map(|target_index| Gate::Operator(
                Box::new(Pauli::Z),
                vec![target_index],
                control_indices.clone(),
            ))
            .collect()
    }

    /// Creates a new Identity gate for the specified qubit index.
    ///
    /// # Arguments
    ///
    /// * `qubit_index` - The index of the qubit on which the Identity gate acts.
    ///
    /// # Returns
    ///
    /// * `Gate` - A new instance of the Gate struct representing an Identity gate.
    pub fn i_gate(qubit_index: usize) -> Self {
        Gate::Operator(Box::new(Identity), vec![qubit_index], vec![])
    }

    /// Creates new Identity gates for the specified qubit indices.
    ///
    /// # Arguments
    ///
    /// * `qubit_indices` - The indices of the qubits on which the Identity gates act.
    ///
    /// # Returns
    ///
    /// * `Vec<Gate>` - A vector of Gate structs representing Identity gates for each qubit index.
    pub fn i_multi_gate(qubit_indices: Vec<usize>) -> Vec<Self> {
        qubit_indices
            .into_iter()
            .map(Gate::i_gate)
            .collect()
    }

    /// Creates new controlled Identity gates for the specified qubit indices.
    /// 
    /// # Arguments
    /// 
    /// * `target_indices` - The indices of the target qubits.
    /// 
    /// * `control_indices` - The indices of the control qubits.
    /// 
    /// # Returns
    /// 
    /// * `Gates` - A vector of Gate structs representing controlled Identity gates for each target qubit index.
    pub fn i_controlled_gates(
        target_indices: Vec<usize>,
        control_indices: Vec<usize>,
    ) -> Vec<Self> {
        target_indices
            .into_iter()
            .map(|target_index| Gate::Operator(
                Box::new(Identity),
                vec![target_index],
                control_indices.clone(),
            ))
            .collect()
    }

    /// Creates a new Phase-S gate for the specified qubit index.
    ///
    /// # Arguments
    ///
    /// * `qubit_index` - The index of the qubit on which the Phase-S gate acts.
    ///
    /// # Returns
    ///
    /// * `Gate` - A new instance of the Gate struct representing a Phase-S gate.
    pub fn s_gate(qubit_index: usize) -> Self {
        Gate::Operator(Box::new(PhaseS), vec![qubit_index], vec![])
    }

    /// Creates new Phase-S gates for the specified qubit indices.
    ///
    /// # Arguments
    ///
    /// * `qubit_indices` - The indices of the qubits on which the Phase-S gates act.
    ///
    /// # Returns
    ///
    /// * `Vec<Gate>` - A vector of Gate structs representing Phase-S gates for each qubit index.
    pub fn s_multi_gate(qubit_indices: Vec<usize>) -> Vec<Self> {
        qubit_indices
            .into_iter()
            .map(Gate::s_gate)
            .collect()
    }

    /// Creates new controlled Phase-S gates for the specified qubit indices.
    /// 
    /// # Arguments
    /// 
    /// * `target_indices` - The indices of the target qubits.
    /// 
    /// * `control_indices` - The indices of the control qubits.
    /// 
    /// # Returns
    /// 
    /// * `Gates` - A vector of Gate structs representing controlled Phase-S gates for each target qubit index.
    pub fn s_controlled_gates(
        target_indices: Vec<usize>,
        control_indices: Vec<usize>,
    ) -> Vec<Self> {
        target_indices
            .into_iter()
            .map(|target_index| Gate::Operator(
                Box::new(PhaseS),
                vec![target_index],
                control_indices.clone(),
            ))
            .collect()
    }

    /// Creates a new Phase-S dagger gate for the specified qubit index.
    ///
    /// # Arguments
    ///
    /// * `qubit_index` - The index of the qubit on which the Phase-S dagger gate acts.
    ///
    /// # Returns
    ///
    /// * `Gate` - A new instance of the Gate struct representing a Phase-S dagger gate.
    pub fn s_dag_gate(qubit_index: usize) -> Self {
        Gate::Operator(Box::new(PhaseSdag), vec![qubit_index], vec![])
    }

    /// Creates new Phase-S dagger gates for the specified qubit indices.
    ///
    /// # Arguments
    ///
    /// * `qubit_indices` - The indices of the qubits on which the Phase-S dagger gates act.
    ///
    /// # Returns
    ///
    /// * `Vec<Gate>` - A vector of Gate structs representing Phase-S dagger gates for each qubit index.
    pub fn s_dag_multi_gate(qubit_indices: Vec<usize>) -> Vec<Self> {
        qubit_indices
            .into_iter()
            .map(Gate::s_dag_gate)
            .collect()
    }

    /// Creates new controlled Phase-S dagger gates for the specified qubit indices.
    /// 
    /// # Arguments
    /// 
    /// * `target_indices` - The indices of the target qubits.
    /// 
    /// * `control_indices` - The indices of the control qubits.
    /// 
    /// # Returns
    /// 
    /// * `Gates` - A vector of Gate structs representing controlled Phase-S dagger gates for each target qubit index.
    pub fn s_dag_controlled_gates(
        target_indices: Vec<usize>,
        control_indices: Vec<usize>,
    ) -> Vec<Self> {
        target_indices
            .into_iter()
            .map(|target_index| Gate::Operator(
                Box::new(PhaseSdag),
                vec![target_index],
                control_indices.clone(),
            ))
            .collect()
    }

    /// Creates a new Phase-T gate for the specified qubit index.
    ///
    /// # Arguments
    ///
    /// * `qubit_index` - The index of the qubit on which the Phase-T gate acts.
    ///
    /// # Returns
    ///
    /// * `Gate` - A new instance of the Gate struct representing a Phase-T gate.
    pub fn t_gate(qubit_index: usize) -> Self {
        Gate::Operator(Box::new(PhaseT), vec![qubit_index], vec![])
    }

    /// Creates new Phase-T gates for the specified qubit indices.
    ///
    /// # Arguments
    ///
    /// * `qubit_indices` - The indices of the qubits on which the Phase-T gates act.
    ///
    /// # Returns
    ///
    /// * `Vec<Gate>` - A vector of Gate structs representing Phase-T gates for each qubit index.
    pub fn t_multi_gate(qubit_indices: Vec<usize>) -> Vec<Self> {
        qubit_indices
            .into_iter()
            .map(Gate::t_gate)
            .collect()
    }

    /// Creates new controlled Phase-T gates for the specified qubit indices.
    /// 
    /// # Arguments
    /// 
    /// * `target_indices` - The indices of the target qubits.
    /// 
    /// * `control_indices` - The indices of the control qubits.
    /// 
    /// # Returns
    /// 
    /// * `Gates` - A vector of Gate structs representing controlled Phase-T gates for each target qubit index.
    pub fn t_controlled_gates(
        target_indices: Vec<usize>,
        control_indices: Vec<usize>,
    ) -> Vec<Self> {
        target_indices
            .into_iter()
            .map(|target_index| Gate::Operator(
                Box::new(PhaseT),
                vec![target_index],
                control_indices.clone(),
            ))
            .collect()
    }

    /// Creates a new Phase-T dagger gate for the specified qubit index.
    ///
    /// # Arguments
    ///
    /// * `qubit_index` - The index of the qubit on which the Phase-T dagger gate acts.
    ///
    /// # Returns
    ///
    /// * `Gate` - A new instance of the Gate struct representing a Phase-T dagger gate.
    pub fn t_dag_gate(qubit_index: usize) -> Self {
        Gate::Operator(Box::new(PhaseTdag), vec![qubit_index], vec![])
    }

    /// Creates new Phase-T dagger gates for the specified qubit indices.
    ///
    /// # Arguments
    ///
    /// * `qubit_indices` - The indices of the qubits on which the Phase-T dagger gates act.
    ///
    /// # Returns
    ///
    /// * `Vec<Gate>` - A vector of Gate structs representing Phase-T dagger gates for each qubit index.
    pub fn t_dag_multi_gate(qubit_indices: Vec<usize>) -> Vec<Self> {
        qubit_indices
            .into_iter()
            .map(Gate::t_dag_gate)
            .collect()
    }

    /// Creates new controlled Phase-T dagger gates for the specified qubit indices.
    /// 
    /// # Arguments
    /// 
    /// * `target_indices` - The indices of the target qubits.
    /// 
    /// * `control_indices` - The indices of the control qubits.
    /// 
    /// # Returns
    /// 
    /// * `Gates` - A vector of Gate structs representing controlled Phase-T dagger gates for each target qubit index.
    pub fn t_dag_controlled_gates(
        target_indices: Vec<usize>,
        control_indices: Vec<usize>,
    ) -> Vec<Self> {
        target_indices
            .into_iter()
            .map(|target_index| Gate::Operator(
                Box::new(PhaseTdag),
                vec![target_index],
                control_indices.clone(),
            ))
            .collect()
    }

    /// Creates a new Phase Shift (P) gate for the specified qubit index and angle.
    ///
    /// # Arguments
    ///
    /// * `qubit_index` - The index of the qubit on which the Phase Shift gate acts.
    /// * `angle` - The phase shift angle in radians.
    ///
    /// # Returns
    ///
    /// * `Gate` - A new instance of the Gate struct representing a Phase Shift gate.
    pub fn p_gate(qubit_index: usize, angle: f64) -> Self {
        Gate::Operator(Box::new(PhaseShift::new(angle)), vec![qubit_index], vec![])
    }

    /// Creates new Phase Shift (P) gates for the specified qubit indices and angle.
    ///
    /// # Arguments
    ///
    /// * `qubit_indices` - The indices of the qubits on which the Phase Shift gates act.
    /// * `angle` - The phase shift angle in radians for all gates.
    ///
    /// # Returns
    ///
    /// * `Vec<Gate>` - A vector of Gate structs representing Phase Shift gates for each qubit index.
    pub fn p_multi_gate(qubit_indices: Vec<usize>, angle: f64) -> Vec<Self> {
        let op_template = PhaseShift::new(angle);
        qubit_indices
            .into_iter()
            .map(|qubit_index| Gate::Operator(Box::new(op_template), vec![qubit_index], vec![]))
            .collect()
    }

    /// Creates new controlled Phase Shift (P) gates for the specified qubit indices and angle.
    /// 
    /// # Arguments
    /// 
    /// * `target_indices` - The indices of the target qubits.
    /// * `control_indices` - The indices of the control qubits.
    /// * `angle` - The phase shift angle in radians for all gates.
    /// 
    /// # Returns
    /// 
    /// * `Gates` - A vector of Gate structs representing controlled Phase Shift gates for each target qubit index.
    pub fn p_controlled_gates(
        target_indices: Vec<usize>,
        control_indices: Vec<usize>,
        angle: f64,
    ) -> Vec<Self> {
        let op_template = PhaseShift::new(angle);
        target_indices
            .into_iter()
            .map(|target_index| Gate::Operator(
                Box::new(op_template),
                vec![target_index],
                control_indices.clone(),
            ))
            .collect()
    }

    /// Creates a new RotateX (RX) gate for the specified qubit index and angle.
    ///
    /// # Arguments
    ///
    /// * `qubit_index` - The index of the qubit on which the RotateX gate acts.
    /// * `angle` - The rotation angle in radians.
    ///
    /// # Returns
    ///
    /// * `Gate` - A new instance of the Gate struct representing a RotateX gate.
    pub fn rx_gate(qubit_index: usize, angle: f64) -> Self {
        Gate::Operator(Box::new(RotateX::new(angle)), vec![qubit_index], vec![])
    }

    /// Creates new RotateX (RX) gates for the specified qubit indices and angle.
    ///
    /// # Arguments
    ///
    /// * `qubit_indices` - The indices of the qubits on which the RotateX gates act.
    /// * `angle` - The rotation angle in radians for all gates.
    ///
    /// # Returns
    ///
    /// * `Vec<Gate>` - A vector of Gate structs representing RotateX gates for each qubit index.
    pub fn rx_multi_gate(qubit_indices: Vec<usize>, angle: f64) -> Vec<Self> {
        let op_template = RotateX::new(angle);
        qubit_indices
            .into_iter()
            .map(|qubit_index| Gate::Operator(Box::new(op_template), vec![qubit_index], vec![]))
            .collect()
    }

    /// Creates new controlled RotateX (RX) gates for the specified qubit indices and angle.
    /// 
    /// # Arguments
    /// 
    /// * `target_indices` - The indices of the target qubits.
    /// * `control_indices` - The indices of the control qubits.
    /// * `angle` - The rotation angle in radians for all gates.
    /// 
    /// # Returns
    /// 
    /// * `Gates` - A vector of Gate structs representing controlled RotateX gates for each target qubit index.
    pub fn rx_controlled_gates(
        target_indices: Vec<usize>,
        control_indices: Vec<usize>,
        angle: f64,
    ) -> Vec<Self> {
        let op_template = RotateX::new(angle);
        target_indices
            .into_iter()
            .map(|target_index| Gate::Operator(
                Box::new(op_template),
                vec![target_index],
                control_indices.clone(),
            ))
            .collect()
    }

    /// Creates a new RotateY (RY) gate for the specified qubit index and angle.
    ///
    /// # Arguments
    ///
    /// * `qubit_index` - The index of the qubit on which the RotateY gate acts.
    /// * `angle` - The rotation angle in radians.
    ///
    /// # Returns
    ///
    /// * `Gate` - A new instance of the Gate struct representing a RotateY gate.
    pub fn ry_gate(qubit_index: usize, angle: f64) -> Self {
        Gate::Operator(Box::new(RotateY::new(angle)), vec![qubit_index], vec![])
    }

    /// Creates new RotateY (RY) gates for the specified qubit indices and angle.
    ///
    /// # Arguments
    ///
    /// * `qubit_indices` - The indices of the qubits on which the RotateY gates act.
    /// * `angle` - The rotation angle in radians for all gates.
    ///
    /// # Returns
    ///
    /// * `Vec<Gate>` - A vector of Gate structs representing RotateY gates for each qubit index.
    pub fn ry_multi_gate(qubit_indices: Vec<usize>, angle: f64) -> Vec<Self> {
        let op_template = RotateY::new(angle);
        qubit_indices
            .into_iter()
            .map(|qubit_index| Gate::Operator(Box::new(op_template), vec![qubit_index], vec![]))
            .collect()
    }

    /// Creates new controlled RotateY (RY) gates for the specified qubit indices and angle.
    /// 
    /// # Arguments
    /// 
    /// * `target_indices` - The indices of the target qubits.
    /// * `control_indices` - The indices of the control qubits.
    /// * `angle` - The rotation angle in radians for all gates.
    /// 
    /// # Returns
    /// 
    /// * `Gates` - A vector of Gate structs representing controlled RotateY gates for each target qubit index.
    pub fn ry_controlled_gates(
        target_indices: Vec<usize>,
        control_indices: Vec<usize>,
        angle: f64,
    ) -> Vec<Self> {
        let op_template = RotateY::new(angle);
        target_indices
            .into_iter()
            .map(|target_index| Gate::Operator(
                Box::new(op_template),
                vec![target_index],
                control_indices.clone(),
            ))
            .collect()
    }

    /// Creates a new RotateZ (RZ) gate for the specified qubit index and angle.
    ///
    /// # Arguments
    ///
    /// * `qubit_index` - The index of the qubit on which the RotateZ gate acts.
    /// * `angle` - The rotation angle in radians.
    ///
    /// # Returns
    ///
    /// * `Gate` - A new instance of the Gate struct representing a RotateZ gate.
    pub fn rz_gate(qubit_index: usize, angle: f64) -> Self {
        Gate::Operator(Box::new(RotateZ::new(angle)), vec![qubit_index], vec![])
    }

    /// Creates new RotateZ (RZ) gates for the specified qubit indices and angle.
    ///
    /// # Arguments
    ///
    /// * `qubit_indices` - The indices of the qubits on which the RotateZ gates act.
    /// * `angle` - The rotation angle in radians for all gates.
    ///
    /// # Returns
    ///
    /// * `Vec<Gate>` - A vector of Gate structs representing RotateZ gates for each qubit index.
    pub fn rz_multi_gate(qubit_indices: Vec<usize>, angle: f64) -> Vec<Self> {
        let op_template = RotateZ::new(angle);
        qubit_indices
            .into_iter()
            .map(|qubit_index| Gate::Operator(Box::new(op_template), vec![qubit_index], vec![]))
            .collect()
    }

    /// Creates new controlled RotateZ (RZ) gates for the specified qubit indices and angle.
    /// 
    /// # Arguments
    /// 
    /// * `target_indices` - The indices of the target qubits.
    /// * `control_indices` - The indices of the control qubits.
    /// * `angle` - The rotation angle in radians for all gates.
    /// 
    /// # Returns
    /// 
    /// * `Gates` - A vector of Gate structs representing controlled RotateZ gates for each target qubit index.
    pub fn rz_controlled_gates(
        target_indices: Vec<usize>,
        control_indices: Vec<usize>,
        angle: f64,
    ) -> Vec<Self> {
        let op_template = RotateZ::new(angle);
        target_indices
            .into_iter()
            .map(|target_index| Gate::Operator(
                Box::new(op_template),
                vec![target_index],
                control_indices.clone(),
            ))
            .collect()
    }

    /// Creates a new Unitary2 gate for the specified qubit index and unitary matrix.
    /// 
    /// # Arguments
    /// 
    /// * `qubit_index` - The index of the qubit on which the Unitary2 gate acts.
    /// * `unitary` - The unitary matrix to be applied.
    /// 
    /// # Returns
    /// 
    /// * `Result<Gate, Error>` - A Gate struct representing a Unitary2 gate if the unitary is valid, else an error.
    pub fn unitary2_gate(qubit_index: usize, unitary: [[Complex<f64>; 2]; 2]) -> Result<Self, Error> {
        Ok(Gate::Operator(
            Box::new(Unitary2::new(unitary)?),
            vec![qubit_index],
            vec![],
        ))
    }

    /// Creates new Unitary2 gates for the specified qubit indices and unitary matrix.
    /// 
    /// # Arguments
    /// 
    /// * `qubit_indices` - The indices of the qubits on which the Unitary2 gates act.
    /// 
    /// * `unitary` - The unitary matrix to be applied.
    /// 
    /// # Returns
    /// 
    /// * `Result<Vec<Gate>, Error>` - A vector of Gate structs representing Unitary2 gates for each qubit index if the unitary is valid,, else an error.
    pub fn unitary2_multi_gate(
        qubit_indices: Vec<usize>,
        unitary: [[Complex<f64>; 2]; 2],
    ) -> Result<Vec<Self>, Error> {
        let op_template = Unitary2::new(unitary)?;
        let gates = qubit_indices
            .into_iter()
            .map(|qubit_index| Gate::Operator(Box::new(op_template), vec![qubit_index], vec![]))
            .collect();
        Ok(gates)
    }

    /// Creates new controlled Unitary2 gates for the specified qubit indices and unitary matrix.
    /// 
    /// # Arguments
    /// 
    /// * `target_indices` - The indices of the target qubits.
    /// 
    /// * `control_indices` - The indices of the control qubits.
    /// 
    /// * `unitary` - The unitary matrix to be applied.
    /// 
    /// # Returns
    /// 
    /// * `Result<Vec<Gate>, Error>` - A vector of Gate structs representing controlled Unitary2 gates for each target qubit index if the unitary is valid, else an error.
    pub fn unitary2_controlled_gates(
        target_indices: Vec<usize>,
        control_indices: Vec<usize>,
        unitary: [[Complex<f64>; 2]; 2],
    ) -> Result<Vec<Self>, Error> {
        let op_template = Unitary2::new(unitary)?;
        let gates = target_indices
            .into_iter()
            .map(|target_index| Gate::Operator(
                Box::new(op_template),
                vec![target_index],
                control_indices.clone(),
            ))
            .collect();
        Ok(gates)
    }

    /// Creates a new Unitary2 gate for the specified qubit index and unitary matrix using a rotation angle and phase shift.
    /// Unlike custom Unitary2 gates, the generated unitary matrix is guaranteed to be valid.
    /// Therefore, this method does not return an error.
    /// 
    /// # Arguments
    /// 
    /// * qubit_index - The index of the qubit on which the Unitary2 gate acts.
    /// 
    /// * theta - The rotation angle in radians.
    /// 
    /// * phi - The phase shift in radians.
    /// 
    /// # Returns
    /// 
    /// * Gate - A new instance of the Gate struct representing a Unitary2 gate.
    pub fn ry_phase_gate(
        qubit_index: usize,
        theta: f64,
        phi: f64,
    ) -> Self {
        Gate::Operator(
            Box::new(Unitary2::from_ry_phase(theta, phi)),
            vec![qubit_index],
            vec![],
        )
    }

    /// Creates new Unitary2 gates for the specified qubit indices and unitary matrix using a rotation angle and phase shift.
    /// Unlike custom Unitary2 gates, the generated unitary matrix is guaranteed to be valid.
    /// Therefore, this method does not return an error.
    /// 
    /// # Arguments
    /// 
    /// * qubit_indices - The indices of the qubits on which the Unitary2 gates act.
    /// 
    /// * theta - The rotation angle in radians for all gates.
    /// 
    /// * phi - The phase shift in radians for all gates.
    /// 
    /// # Returns
    /// 
    /// * `Vec<Gate>` - A vector of Gate structs representing Unitary2 gates for each qubit index.
    pub fn ry_phase_multi_gate(
        qubit_indices: Vec<usize>,
        theta: f64,
        phi: f64,
    ) -> Vec<Self> {
        let op_template = Unitary2::from_ry_phase(theta, phi);
        qubit_indices
            .into_iter()
            .map(|qubit_index| Gate::Operator(Box::new(op_template), vec![qubit_index], vec![]))
            .collect()
    }

    /// Creates new controlled Unitary2 gates for the specified qubit indices and unitary matrix using a rotation angle and phase shift.
    /// 
    /// # Arguments
    /// 
    /// * target_indices - The indices of the target qubits.
    /// 
    /// * control_indices - The indices of the control qubits.
    /// 
    /// * theta - The rotation angle in radians for all gates.
    /// 
    /// * phi - The phase shift in radians for all gates.
    /// 
    /// # Returns
    /// 
    /// * `Vec<Gate>` - A vector of Gate structs representing controlled Unitary2 gates for each target qubit index.
    pub fn ry_phase_controlled_gates(
        target_indices: Vec<usize>,
        control_indices: Vec<usize>,
        theta: f64,
        phi: f64,
    ) -> Vec<Self> {
        let op_template = Unitary2::from_ry_phase(theta, phi);
        target_indices
            .into_iter()
            .map(|target_index| Gate::Operator(
                Box::new(op_template),
                vec![target_index],
                control_indices.clone(),
            ))
            .collect()
    }

    // -- MULTI-QUBIT GATES --

    /// Creates a CNOT gate for the specified target and control qubit indices.
    ///
    /// # Arguments
    ///
    /// * `target_index` - The index of the target qubit.
    ///
    /// * `control_index` - The index of the control qubit.
    ///
    /// # Returns
    ///
    /// * `Gate` - A new instance of the Gate struct representing a CNOT gate.
    pub fn cnot_gate(target_index: usize, control_index: usize) -> Self {
        Gate::Operator(
            Box::new(CNOT),
            vec![target_index],
            vec![control_index],
        )
    }

    /// Creates a new SWAP gate for the specified qubit index.
    ///
    /// # Arguments
    ///
    /// * `qubit1` - The index of the first qubit.
    /// * `qubit2` - The index of the second qubit.
    ///
    /// # Returns
    ///
    /// * `Gate` - A new instance of the Gate struct representing a SWAP gate.
    pub fn swap_gate(qubit1_index: usize, qubit2_index: usize) -> Self {
        Gate::Operator(Box::new(SWAP), vec![qubit1_index, qubit2_index], vec![])
    }

    /// Creates a new controlled SWAP gate for the specified target and control qubit indices.
    /// 
    /// # Arguments
    /// 
    /// * `target_indices` - The indices of the target qubits.
    /// * `control_indices` - The indices of the control qubits.
    /// 
    /// # Returns
    /// 
    /// * `Gate` - A new instance of the Gate struct representing a controlled SWAP gate.
    pub fn swap_controlled_gate(
        target_indices: Vec<usize>,
        control_indices: Vec<usize>,
    ) -> Self {
        Gate::Operator(Box::new(SWAP), target_indices, control_indices)
    }

    /// Creates a new Toffoli gate for the specified target and control qubit indices.
    ///
    /// # Arguments
    ///
    /// * `target_index` - The index of the target qubit.
    ///
    /// * `control_indices` - The indices of the control qubits.
    ///
    /// # Returns
    ///
    /// * `Gate` - A new instance of the Gate struct representing a Toffoli gate.
    pub fn toffoli_gate(target_index: usize, control_indices: Vec<usize>) -> Self {
        Gate::Operator(Box::new(Toffoli), vec![target_index], control_indices)
    }

    /// Creates a new Matchgate with the specified qubit index and its adjacent as targets.
    ///
    /// # Arguments
    ///
    /// * `target_index` - The index of the first target qubit. The second target qubit is assumed to be the next qubit.
    /// * `theta` - The angle of rotation in radians.
    /// * `phi1` - The first phase shift in radians.
    /// * `phi2` - The second phase shift in radians.
    ///
    /// # Returns
    ///
    /// * `Gate` - A new instance of the Gate struct representing a Matchgate.
    /// 
    /// # Warning
    ///
    /// This gate is not yet compilable to OpenQASM, since it requires advanced decomposition techniques.
    pub fn matchgate(
        target_index: usize,
        theta: f64,
        phi1: f64,
        phi2: f64,
    ) -> Self {
        Gate::Operator(
            Box::new(Matchgate::new(theta, phi1, phi2)),
            vec![target_index],
            vec![],
        )
    }

    /// Creates a new controlled Matchgate with the specified qubit index and its adjacent as target qubits.
    ///
    /// # Arguments
    ///
    /// * `target_index` - The index of the first target qubit. The second target qubit is assumed to be the next qubit.
    /// * `control_indices` - The indices of the control qubits.
    /// * `theta` - The angle of rotation in radians.
    /// * `phi1` - The first phase shift in radians.
    /// * `phi2` - The second phase shift in radians.
    ///
    /// # Returns
    ///
    /// * `Gate` - A new instance of the Gate struct representing a controlled Matchgate.
    /// 
    /// # Warning
    ///
    /// This gate is not yet compilable to OpenQASM, since it requires advanced decomposition techniques.
    pub fn controlled_matchgate(
        target_index: usize,
        control_indices: Vec<usize>,
        theta: f64,
        phi1: f64,
        phi2: f64,
    ) -> Self {
        Gate::Operator(
            Box::new(Matchgate::new(theta, phi1, phi2)),
            vec![target_index],
            control_indices,
        )
    }
}
