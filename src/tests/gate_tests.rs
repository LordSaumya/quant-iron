use crate::{
    components::{
        gate::Gate, measurement::MeasurementBasis, operator::{Hadamard, CNOT}, state::State
    },
    errors::Error,
};

#[test]
fn test_gate_new_operator() {
    // Create a new operator gate
    let _gate: Gate = Gate::Operator(Box::new(Hadamard), vec![0, 1], None);
}

#[test]
fn test_gate_new_measurement() {
    // Create a new measurement gate
    let _gate: Gate = Gate::Measurement(MeasurementBasis::Computational, vec![0, 1]);
}

#[test]
fn test_gate_get_target_qubits() {
    // Create a new operator gate
    let gate: Gate = Gate::Operator(Box::new(Hadamard), vec![0, 1], None);
    
    // Get the target indices of the gate
    let target_indices: &Vec<usize> = gate.get_target_qubits();
    
    // Check if the target indices are correct
    assert_eq!(target_indices, &vec![0, 1]);

    // Create a new measurement gate
    let gate: Gate = Gate::Measurement(MeasurementBasis::Computational, vec![0, 2]);

    // Get the target indices of the gate
    let target_indices: &Vec<usize> = gate.get_target_qubits();

    // Check if the target indices are correct
    assert_eq!(target_indices, &vec![0, 2]);
}

#[test]
fn test_gate_get_control_qubits() {
    // Create a new operator gate with control qubits
    let gate: Gate = Gate::Operator(Box::new(CNOT), vec![0, 1], Some(vec![2]));

    // Get the control indices of the gate
    let control_indices: Option<&Vec<usize>> = gate.get_control_qubits();

    // Check if the control indices are correct
    assert_eq!(control_indices, Some(&vec![2]));

    // Create a new measurement gate
    let gate: Gate = Gate::Measurement(MeasurementBasis::Computational, vec![0, 2]);

    // Get the control indices of the gate
    let control_indices: Option<&Vec<usize>> = gate.get_control_qubits();

    // Check if the control indices are None
    assert_eq!(control_indices, None);
}

#[test]
fn test_gate_apply() {
    // Create a new operator gate
    let gate: Gate = Gate::Operator(Box::new(Hadamard), vec![0], None);
    let state: State = State::new_zero(1).unwrap();

    // Apply gate
    let result: Result<State, Error> = gate.apply(&state);

    // Check if the result is correct
    assert!(result.is_ok());
    let new_state: State = result.unwrap();
    assert_eq!(new_state, State::new_plus(1).unwrap());

    // Create a new measurement gate
    let gate: Gate = Gate::Measurement(MeasurementBasis::Computational, vec![0]);
    let state: State = State::new_plus(1).unwrap();

    // Apply gate
    let result: Result<State, Error> = gate.apply(&state);

    // Check if the result is correct
    assert!(result.is_ok());
    let new_state: State = result.unwrap();
    assert!(new_state == State::new_zero(1).unwrap() || new_state == State::new_basis_n(1, 1).unwrap());
}