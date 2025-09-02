use crate::{
    components::{
        gate::Gate, measurement::MeasurementBasis, operator::{Hadamard, Pauli, CNOT}, parametric::parameter::Parameter, pauli_string::PauliString, state::State
    },
    errors::Error, parametric_gate::{ParametricP, ParametricRy},
};
use std::f64::consts::PI;

#[test]
fn test_gate_new_operator() {
    // Create a new operator gate
    let _gate: Gate = Gate::Operator(Box::new(Hadamard), vec![0, 1], vec![]);
}

#[test]
fn test_gate_new_measurement() {
    // Create a new measurement gate
    let _gate: Gate = Gate::Measurement(MeasurementBasis::Computational, vec![0, 1]);
}

#[test]
fn test_gate_new_parametric() {
    // Create a new parametric gate
    let _gate: Gate = Gate::Parametric(Box::new(ParametricP{ parameter: Parameter::new([0.5]) }), vec![0], vec![]);
}

#[test]
fn test_gate_new_pauli_string() {
    // Create a new Pauli string gate
    let pauli_string = PauliString::new(1.0.into()).with_op(1, Pauli::X).with_op(2, Pauli::Y);
    let _gate: Gate = Gate::PauliString(pauli_string);
}

#[test]
fn test_gate_get_target_qubits() {
    // Create a new operator gate
    let gate: Gate = Gate::Operator(Box::new(Hadamard), vec![0, 1], vec![]);

    // Get the target indices of the gate
    let target_indices: &Vec<usize> = &gate.get_target_qubits();

    // Check if the target indices are correct
    assert_eq!(target_indices, &vec![0, 1]);

    // Create a new measurement gate
    let gate: Gate = Gate::Measurement(MeasurementBasis::Computational, vec![0, 2]);

    // Get the target indices of the gate
    let target_indices: &Vec<usize> = &gate.get_target_qubits();

    // Check if the target indices are correct
    assert_eq!(target_indices, &vec![0, 2]);

    // Create a new parametric gate
    let gate: Gate = Gate::Parametric(Box::new(ParametricP{ parameter: Parameter::new([0.5]) }), vec![0, 1], vec![]);

    // Get the target indices of the gate
    let target_indices: &Vec<usize> = &gate.get_target_qubits();

    // Check if the target indices are correct
    assert_eq!(target_indices, &vec![0, 1]);
}

#[test]
fn test_gate_get_control_qubits() {
    // Create a new operator gate with control qubits
    let gate: Gate = Gate::Operator(Box::new(CNOT), vec![0, 1], vec![2]);

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

    // Create a new parametric gate
    let gate: Gate = Gate::Parametric(Box::new(ParametricP{ parameter: Parameter::new([0.5]) }), vec![0], vec![1, 2]);

    // Get the control indices of the gate
    let control_indices: Option<&Vec<usize>> = gate.get_control_qubits();

    // Check if the control indices are correct
    assert_eq!(control_indices, Some(&vec![1, 2]));
}

#[test]
fn test_gate_apply() {
    // Create a new operator gate
    let gate: Gate = Gate::Operator(Box::new(Hadamard), vec![0], vec![]);
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
    assert!(
        new_state == State::new_zero(1).unwrap() || new_state == State::new_basis_n(1, 1).unwrap()
    );

    // Create a new parametric gate
    let gate: Gate = Gate::Parametric(Box::new(ParametricRy{parameter: Parameter::new([PI/2.0])}), vec![0], vec![]);
    let state: State = State::new_zero(1).unwrap();

    // Apply gate
    let result: Result<State, Error> = gate.apply(&state);

    // Check if the result is correct
    assert!(result.is_ok());
    let new_state: State = result.unwrap();
    assert_eq!(new_state, State::new_plus(1).unwrap());

    // Create a new pauli string gate
    let gate: Gate = Gate::PauliString(PauliString::new(2.0.into()).with_op(0, Pauli::X).with_op(1, Pauli::X));
    let state: State = State::new_zero(2).unwrap();

    // Apply gate
    let result: Result<State, Error> = gate.apply(&state);

    // Check if the result is correct
    assert!(result.is_ok());
    let new_state: State = result.unwrap();
    assert_eq!(new_state, State::new_basis_n(2,3).unwrap());
}
