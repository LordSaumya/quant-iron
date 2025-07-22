use crate::{
    circuit::{Circuit, CircuitBuilder},
    components::{gate::Gate, measurement::MeasurementBasis, state::State},
    errors::Error,
};
use std::f64::consts::PI;

#[test]
fn test_circuit_new() {
    // Create a new circuit with 2 qubits
    let circuit = Circuit::new(2);
    assert_eq!(circuit.num_qubits, 2);
    assert!(circuit.gates.is_empty());
}

#[test]
fn test_circuit_with_gates_success() {
    let gates = vec![Gate::h_gate(1), Gate::cnot_gate(0, 1)];
    let circuit = Circuit::with_gates(gates, 2).expect("Failed to create circuit with gates");
    assert_eq!(circuit.num_qubits, 2);
    assert_eq!(circuit.gates.len(), 2);
}

#[test]
fn test_circuit_with_gates_failure() {
    let gates = vec![Gate::h_gate(1), Gate::cnot_gate(0, 3)]; // Invalid qubit index
    let circuit = Circuit::with_gates(gates, 2);
    assert!(circuit.is_err());
    assert_eq!(circuit.err().unwrap(), Error::InvalidQubitIndex(3, 2));
}

#[test]
fn test_circuit_add_gate_success() {
    let mut circuit = Circuit::new(2);
    let gate = Gate::h_gate(1);
    assert!(circuit.add_gate(gate).is_ok());
    assert_eq!(circuit.gates.len(), 1);
}

#[test]
fn test_circuit_add_gate_failure() {
    let mut circuit = Circuit::new(2);
    let gate = Gate::cnot_gate(0, 3); // Invalid qubit index
    assert!(circuit.add_gate(gate.clone()).is_err());
    assert_eq!(
        circuit.add_gate(gate).unwrap_err(),
        Error::InvalidQubitIndex(3, 2)
    );
}

#[test]
fn test_circuit_add_gates_success() {
    let mut circuit = Circuit::new(2);
    let gates = vec![Gate::h_gate(1), Gate::cnot_gate(0, 1)];
    assert!(circuit.add_gates(gates).is_ok());
    assert_eq!(circuit.gates.len(), 2);
}

#[test]
fn test_circuit_add_gates_failure() {
    let mut circuit = Circuit::new(2);
    let gates = vec![Gate::h_gate(1), Gate::cnot_gate(0, 3)]; // Invalid qubit index
    assert!(circuit.add_gates(gates.clone()).is_err());
    assert_eq!(
        circuit.add_gates(gates).unwrap_err(),
        Error::InvalidQubitIndex(3, 2)
    );
}

#[test]
fn test_circuit_get_num_qubits() {
    let circuit = Circuit::new(3);
    assert_eq!(circuit.get_num_qubits(), 3);
}

#[test]
fn test_circuit_get_gates() {
    let gates = vec![Gate::h_gate(0), Gate::cnot_gate(0, 1)];
    let circuit = Circuit::with_gates(gates, 2).expect("Failed to create circuit with gates");
    assert_eq!(circuit.get_gates().len(), 2);
}

#[test]
fn test_circuit_execute_success() {
    let gates = Gate::h_multi_gate(vec![0, 1]);
    let circuit = Circuit::with_gates(gates, 2).expect("Failed to create circuit with gates");

    let initial_state = State::new_zero(2).expect("Failed to create initial state"); // Correct number of qubits
    let expected_state = State::new_plus(2).expect("Failed to create expected state");
    let final_state = circuit
        .execute(&initial_state)
        .expect("Failed to execute circuit");

    assert_eq!(final_state, expected_state);
}

#[test]
fn test_circuit_execute_failure() {
    let gates = vec![Gate::h_gate(0), Gate::cnot_gate(0, 1)];
    let circuit = Circuit::with_gates(gates, 2).expect("Failed to create circuit with gates");

    let initial_state = State::new_zero(1).expect("Failed to create initial state"); // Incorrect number of qubits
    let result = circuit.execute(&initial_state);

    assert!(result.is_err());
    assert_eq!(result.unwrap_err(), Error::InvalidNumberOfQubits(1));
}

#[test]
fn test_circuit_trace_execution_success() {
    let gates = vec![Gate::h_gate(0), Gate::cnot_gate(0, 1)];
    let circuit = Circuit::with_gates(gates, 2).expect("Failed to create circuit with gates");

    let initial_state = State::new_zero(2).expect("Failed to create initial state");
    let trace = circuit
        .trace_execution(&initial_state)
        .expect("Failed to trace execution");

    assert_eq!(trace.len(), 3); // Should have three states: initial, after H, after CNOT
}

#[test]
fn test_circuit_trace_execution_failure() {
    let gates = vec![Gate::h_gate(0), Gate::cnot_gate(0, 1)];
    let circuit = Circuit::with_gates(gates, 2).expect("Failed to create circuit with gates");

    let initial_state = State::new_zero(1).expect("Failed to create initial state"); // Incorrect number of qubits
    let result = circuit.trace_execution(&initial_state);

    assert!(result.is_err());
    assert_eq!(result.unwrap_err(), Error::InvalidNumberOfQubits(1));
}

#[test]
fn test_circuit_to_qasm() {
    let circuit = CircuitBuilder::new(5)
        .h_gate(0)
        .cx_gates(vec![1, 2], vec![0])
        .p_gates(vec![0, 1, 2], PI)
        .measure_gate(MeasurementBasis::X, vec![0, 1])
        .swap_gate(3, 4)
        .ctdag_gates(vec![2, 3], vec![0, 3])
        .crz_gates(vec![4], vec![0, 1], PI)
        .measure_gate(
            MeasurementBasis::Custom([[0.0.into(), 1.0.into()], [1.0.into(), 0.0.into()]]),
            vec![2, 3, 4],
        )
        .build_final();

    let qasm = circuit
        .expect("Could not build circuit")
        .to_qasm(None::<&str>)
        .expect("Could not convert circuit to QASM");

    assert!(!qasm.is_empty());
    assert!(qasm.contains("OPENQASM 3.0")); // Full testing for this function is performed manually by inspection
}

#[test]
fn test_circuit_builder_new() {
    let builder = CircuitBuilder::new(3);
    assert_eq!(builder.num_qubits, 3);
    assert!(builder.gates.is_empty());
}

#[test]
fn test_circuit_builder_add_gate() {
    let mut builder = CircuitBuilder::new(2);
    let gate = Gate::h_gate(1);
    builder.add_gate(gate);
    assert_eq!(builder.gates.len(), 1);
}

#[test]
fn test_circuit_builder_add_gates() {
    let mut builder = CircuitBuilder::new(2);
    let gates = vec![Gate::h_gate(1), Gate::cnot_gate(0, 1)];
    builder.add_gates(gates);
    assert_eq!(builder.gates.len(), 2);
}

#[test]
fn test_circuit_builder_build_success() {
    let mut builder = CircuitBuilder::new(2);
    let circuit = builder
        .h_gate(0)
        .cnot_gate(0, 1)
        .build()
        .expect("Failed to build circuit");

    assert_eq!(circuit.num_qubits, 2);
    assert_eq!(circuit.gates.len(), 2);
    assert!(!builder.gates.is_empty()); // Gates should remain in the builder
}

#[test]
fn test_circuit_builder_build_failure() {
    let mut builder = CircuitBuilder::new(2);
    let circuit = builder
        .h_gate(0)
        .cnot_gate(0, 3) // Invalid qubit index
        .build();

    assert!(circuit.is_err());
    assert_eq!(circuit.err().unwrap(), Error::InvalidQubitIndex(3, 2));
}

#[test]
fn test_circuit_builder_build_final_success() {
    let mut builder = CircuitBuilder::new(2);
    let circuit = builder
        .h_gate(0)
        .cnot_gate(0, 1)
        .build_final()
        .expect("Failed to build final circuit");

    assert_eq!(circuit.num_qubits, 2);
    assert_eq!(circuit.gates.len(), 2);
    assert!(builder.gates.is_empty()); // Gates should NOT remain in the builder
}

#[test]
fn test_circuit_builder_build_final_failure() {
    let mut builder = CircuitBuilder::new(2);
    let circuit = builder
        .h_gate(0)
        .cnot_gate(0, 3) // Invalid qubit index
        .build_final();
    assert!(circuit.is_err());
    assert_eq!(circuit.err().unwrap(), Error::InvalidQubitIndex(3, 2));
}

#[test]
fn test_circuit_builder_build_subroutine() {
    let mut builder = CircuitBuilder::new(2);
    let subroutine = builder
        .h_gate(0)
        .cnot_gate(0, 1)
        .build_subroutine();

    assert_eq!(subroutine.num_qubits, 2);
    assert_eq!(subroutine.gates.len(), 2);
}

#[test]
fn test_circuit_builder_add_subroutine() {
    let mut builder = CircuitBuilder::new(2);
    let subroutine = builder
        .h_gate(0)
        .cnot_gate(0, 1)
        .build_subroutine();

    builder.add_subroutine(subroutine);
    assert_eq!(builder.gates.len(), 2);
}
