use crate::{
    errors::Error,
};

use std::f64::consts::PI;

#[test]
fn test_circuit_macro_single_qubit_gates_success() {
    let circuit = circuit! {
        qubits: 3,
        h(0),
        x(1),
        y(2),
        z(0),
        s(1),
        t(2),
        id(0),
        sdag(1),
        tdag(2)
    }
    .expect("Failed to create circuit with single qubit gates");

    assert_eq!(circuit.num_qubits, 3);
    assert_eq!(circuit.gates.len(), 9);
}

#[test]
fn test_circuit_macro_single_qubit_gates_failure() {
    let circuit = circuit! {
        qubits: 3,
        h(0),
        x(1),
        y(2),
        z(0),
        s(1),
        t(2),
        id(0),
        sdag(1),
        tdag(3) // Invalid qubit index
    };

    assert!(circuit.is_err());
    assert_eq!(circuit.err().unwrap(), Error::InvalidQubitIndex(3, 3));
}

#[test]
fn test_circuit_macro_mixed_multi_qubit_gates_success() {
    let circuit = circuit! {
        qubits: 3,
        h(0),
        h([0, 1]),
        x(1),
        x([1, 2]),
        y(2),
        y([0, 2]),
        z(0),
        z([0, 1]),
        s(1),
        s([1, 2]),
        t(2),
        t([0, 2]),
        id(0),
        id([0, 1]),
        sdag(1),
        sdag([1, 2]),
        tdag(2),
        tdag([0, 2])
    }
    .expect("Failed to create circuit with mixed multi-qubit gates");

    assert_eq!(circuit.num_qubits, 3);
    assert_eq!(circuit.gates.len(), 9 * 3);
}

#[test]
fn test_circuit_macro_mixed_multi_qubit_gates_failure() {
    let circuit = circuit! {
        qubits: 3,
        h(0),
        h([0, 1]),
        x(1),
        x([1, 2]),
        y(2),
        y([0, 2]),
        z(0),
        z([0, 1]),
        s(1),
        s([1, 2]),
        t(2),
        t([0, 2]),
        id(0),
        id([0, 1]),
        sdag(1),
        sdag([1, 2]),
        tdag(2),
        tdag([0, 3]) // Invalid qubit index
    };

    assert!(circuit.is_err());
    assert_eq!(circuit.err().unwrap(), Error::InvalidQubitIndex(3, 3));
}

#[test]
fn test_circuit_macro_two_three_qubit_gates_success() {
    let circuit = circuit! {
        qubits: 5,
        cnot(0, 1),
        swap(1, 2),
        cswap(0, 1, 2),
        cswap(0, 1, [2, 3]),
        toffoli(0, 1, 2)
    }
    .expect("Failed to create circuit with two/three-argument gates");

    assert_eq!(circuit.num_qubits, 5);
    assert_eq!(circuit.gates.len(), 5);
}

#[test]
fn test_circuit_macro_two_three_qubit_gates_failure() {
    let circuit = circuit! {
        qubits: 5,
        cnot(0, 1),
        swap(1, 2),
        cswap(0, 1, 2),
        cswap(0, 1, [2, 6]), // Invalid qubit index
        toffoli(0, 1, 2)
    };

    assert!(circuit.is_err());
    assert_eq!(circuit.err().unwrap(), Error::InvalidQubitIndex(6, 5));
}

#[test]
fn test_circuit_macro_controlled_gates_success() {
    let circuit = circuit! {
        qubits: 4,
        ch(0, 1),
        ch([0, 1], 2),
        ch(0, [1, 2]),
        ch([0, 1], [2, 3]),

        cx(0, 1),
        cx([0, 1], 2),
        cx(0, [1, 2]),
        cx([0, 1], [2, 3]),

        cy(0, 1),
        cy([0, 1], 2),
        cy(0, [1, 2]),
        cy([0, 1], [2, 3]),

        cz(0, 1),
        cz([0, 1], 2),
        cz(0, [1, 2]),
        cz([0, 1], [2, 3]),
        
        cs(0, 1),
        cs([0, 1], 2),
        cs(0, [1, 2]),
        cs([0, 1], [2, 3]),

        ct(0, 1),
        ct([0, 1], 2),
        ct(0, [1, 2]),
        ct([0, 1], [2, 3]),

        csdag(0, 1),
        csdag([0, 1], 2),
        csdag(0, [1, 2]),
        csdag([0, 1], [2, 3]),
        
        ctdag(0, 1),
        ctdag([0, 1], 2),
        ctdag(0, [1, 2]),
        ctdag([0, 1], [2, 3]),

        crx(0, 1, PI / 4.0),
        crx([0, 1], 2, PI / 4.0),
        crx(0, [1, 2], PI / 4.0),
        crx([0, 1], [2, 3], PI / 4.0),

        cry(0, 1, PI / 4.0),
        cry([0, 1], 2, PI / 4.0),
        cry(0, [1, 2], PI / 4.0),
        cry([0, 1], [2, 3], PI / 4.0),

        cry_phase(0, 1, PI / 4.0, PI / 2.0),
        cry_phase([0, 1], 2, PI / 4.0, PI / 2.0),
        cry_phase(0, [1, 2], PI / 4.0, PI / 2.0),
        cry_phase([0, 1], [2, 3], PI / 4.0, PI / 2.0),

        cmatchgate(0, 1, PI / 4.0, PI / 2.0, PI / 3.0),
        cmatchgate(0, [1, 2], PI / 4.0, PI / 2.0, PI / 3.0),
        
        crz(0, 1, PI / 4.0),
        crz([0, 1], 2, PI / 4.0),
        crz(0, [1, 2], PI / 4.0),
        crz([0, 1], [2, 3], PI / 4.0),

        cp(0, 1, PI / 4.0),
        cp([0, 1], 2, PI / 4.0),
        cp(0, [1, 2], PI / 4.0),
        cp([0, 1], [2, 3], PI / 4.0)
    }
    .expect("Failed to create circuit with controlled gates");

    assert_eq!(circuit.num_qubits, 4);
    assert_eq!(circuit.gates.len(), 6 * 12);
}

#[test]
fn test_circuit_macro_controlled_gates_failure() {
    let circuit = circuit! {
        qubits: 4,
        ch(0, 1),
        ch([0, 1], 2),
        ch(0, [1, 2]),
        ch([0, 1], [2, 3]),

        cx(0, 1),
        cx([0, 1], 2),
        cx(0, [1, 2]),
        cx([0, 1], [2, 6]), // Invalid qubit index
    };

    assert!(circuit.is_err());
    assert_eq!(circuit.err().unwrap(), Error::InvalidQubitIndex(6, 4));
}

#[test]
fn test_circuit_macro_unitary_cunitary_gates_success() {
    let unitary_matrix = [[0.0.into(), 1.0.into()], [1.0.into(), 0.0.into()]];
    let circuit = circuit! {
        qubits: 4,
        unitary(0, unitary_matrix),
        unitary([0, 1], unitary_matrix),

        cunitary(0, 1, unitary_matrix),
        cunitary([0, 1], 2, unitary_matrix),
        cunitary(0, [1, 2], unitary_matrix),
        cunitary([0, 1], [2, 3], unitary_matrix)
    }
    .expect("Failed to create circuit with unitary gates");

    assert_eq!(circuit.num_qubits, 4);
    assert_eq!(circuit.gates.len(), 9);
}

#[test]
fn test_circuit_macro_unitary_cunitary_gates_failure() {
    let unitary_matrix = [[0.0.into(), 1.0.into()], [1.0.into(), 0.0.into()]];
    let circuit = circuit! {
        qubits: 4,
        unitary(0, unitary_matrix),
        unitary([0, 1], unitary_matrix),

        cunitary(0, 1, unitary_matrix),
        cunitary([0, 1], 2, unitary_matrix),
        cunitary(0, [1, 2], unitary_matrix),
        cunitary([0, 1], [2, 6], unitary_matrix) // Invalid qubit index
    };

    assert!(circuit.is_err());
    assert_eq!(circuit.err().unwrap(), Error::InvalidQubitIndex(6, 4));
}

#[test]
fn test_circuit_macro_measurement_success() {
    let unitary_matrix = [[0.0.into(), 1.0.into()], [1.0.into(), 0.0.into()]];
    let circuit = circuit! {
        qubits: 3,
        measurex(0),
        measurex([1, 2]),
        measurey(0),
        measurey([1, 2]),
        measurez(0),
        measurez([1, 2]),
        measure_custom(0, unitary_matrix),
        measure_custom([1, 2], unitary_matrix)
    }
    .expect("Failed to create circuit with measurement gates");

    assert_eq!(circuit.num_qubits, 3);
    assert_eq!(circuit.gates.len(), 8);
}

#[test]
fn test_circuit_macro_angle_gates_success() {
    let circuit = circuit! {
        qubits: 3,
        rx(0, PI / 2.0),
        ry(1, PI / 2.0),
        rz(2, PI / 2.0),
        p(0, PI / 2.0),
        ry_phase(1, PI / 2.0, PI / 4.0),
        matchgate(0, PI / 2.0, PI / 3.0, PI / 4.0),
        rx([0, 1], PI / 2.0),
        ry([1, 2], PI / 2.0),
        rz([0, 2], PI / 2.0),
        ry_phase([0, 1], PI / 2.0, PI / 4.0),
        p([0, 1, 2], PI / 2.0)
    }
    .expect("Failed to create circuit with angle gates");

    assert_eq!(circuit.num_qubits, 3);
    assert_eq!(circuit.gates.len(), 4 + 2 + 2 + 2 + 3);
}

#[test]
fn test_circuit_macro_match_gates_success() {
    let circuit = circuit! {
        qubits: 4,
        matchgate(0, PI / 2.0, PI / 3.0, PI / 4.0),

        cmatchgate(0, 2, PI / 2.0, PI / 3.0, PI / 4.0),
        cmatchgate(0, [2, 3], PI / 2.0, PI / 3.0, PI / 4.0)
    }
    .expect("Failed to create circuit with match gates");

    assert_eq!(circuit.num_qubits, 4);
    assert_eq!(circuit.gates.len(), 3);
}
