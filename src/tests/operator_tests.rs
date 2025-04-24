use crate::{
    components::{
        ChainableState,
        operator::{
            Hadamard,
            Identity,
            Operator,
            Pauli,
            PhaseS,
            PhaseT,
        },
        state::State,
    },
    errors::Error,
};
use num_complex::Complex;
use std::f64::consts::PI;

#[test]
fn test_state_hadamard_success() {
    // h(|0>) = |+>
    // h(|1>) = |->
    // h(|+>) = |0>
    // h(|->) = |1>
    let zero_state: State = State::new_zero(1).unwrap();
    let one_state: State = State::new_basis_n(1, 1).unwrap();
    let plus_state: State = State::new_plus(1).unwrap();
    let minus_state: State = State::new_minus(1).unwrap();

    assert_eq!(zero_state.h(0).unwrap(), plus_state);
    assert_eq!(one_state.h(0).unwrap(), minus_state);
    assert_eq!(plus_state.h(0).unwrap(), zero_state);
    assert_eq!(minus_state.h(0).unwrap(), one_state);

    // h(|00>) = |++>
    let two_qubit_state: State = State::new_zero(2).unwrap();
    let new_state: State = two_qubit_state.h_multi(&[0, 1]).unwrap();
    let expected_state: State = State::new_plus(2).unwrap();

    assert_eq!(new_state, expected_state);

    // Base qubits = 1
    assert_eq!(Hadamard {}.base_qubits(), 1);
}

#[test]
fn test_state_pauli_x_success() {
    // x(|0>) = |1>
    // x(|1>) = |0>
    // x(|+>) = |+> (|+> is an eigenstate of X with eigenvalue = +1)
    // x(|->) = -|-> (|-> is an eigenstate of X with eigenvalue = -1)
    let zero_state: State = State::new_zero(1).unwrap();
    let one_state: State = State::new_basis_n(1, 1).unwrap();
    let plus_state: State = State::new_plus(1).unwrap();
    let minus_state: State = State::new_minus(1).unwrap();

    assert_eq!(zero_state.x(0).unwrap(), one_state);
    assert_eq!(one_state.x(0).unwrap(), zero_state);
    assert_eq!(plus_state.x(0).unwrap(), plus_state);
    assert_eq!(minus_state.x(0).unwrap(), minus_state * -1.0);

    // x(|00>) = |11>
    let two_qubit_state: State = State::new_zero(2).unwrap();
    let new_state: State = two_qubit_state.x_multi(&[0, 1]).unwrap();
    let expected_state: State = State::new_basis_n(2, 3).unwrap(); // |11> in 2 qubits
    assert_eq!(new_state, expected_state);

    // x(x(|0>)) = |0> (Pauli-X matrix is self-inverse)
    let new_state: State = zero_state.x(0).x(0).unwrap();
    assert_eq!(new_state, zero_state);

    // Base qubits = 1
    assert_eq!(Pauli::X.base_qubits(), 1);
}

#[test]
fn test_state_pauli_y_success() {
    //  y(|0>) = i|1>
    //  y(|1>) = -i|0>
    //  y(|+>) = -i|->
    //  y(|->) = i|+>
    let zero_state: State = State::new_zero(1).unwrap();
    let one_state: State = State::new_basis_n(1, 1).unwrap();
    let plus_state: State = State::new_plus(1).unwrap();
    let minus_state: State = State::new_minus(1).unwrap();

    let i: Complex<f64> = Complex::new(0.0, 1.0);
    let neg_i: Complex<f64> = Complex::new(0.0, -1.0);
    let neg_one: Complex<f64> = Complex::new(-1.0, 0.0);

    assert_eq!(zero_state.y(0).unwrap(), i * one_state.clone());
    assert_eq!(one_state.y(0).unwrap(), neg_i * zero_state.clone());
    assert_eq!(plus_state.y(0).unwrap(), neg_i * minus_state.clone());
    assert_eq!(minus_state.y(0).unwrap(), i * plus_state.clone());

    // y(|00>) = -|11>
    let two_qubit_state: State = State::new_zero(2).unwrap();
    let new_state: State = two_qubit_state.y_multi(&[0, 1]).unwrap();
    let expected_state: State = State::new_basis_n(2, 3).unwrap() * neg_one;
    assert_eq!(new_state, expected_state);

    // y(y(|0>)) = |0> (Pauli-Y matrix is self-inverse)
    let new_state: State = zero_state.y(0).y(0).unwrap();
    assert_eq!(new_state, zero_state);

    // Base qubits = 1
    assert_eq!(Pauli::Y.base_qubits(), 1);
}

#[test]
fn test_state_pauli_success() {
    // z(|0>) = |0> (|0> is an eigenstate of Z with eigenvalue = +1)
    // z(|1>) = -|1> (|1> is an eigenstate of Z with eigenvalue = -1)
    // z(|+>) = |->
    // z(|->) = |+>
    let zero_state: State = State::new_zero(1).unwrap();
    let one_state: State = State::new_basis_n(1, 1).unwrap();
    let plus_state: State = State::new_plus(1).unwrap();
    let minus_state: State = State::new_minus(1).unwrap();

    assert_eq!(zero_state.z(0).unwrap(), zero_state.clone());
    assert_eq!(one_state.z(0).unwrap(), one_state.clone() * -1.0);
    assert_eq!(plus_state.z(0).unwrap(), minus_state.clone());
    assert_eq!(minus_state.z(0).unwrap(), plus_state.clone());

    // z(|00>) = |00>
    let two_qubit_state: State = State::new_zero(2).unwrap();
    let new_state: State = two_qubit_state.z_multi(&[0, 1]).unwrap();
    let expected_state: State = State::new_zero(2).unwrap();
    assert_eq!(new_state, expected_state);

    // z(z(|+>)) = |+> (Pauli-Z matrix is self-inverse)
    let new_state: State = plus_state.z(0).z(0).unwrap();
    assert_eq!(new_state, plus_state);

    // Base qubits = 1
    assert_eq!(Pauli::Z.base_qubits(), 1);
}

#[test]
fn test_state_identity_success() {
    // i(|0>) = |0>
    // i(|1>) = |1>
    // i(|+>) = |+>
    // i(|->) = |->
    let zero_state: State = State::new_zero(1).unwrap();
    let one_state: State = State::new_basis_n(1, 1).unwrap();
    let plus_state: State = State::new_plus(1).unwrap();
    let minus_state: State = State::new_minus(1).unwrap();

    assert_eq!(zero_state.i(0).unwrap(), zero_state.clone());
    assert_eq!(one_state.i(0).unwrap(), one_state.clone());
    assert_eq!(plus_state.i(0).unwrap(), plus_state.clone());
    assert_eq!(minus_state.i(0).unwrap(), minus_state.clone());

    // i(|00>) = |00>
    let two_qubit_state: State = State::new_zero(2).unwrap();
    let new_state: State = two_qubit_state.i_multi(&[0, 1]).unwrap();
    let expected_state: State = State::new_zero(2).unwrap();
    assert_eq!(new_state, expected_state);

    // Base qubits = 1
    assert_eq!(Identity {}.base_qubits(), 1);
}

#[test]
fn test_state_phase_s_success() {
    // s(|0>) = |0>
    // s(|1>) = i|1>
    // s(|+>) = 1/sqrt(2)(|0> + i|1>)
    // s(|->) = 1/sqrt(2)(|0> - i|1>)
    let zero_state: State = State::new_zero(1).unwrap();
    let one_state: State = State::new_basis_n(1, 1).unwrap();
    let plus_state: State = State::new_plus(1).unwrap();
    let minus_state: State = State::new_minus(1).unwrap();

    let i: Complex<f64> = Complex::new(0.0, 1.0);
    let invrt2: Complex<f64> = Complex::new(1.0 / 2.0_f64.sqrt(), 0.0);
    let expected_plus: State = invrt2 * (zero_state.clone() + i * one_state.clone());
    let expected_minus: State = invrt2 * (zero_state.clone() - i * one_state.clone());

    assert_eq!(zero_state.s(0).unwrap(), zero_state.clone());
    assert_eq!(one_state.s(0).unwrap(), i * one_state.clone());
    assert_eq!(plus_state.s(0).unwrap(), expected_plus.clone());
    assert_eq!(minus_state.s(0).unwrap(), expected_minus.clone());

    // s(|00>) = |00>
    let two_qubit_state: State = State::new_zero(2).unwrap();
    let new_state: State = two_qubit_state.s_multi(&[0, 1]).unwrap();
    let expected_state: State = State::new_zero(2).unwrap();
    assert_eq!(new_state, expected_state);

    // Base qubits = 1
    assert_eq!(PhaseS{}.base_qubits(), 1);
}

#[test]
fn test_state_phase_t_success() {
    // t(|0>) = |0>
    // t(|1>) = e^(i*PI/4)|1>
    // t(|+>) = 1/sqrt(2)(|0> + e^(i*PI/4)|1>)
    // t(|->) = 1/sqrt(2)(|0> - e^(i*PI/4)|1>)
    let zero_state: State = State::new_zero(1).unwrap();
    let one_state: State = State::new_basis_n(1, 1).unwrap();
    let plus_state: State = State::new_plus(1).unwrap();
    let minus_state: State = State::new_minus(1).unwrap();

    let eipi4: Complex<f64> = Complex::new(0.0, PI / 4.0).exp();
    let invrt2: Complex<f64> = Complex::new(1.0 / 2.0_f64.sqrt(), 0.0);
    let expected_plus: State = invrt2 * (zero_state.clone() + eipi4 * one_state.clone());
    let expected_minus: State = invrt2 * (zero_state.clone() - eipi4 * one_state.clone());

    assert_eq!(zero_state.t(0).unwrap(), zero_state.clone());
    assert_eq!(one_state.t(0).unwrap(), eipi4 * one_state.clone());
    assert_eq!(plus_state.t(0).unwrap(), expected_plus.clone());
    assert_eq!(minus_state.t(0).unwrap(), expected_minus.clone());

    // t(|00>) = |00>
    let two_qubit_state: State = State::new_zero(2).unwrap();
    let new_state: State = two_qubit_state.t_multi(&[0, 1]).unwrap();
    let expected_state: State = State::new_zero(2).unwrap();
    assert_eq!(new_state, expected_state);

    // Base qubits = 1
    assert_eq!(PhaseT{}.base_qubits(), 1);
}

// -- TEST ALL ERRORS --

// Macro to generate error test logic for single-qubit gates
macro_rules! generate_single_qubit_gate_error_assertions {
    // Match gates without angles
    ($state:ident, $gate_method:ident) => {
        paste::paste! { // Use paste crate to concatenate identifiers
            // Test single application with invalid index
            let result_single = $state.$gate_method(2); // Index 2 is invalid for a 2-qubit state
            assert!(matches!(result_single, Err(Error::InvalidQubitIndex(2, 2))),
                    "Gate '{}' single failed: Expected InvalidQubitIndex(2, 2), got {:?}", stringify!($gate_method), result_single);

            // Test multi application with invalid index
            let result_multi = $state.[<$gate_method _multi>](&[0, 2]); // Index 2 is invalid
            assert!(matches!(result_multi, Err(Error::InvalidQubitIndex(2, 2))),
                    "Gate '{}' multi failed: Expected InvalidQubitIndex(2, 2), got {:?}", stringify!($gate_method), result_multi);
        }
    };
    // Match gates with angles
    ($state:ident, $gate_method:ident, $angle:expr) => {
        paste::paste! { // Use paste crate to concatenate identifiers
            // Test single application with invalid index
            let result_single = $state.$gate_method(2, $angle); // Index 2 is invalid for a 2-qubit state
            assert!(matches!(result_single, Err(Error::InvalidQubitIndex(2, 2))),
                    "Gate '{}' single failed: Expected InvalidQubitIndex(2, 2), got {:?}", stringify!($gate_method), result_single);

            // Test multi application with invalid index
            let result_multi = $state.[<$gate_method _multi>](&[0, 2], $angle); // Index 2 is invalid
            assert!(matches!(result_multi, Err(Error::InvalidQubitIndex(2, 2))),
                    "Gate '{}' multi failed: Expected InvalidQubitIndex(2, 2), got {:?}", stringify!($gate_method), result_multi);
        }
    };
}

#[test]
fn test_single_qubit_gate_errors() {
    let state = State::new_zero(2).unwrap();
    let angle = PI / 4.0; // Example angle

    // Instantiate macro for each single-qubit gate
    generate_single_qubit_gate_error_assertions!(state, h);
    generate_single_qubit_gate_error_assertions!(state, x);
    generate_single_qubit_gate_error_assertions!(state, y);
    generate_single_qubit_gate_error_assertions!(state, z);
    generate_single_qubit_gate_error_assertions!(state, s);
    generate_single_qubit_gate_error_assertions!(state, t);
    generate_single_qubit_gate_error_assertions!(state, s_dag);
    generate_single_qubit_gate_error_assertions!(state, t_dag);
    generate_single_qubit_gate_error_assertions!(state, i);

    // Instantiate for gates requiring an angle
    generate_single_qubit_gate_error_assertions!(state, p, angle);
    generate_single_qubit_gate_error_assertions!(state, rx, angle);
    generate_single_qubit_gate_error_assertions!(state, ry, angle);
    generate_single_qubit_gate_error_assertions!(state, rz, angle);
}

#[test]
fn test_multi_qubit_gate_errors() {
    let num_qubits = 3;
    let invalid_index = 3; // Index 3 is out of bounds for a 3-qubit state (0, 1, 2)
    let state = State::new_zero(num_qubits).unwrap();

    // --- CNOT Gate Index Errors ---
    // Invalid control index
    let cnot_invalid_control = state.cnot(invalid_index, 0);
    assert!(
        matches!(
            cnot_invalid_control,
            Err(Error::InvalidQubitIndex(_invalid_index, _num_qubits))
        ),
        "CNOT failed (invalid control): Expected InvalidQubitIndex({}, {}), got {:?}",
        invalid_index,
        num_qubits,
        cnot_invalid_control
    );
    // Invalid target index
    let cnot_invalid_target = state.cnot(0, invalid_index);
    assert!(
        matches!(
            cnot_invalid_target,
            Err(Error::InvalidQubitIndex(_invalid_index, _num_qubits))
        ),
        "CNOT failed (invalid target): Expected InvalidQubitIndex({}, {}), got {:?}",
        invalid_index,
        num_qubits,
        cnot_invalid_target
    );

    // --- SWAP Gate Index Errors ---
    // Invalid first qubit index
    let swap_invalid_q1 = state.swap(invalid_index, 1);
    assert!(
        matches!(
            swap_invalid_q1,
            Err(Error::InvalidQubitIndex(_invalid_index, _num_qubits))
        ),
        "SWAP failed (invalid qubit1): Expected InvalidQubitIndex({}, {}), got {:?}",
        invalid_index,
        num_qubits,
        swap_invalid_q1
    );
    // Invalid second qubit index
    let swap_invalid_q2 = state.swap(0, invalid_index);
    assert!(
        matches!(
            swap_invalid_q2,
            Err(Error::InvalidQubitIndex(_invalid_index, _num_qubits))
        ),
        "SWAP failed (invalid qubit2): Expected InvalidQubitIndex({}, {}), got {:?}",
        invalid_index,
        num_qubits,
        swap_invalid_q2
    );

    // --- Toffoli Gate Index Errors ---
    // Invalid first control index
    let toffoli_invalid_c1 = state.toffoli(invalid_index, 1, 2);
    assert!(
        matches!(
            toffoli_invalid_c1,
            Err(Error::InvalidQubitIndex(_invalid_index, _num_qubits))
        ),
        "Toffoli failed (invalid control1): Expected InvalidQubitIndex({}, {}), got {:?}",
        invalid_index,
        num_qubits,
        toffoli_invalid_c1
    );
    // Invalid second control index
    let toffoli_invalid_c2 = state.toffoli(0, invalid_index, 2);
    assert!(
        matches!(
            toffoli_invalid_c2,
            Err(Error::InvalidQubitIndex(_invalid_index, _num_qubits))
        ),
        "Toffoli failed (invalid control2): Expected InvalidQubitIndex({}, {}), got {:?}",
        invalid_index,
        num_qubits,
        toffoli_invalid_c2
    );
    // Invalid target index
    let toffoli_invalid_target = state.toffoli(0, 1, invalid_index);
    assert!(
        matches!(
            toffoli_invalid_target,
            Err(Error::InvalidQubitIndex(_invalid_index, _num_qubits))
        ),
        "Toffoli failed (invalid target): Expected InvalidQubitIndex({}, {}), got {:?}",
        invalid_index,
        num_qubits,
        toffoli_invalid_target
    );
}
