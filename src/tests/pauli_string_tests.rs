use crate::{
    components::{
        operator::Pauli,
        pauli_string::PauliString,
        state::{ChainableState, State},
    },
    errors::Error,
};

use num_complex::Complex;
use std::collections::HashMap;

#[test]
fn test_pauli_string_new() {
    // Test creating a new PauliString with a coefficient
    let coefficient = Complex::new(1.0, 0.0);
    let pauli_string = PauliString::new(coefficient.clone());

    assert_eq!(pauli_string.coefficient(), coefficient);
    assert_eq!(pauli_string.ops().len(), 0);
}

#[test]
fn test_pauli_string_with_ops() {
    // Test creating a new PauliString with a coefficient and operators
    let coefficient: Complex<f64> = Complex::new(1.0, 0.0);
    let mut ops: HashMap<usize, Pauli> = HashMap::new();
    ops.insert(0, Pauli::X);
    ops.insert(1, Pauli::Y);

    let pauli_string: PauliString = PauliString::with_ops(coefficient.clone(), ops.clone());

    assert_eq!(pauli_string.coefficient(), coefficient);
    assert_eq!(pauli_string.ops().len(), 2);
    assert_eq!(pauli_string.ops(), &ops);
}

#[test]
fn test_pauli_string_add_op() {
    // Test adding a Pauli operator to the PauliString
    let coefficient: Complex<f64> = Complex::new(1.0, 0.0);
    let mut pauli_string: PauliString = PauliString::new(coefficient.clone());

    pauli_string.add_op(0, Pauli::X);
    pauli_string.add_op(1, Pauli::Y);

    assert_eq!(pauli_string.coefficient(), coefficient);
    assert_eq!(pauli_string.ops().len(), 2);
    assert_eq!(pauli_string.ops().get(&0), Some(&Pauli::X));
    assert_eq!(pauli_string.ops().get(&1), Some(&Pauli::Y));
}

#[test]
fn test_pauli_string_apply_success_non_empty() {
    let coefficient: Complex<f64> = Complex::new(2.0, 2.0);
    let mut pauli_string: PauliString = PauliString::new(coefficient.clone());
    pauli_string.add_op(0, Pauli::X);
    pauli_string.add_op(1, Pauli::Y);


    let state: State = State::new_basis_n(2, 3).unwrap(); // Example state |11>
    let result: State = pauli_string.apply(&state).unwrap();
    let expected_result: State = state.x(0).y(1).unwrap() * coefficient; // Apply X on qubit 0 and Y on qubit 1

    assert_eq!(result, expected_result);
}

#[test]
fn test_pauli_string_apply_success_empty() {
    let coefficient: Complex<f64> = Complex::new(2.0, 2.0);
    let pauli_string: PauliString = PauliString::new(coefficient.clone());

    let state: State = State::new_basis_n(2, 3).unwrap(); // Example state |11>
    let result: State = pauli_string.apply(&state).unwrap();
    let expected_result: State = state * coefficient; // Just multiply the state by the coefficient

    assert_eq!(result, expected_result);
}

#[test]
fn test_pauli_string_apply_error() {
    let coefficient: Complex<f64> = Complex::new(2.0, 2.0);
    let mut pauli_string: PauliString = PauliString::new(coefficient.clone());
    pauli_string.add_op(0, Pauli::X);
    pauli_string.add_op(1, Pauli::Y);

    let state: State = State::new_basis_n(1, 1).unwrap(); // Example state |1>
    let result: Result<State, Error> = pauli_string.apply(&state);

    assert!(result.is_err());
}

#[test]
fn test_pauli_string_apply_exp_success_non_empty() {
    let coefficient: Complex<f64> = Complex::new(2.0, 2.0);
    let mut pauli_string: PauliString = PauliString::new(coefficient.clone());
    pauli_string.add_op(0, Pauli::X);
    pauli_string.add_op(1, Pauli::Y);

    let state: State = State::new_basis_n(2, 3).unwrap(); // Example state |11>
    let result: State = pauli_string.apply_exp(&state).unwrap();
    let alpha: Complex<f64> = coefficient;

    // Calculate the expected result using the formula exp(alpha * P_ops) = cosh(alpha)*I + sinh(alpha)*P_ops
    let cosh_alpha: Complex<f64> = alpha.cosh();
    let sinh_alpha: Complex<f64> = alpha.sinh();
    let expected_result: State = state.clone() * cosh_alpha + state.x(0).y(1).unwrap() * sinh_alpha;

    assert_eq!(result, expected_result);
}

#[test]
fn test_pauli_string_apply_exp_success_empty() {
    let coefficient: Complex<f64> = Complex::new(2.0, 2.0);
    let pauli_string: PauliString = PauliString::new(coefficient.clone());

    let state: State = State::new_basis_n(2, 3).unwrap(); // Example state |11>
    let result: State = pauli_string.apply_exp(&state).unwrap();
    let expected_result: State = state * coefficient.exp(); // Just multiply the state by exp(coefficient)

    assert_eq!(result, expected_result);
}

#[test]
fn test_pauli_string_apply_exp_error() {
    let coefficient: Complex<f64> = Complex::new(2.0, 2.0);
    let mut pauli_string: PauliString = PauliString::new(coefficient.clone());
    pauli_string.add_op(0, Pauli::X);
    pauli_string.add_op(1, Pauli::Y);

    let state: State = State::new_basis_n(1, 1).unwrap(); // Example state |1>
    let result: Result<State, Error> = pauli_string.apply_exp(&state);

    assert!(result.is_err());
}

#[test]
fn test_pauli_string_apply_exp_factor_success_non_empty() {
    let coefficient: Complex<f64> = Complex::new(2.0, 2.0);
    let mut pauli_string: PauliString = PauliString::new(coefficient.clone());
    pauli_string.add_op(0, Pauli::X);
    pauli_string.add_op(1, Pauli::Y);

    let state: State = State::new_basis_n(2, 3).unwrap(); // Example state |11>
    let result: State = pauli_string.apply_exp_factor(&state, Complex::new(0.5, 0.0)).unwrap();
    let alpha: Complex<f64> = coefficient;

    // Calculate the expected result using the formula exp(alpha * P_ops) = cosh(alpha * factor)*I + sinh(alpha * factor)*P_ops
    let cosh_alpha: Complex<f64> = (alpha * 0.5).cosh();
    let sinh_alpha: Complex<f64> = (alpha * 0.5).sinh();
    let expected_result: State = state.clone() * cosh_alpha + state.x(0).y(1).unwrap() * sinh_alpha;

    assert_eq!(result, expected_result);
}

#[test]
fn test_pauli_string_apply_exp_factor_success_empty() {
    let coefficient: Complex<f64> = Complex::new(2.0, 2.0);
    let pauli_string: PauliString = PauliString::new(coefficient.clone());

    let state: State = State::new_basis_n(2, 3).unwrap(); // Example state |11>
    let result: State = pauli_string.apply_exp_factor(&state, Complex::new(0.5, 0.0)).unwrap();
    let expected_result: State = state * (coefficient * 0.5).exp(); // Just multiply the state by exp(coefficient * factor)

    assert_eq!(result, expected_result);
}

#[test]
fn test_pauli_string_apply_exp_factor_error() {
    let coefficient: Complex<f64> = Complex::new(2.0, 2.0);
    let mut pauli_string: PauliString = PauliString::new(coefficient.clone());
    pauli_string.add_op(0, Pauli::X);
    pauli_string.add_op(1, Pauli::Y);

    let state: State = State::new_basis_n(1, 1).unwrap(); // Example state |1>
    let result: Result<State, Error> = pauli_string.apply_exp_factor(&state, Complex::new(0.5, 0.0));

    assert!(result.is_err());
}

#[test]
fn test_pauli_string_hermitian_conjugate() {
    // Test the hermitian_conjugate method
    let coefficient: Complex<f64> = Complex::new(2.0, 2.0);
    let mut pauli_string: PauliString = PauliString::new(coefficient.clone());
    pauli_string.add_op(0, Pauli::X);
    pauli_string.add_op(1, Pauli::Y);

    let hermitian_conjugate: PauliString = pauli_string.hermitian_conjugate();

    assert_eq!(hermitian_conjugate.coefficient(), Complex::new(2.0, -2.0)); // Coefficient should be conjugated
    assert_eq!(hermitian_conjugate.ops().len(), 2);
    assert_eq!(hermitian_conjugate.ops().get(&0), Some(&Pauli::X));
    assert_eq!(hermitian_conjugate.ops().get(&1), Some(&Pauli::Y));
}