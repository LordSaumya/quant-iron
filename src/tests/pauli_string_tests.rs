use crate::{
    components::{
        gate::Gate, operator::Pauli, pauli_string::{PauliString, SumOp}, state::{ChainableState, State}
    },
    errors::Error,
};

use num_complex::Complex;
use std::collections::HashMap;

#[test]
fn test_pauli_string_new() {
    // Test creating a new PauliString with a coefficient
    let coefficient = Complex::new(1.0, 0.0);
    let pauli_string = PauliString::new(coefficient);

    assert_eq!(pauli_string.coefficient(), coefficient);
    assert_eq!(pauli_string.ops().len(), 0);
}

#[test]
fn test_pauli_string_with_ops_success() {
    // Test creating a new PauliString with a coefficient and operators
    let coefficient: Complex<f64> = Complex::new(1.0, 0.0);
    let mut ops: HashMap<usize, Pauli> = HashMap::new();
    ops.insert(0, Pauli::X);
    ops.insert(1, Pauli::Y);

    let pauli_string: PauliString = PauliString::with_ops(coefficient, ops.clone());

    assert_eq!(pauli_string.coefficient(), coefficient);
    assert_eq!(pauli_string.ops().len(), 2);
    assert_eq!(pauli_string.ops(), &ops);
}

#[test]
fn test_pauli_string_add_op_success() {
    // Test adding a Pauli operator to the PauliString
    let coefficient: Complex<f64> = Complex::new(1.0, 0.0);
    let mut pauli_string: PauliString = PauliString::new(coefficient);

    pauli_string.add_op(0, Pauli::X);
    pauli_string.add_op(1, Pauli::Y);

    assert_eq!(pauli_string.coefficient(), coefficient);
    assert_eq!(pauli_string.ops().len(), 2);
    assert_eq!(pauli_string.ops().get(&0), Some(&Pauli::X));
    assert_eq!(pauli_string.ops().get(&1), Some(&Pauli::Y));
}

#[test]
#[should_panic]
fn test_pauli_string_add_op_panics() {
    let coefficient: Complex<f64> = Complex::new(1.0, 0.0);
    let mut pauli_string: PauliString = PauliString::new(coefficient);
    pauli_string.add_op(0, Pauli::X);
    pauli_string.add_op(0, Pauli::Y); // This should panic
}

#[test]
fn test_pauli_string_with_op_success() {
    let pauli_string: PauliString = PauliString::new(Complex::new(1.0, 0.0))
        .with_op(0, Pauli::X)
        .with_op(1, Pauli::Y);

    assert_eq!(pauli_string.coefficient(), Complex::new(1.0, 0.0));
    assert_eq!(pauli_string.ops().len(), 2);
    assert_eq!(pauli_string.ops().get(&0), Some(&Pauli::X));
    assert_eq!(pauli_string.ops().get(&1), Some(&Pauli::Y));
}

#[test]
#[should_panic]
fn test_pauli_string_with_op_panics() {
    let _pauli_string: PauliString = PauliString::new(Complex::new(1.0, 0.0))
        .with_op(0, Pauli::X)
        .with_op(0, Pauli::Y); // This should panic
}

#[test]
fn test_pauli_string_apply_success_non_empty() {
    let coefficient: Complex<f64> = Complex::new(2.0, 2.0);
    let mut pauli_string: PauliString = PauliString::new(coefficient);
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
    let pauli_string: PauliString = PauliString::new(coefficient);

    let state: State = State::new_basis_n(2, 3).unwrap(); // Example state |11>
    let result: State = pauli_string.apply(&state).unwrap();
    let expected_result: State = state * coefficient; // Just multiply the state by the coefficient

    assert_eq!(result, expected_result);
}

#[test]
fn test_pauli_string_apply_error() {
    let coefficient: Complex<f64> = Complex::new(2.0, 2.0);
    let mut pauli_string: PauliString = PauliString::new(coefficient);
    pauli_string.add_op(0, Pauli::X);
    pauli_string.add_op(1, Pauli::Y);

    let state: State = State::new_basis_n(1, 1).unwrap(); // Example state |1>
    let result: Result<State, Error> = pauli_string.apply(&state);

    assert!(result.is_err());
}

#[test]
fn test_pauli_string_apply_normalised_success_non_empty() {
    let coefficient: Complex<f64> = Complex::new(2.0, 2.0);
    let mut pauli_string: PauliString = PauliString::new(coefficient);
    pauli_string.add_op(0, Pauli::X);
    pauli_string.add_op(1, Pauli::Y);

    let state: State = State::new_basis_n(2, 3).unwrap(); // Example state |11>
    let result: State = pauli_string.apply_normalised(&state).unwrap();
    let expected_result: State = state.x(0).y(1).unwrap(); // Apply X on qubit 0 and Y on qubit 1

    assert_eq!(result, expected_result);
}

#[test]
fn test_pauli_string_apply_normalised_success_empty() {
    let coefficient: Complex<f64> = Complex::new(2.0, 2.0);
    let pauli_string: PauliString = PauliString::new(coefficient);

    let state: State = State::new_basis_n(2, 3).unwrap(); // Example state |11>
    let result: State = pauli_string.apply_normalised(&state).unwrap();
    let expected_result: State = state.clone(); // Just return the state as-is

    assert_eq!(result, expected_result);
}

#[test]
fn test_pauli_string_apply_exp_success_non_empty() {
    let coefficient: Complex<f64> = Complex::new(2.0, 2.0);
    let mut pauli_string: PauliString = PauliString::new(coefficient);
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
    let pauli_string: PauliString = PauliString::new(coefficient);

    let state: State = State::new_basis_n(2, 3).unwrap(); // Example state |11>
    let result: State = pauli_string.apply_exp(&state).unwrap();
    let expected_result: State = state * coefficient.exp(); // Just multiply the state by exp(coefficient)

    assert_eq!(result, expected_result);
}

#[test]
fn test_pauli_string_apply_exp_error() {
    let coefficient: Complex<f64> = Complex::new(2.0, 2.0);
    let mut pauli_string: PauliString = PauliString::new(coefficient);
    pauli_string.add_op(0, Pauli::X);
    pauli_string.add_op(1, Pauli::Y);

    let state: State = State::new_basis_n(1, 1).unwrap(); // Example state |1>
    let result: Result<State, Error> = pauli_string.apply_exp(&state);

    assert!(result.is_err());
}

#[test]
fn test_pauli_string_apply_exp_factor_success_non_empty() {
    let coefficient: Complex<f64> = Complex::new(2.0, 2.0);
    let mut pauli_string: PauliString = PauliString::new(coefficient);
    pauli_string.add_op(0, Pauli::X);
    pauli_string.add_op(1, Pauli::Y);

    let state: State = State::new_basis_n(2, 3).unwrap(); // Example state |11>
    let result: State = pauli_string
        .apply_exp_factor(&state, Complex::new(0.5, 0.0))
        .unwrap();
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
    let pauli_string: PauliString = PauliString::new(coefficient);

    let state: State = State::new_basis_n(2, 3).unwrap(); // Example state |11>
    let result: State = pauli_string
        .apply_exp_factor(&state, Complex::new(0.5, 0.0))
        .unwrap();
    let expected_result: State = state * (coefficient * 0.5).exp(); // Just multiply the state by exp(coefficient * factor)

    assert_eq!(result, expected_result);
}

#[test]
fn test_pauli_string_apply_exp_factor_error() {
    let coefficient: Complex<f64> = Complex::new(2.0, 2.0);
    let mut pauli_string: PauliString = PauliString::new(coefficient);
    pauli_string.add_op(0, Pauli::X);
    pauli_string.add_op(1, Pauli::Y);

    let state: State = State::new_basis_n(1, 1).unwrap(); // Example state |1>
    let result: Result<State, Error> =
        pauli_string.apply_exp_factor(&state, Complex::new(0.5, 0.0));

    assert!(result.is_err());
}

#[test]
fn test_pauli_string_hermitian_conjugate() {
    // Test the hermitian_conjugate method
    let coefficient: Complex<f64> = Complex::new(2.0, 2.0);
    let mut pauli_string: PauliString = PauliString::new(coefficient);
    pauli_string.add_op(0, Pauli::X);
    pauli_string.add_op(1, Pauli::Y);

    let hermitian_conjugate: PauliString = pauli_string.hermitian_conjugate();

    assert_eq!(hermitian_conjugate.coefficient(), Complex::new(2.0, -2.0)); // Coefficient should be conjugated
    assert_eq!(hermitian_conjugate.ops().len(), 2);
    assert_eq!(hermitian_conjugate.ops().get(&0), Some(&Pauli::X));
    assert_eq!(hermitian_conjugate.ops().get(&1), Some(&Pauli::Y));
}

#[test]
fn test_sumop_new() {
    let terms = vec![
        PauliString::new(Complex::new(1.0, 0.0)),
        PauliString::new(Complex::new(2.0, 0.0)),
    ];

    let sum_op = SumOp::new(terms.clone());
    assert_eq!(sum_op.terms.len(), 2);
}

#[test]
fn test_sumop_num_terms() {
    let terms: Vec<PauliString> = vec![
        PauliString::new(Complex::new(1.0, 0.0)),
        PauliString::new(Complex::new(2.0, 0.0)),
    ];

    let sum_op: SumOp = SumOp::new(terms.clone());
    assert_eq!(sum_op.num_terms(), 2);
}

#[test]
fn test_sumop_add_term() {
    let terms: Vec<PauliString> = vec![
        PauliString::new(Complex::new(1.0, 0.0)),
        PauliString::new(Complex::new(2.0, 0.0)),
    ];

    let new_term: PauliString = PauliString::new(Complex::new(3.0, 0.0));

    let mut sum_op: SumOp = SumOp::new(terms.clone());
    sum_op.add_term(new_term.clone());

    assert_eq!(sum_op.num_terms(), 3);
}

#[test]
fn test_sumop_apply_non_empty_success() {
    let mut pauli_string_1 = PauliString::new(Complex::new(2.0, 0.0));
    pauli_string_1.add_op(0, Pauli::X);

    let mut pauli_string_2 = PauliString::new(Complex::new(3.0, 0.0));
    pauli_string_2.add_op(1, Pauli::Y);

    let sum_op = SumOp::new(vec![pauli_string_1, pauli_string_2]);
    let state: State = State::new_basis_n(2, 3).unwrap(); // Example state |11>
    let result: State = sum_op.apply(&state).unwrap();
    let expected_result: State = 2.0 * state.x(0).unwrap() + 3.0 * state.y(1).unwrap();

    assert_eq!(result, expected_result);
}

#[test]
fn test_sumop_apply_error() {
    let mut pauli_string_1 = PauliString::new(Complex::new(2.0, 0.0));
    pauli_string_1.add_op(0, Pauli::X);

    let mut pauli_string_2 = PauliString::new(Complex::new(3.0, 0.0));
    pauli_string_2.add_op(1, Pauli::Y);

    let sum_op = SumOp::new(vec![pauli_string_1, pauli_string_2]);
    let state: State = State::new_basis_n(1, 1).unwrap(); // Example state |1>
    let result: Result<State, Error> = sum_op.apply(&state);

    assert!(result.is_err());
}

#[test]
fn test_sumop_apply_empty_success() {
    let sum_op = SumOp::new(vec![]);
    let state: State = State::new_basis_n(2, 3).unwrap(); // Example state |11>
    let result: State = sum_op.apply(&state).unwrap();
    let expected_result: State = state.clone() * 0.0; // Should result in the zero state

    assert_eq!(result, expected_result);
}

#[test]
fn test_sumop_expectation_value() {
    let mut pauli_string_1 = PauliString::new(Complex::new(2.0, 0.0));
    pauli_string_1.add_op(0, Pauli::X);

    let mut pauli_string_2 = PauliString::new(Complex::new(3.0, 0.0));
    pauli_string_2.add_op(1, Pauli::Y);

    let mut pauli_string_3 = PauliString::new(Complex::new(4.0, 0.0));
    pauli_string_3.add_op(1, Pauli::Z);

    let sum_op: SumOp = SumOp::new(vec![pauli_string_1, pauli_string_2, pauli_string_3]);
    let state: State = State::new_basis_n(2, 3).unwrap(); // Example state |11>
    let result: Complex<f64> = sum_op.expectation_value(&state).unwrap();
    let expected_result: Complex<f64> = Complex::new(-4.0, 0.0); // Expectation value of X on qubit 0 and Y on qubit 1

    assert_eq!(result, expected_result);
}

#[test]
fn test_pauli_string_get_targets() {
    let mut pauli_string: PauliString = PauliString::new(Complex::new(1.0, 0.0));
    pauli_string.add_op(2, Pauli::X);
    pauli_string.add_op(0, Pauli::Y);
    pauli_string.add_op(1, Pauli::Z);

    let mut targets: Vec<usize> = pauli_string.get_targets();
    targets.sort();

    let expected_targets: Vec<usize> = vec![0, 1, 2];

    assert_eq!(targets, expected_targets);
}

#[test]
fn test_pauli_string_to_gates() {
    let mut pauli_string: PauliString = PauliString::new(Complex::new(1.0, 0.0));
    pauli_string.add_op(2, Pauli::X);
    pauli_string.add_op(0, Pauli::Y);
    pauli_string.add_op(1, Pauli::Z);

    let gates = pauli_string.to_gates();
    assert_eq!(gates.len(), 3);

    let expected_gates = vec![
        Gate::x_gate(2),
        Gate::y_gate(0),
        Gate::z_gate(1),
    ];

    let gates_strings = gates.iter().map(|g| format!("{:?}", g)).collect::<Vec<String>>();
    let expected_gates_strings = expected_gates.iter().map(|g| format!("{:?}", g)).collect::<Vec<String>>();

    for str in gates_strings {
        assert!(expected_gates_strings.contains(&str));
    }
}