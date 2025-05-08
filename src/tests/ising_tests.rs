use crate::{
    components::{
        operator::Pauli,
        pauli_string::{PauliString, SumOp},
        state::State,
    },
    errors::Error,
    models::ising::{ising_1d, ising_1d_uniform, ising_2d, ising_2d_uniform},
};

use num_complex::Complex;

#[test]
fn test_ising_1d_success() {
    // 3 qubit system
    let h: [f64; 3] = [1.0, 2.0, 3.0];
    let j: [f64; 3] = [0.5, 1.0, 1.5];
    let m: f64 = 0.1;

    let result: SumOp = ising_1d(h, j, m).unwrap();
    assert_eq!(result.terms.len(), 6);

    let expected_terms: Vec<PauliString> = vec![
        // Field terms
        m * h[0] * PauliString::new(Complex::new(-1.0, 0.0)).with_op(0, Pauli::Z),
        m * h[1] * PauliString::new(Complex::new(-1.0, 0.0)).with_op(1, Pauli::Z),
        m * h[2] * PauliString::new(Complex::new(-1.0, 0.0)).with_op(2, Pauli::Z),
        // Coupling terms
        j[0] * PauliString::new(Complex::new(-1.0, 0.0))
            .with_op(0, Pauli::Z)
            .with_op(1, Pauli::Z),
        j[1] * PauliString::new(Complex::new(-1.0, 0.0))
            .with_op(1, Pauli::Z)
            .with_op(2, Pauli::Z),
        j[2] * PauliString::new(Complex::new(-1.0, 0.0))
            .with_op(2, Pauli::Z)
            .with_op(0, Pauli::Z),
    ];

    let state: State = State::new_basis_n(3, 6).unwrap();
    let result_state: State = result.apply(&state).unwrap();
    let expected_state: State = SumOp::new(expected_terms).apply(&state).unwrap();

    assert_eq!(result_state, expected_state);
}

#[test]
fn test_ising_1d_error() {
    // 1 qubit system (invalid case)
    let h: [f64; 1] = [1.0];
    let j: [f64; 1] = [0.5];
    let m: f64 = 0.1;

    let result: Result<SumOp, Error> = ising_1d(h, j, m);
    assert_eq!(result.unwrap_err(), Error::InvalidNumberOfInputs(1, 2));
}

#[test]
fn test_ising_1d_uniform_success() {
    // 3 qubit system
    let h: f64 = 1.0;
    let j: f64 = 2.0;
    let m: f64 = 0.1;

    let result: SumOp = ising_1d_uniform(3, h, j, m).unwrap();
    assert_eq!(result.terms.len(), 6);

    let expected_terms: Vec<PauliString> = vec![
        // Field terms
        m * h * PauliString::new(Complex::new(-1.0, 0.0)).with_op(0, Pauli::Z),
        m * h * PauliString::new(Complex::new(-1.0, 0.0)).with_op(1, Pauli::Z),
        m * h * PauliString::new(Complex::new(-1.0, 0.0)).with_op(2, Pauli::Z),
        // Coupling terms
        j * PauliString::new(Complex::new(-1.0, 0.0))
            .with_op(0, Pauli::Z)
            .with_op(1, Pauli::Z),
        j * PauliString::new(Complex::new(-1.0, 0.0))
            .with_op(1, Pauli::Z)
            .with_op(2, Pauli::Z),
        j * PauliString::new(Complex::new(-1.0, 0.0))
            .with_op(2, Pauli::Z)
            .with_op(0, Pauli::Z),
    ];

    let state: State = State::new_basis_n(3, 6).unwrap();
    let result_state: State = result.apply(&state).unwrap();
    let expected_state: State = SumOp::new(expected_terms).apply(&state).unwrap();

    assert_eq!(result_state, expected_state);
}

#[test]
fn test_ising_1d_uniform_error() {
    // 1 qubit system (invalid case)
    let h: f64 = 1.0;
    let j: f64 = 2.0;
    let m: f64 = 0.1;

    let result: Result<SumOp, Error> = ising_1d_uniform(1, h, j, m);
    assert_eq!(result.unwrap_err(), Error::InvalidNumberOfInputs(1, 2));
}

#[test]
fn test_ising_2d_success() {
    // 3x3 qubit system
    let h: [[f64; 3]; 3] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
    let j: [[[f64; 2]; 3]; 3] = [
        [[0.5, 1.0], [1.5, 2.0], [2.5, 3.0]],
        [[3.5, 4.0], [4.5, 5.0], [5.5, 6.0]],
        [[6.5, 7.0], [7.5, 8.0], [8.5, 9.0]],
    ];
    let m: f64 = 0.1;

    let result: SumOp = ising_2d(h, j, m).unwrap();
    println!("Result: {}", result);
    assert_eq!(result.terms.len(), 27);

    let expected_terms: Vec<PauliString> = vec![
        // Site (0,0) - Qubit 0
        PauliString::new(Complex::new(-0.1 * 1.0, 0.0)).with_op(0, Pauli::Z), // Field Z0
        PauliString::new(Complex::new(-0.5, 0.0))
            .with_op(0, Pauli::Z)
            .with_op(3, Pauli::Z), // Vertical Z0Z3
        PauliString::new(Complex::new(-1.0, 0.0))
            .with_op(0, Pauli::Z)
            .with_op(1, Pauli::Z), // Horizontal Z0Z1
        // Site (0,1) - Qubit 1
        PauliString::new(Complex::new(-0.1 * 2.0, 0.0)).with_op(1, Pauli::Z), // Field Z1
        PauliString::new(Complex::new(-1.5, 0.0))
            .with_op(1, Pauli::Z)
            .with_op(4, Pauli::Z), // Vertical Z1Z4
        PauliString::new(Complex::new(-2.0, 0.0))
            .with_op(1, Pauli::Z)
            .with_op(2, Pauli::Z), // Horizontal Z1Z2
        // Site (0,2) - Qubit 2
        PauliString::new(Complex::new(-0.1 * 3.0, 0.0)).with_op(2, Pauli::Z), // Field Z2
        PauliString::new(Complex::new(-2.5, 0.0))
            .with_op(2, Pauli::Z)
            .with_op(5, Pauli::Z), // Vertical Z2Z5
        PauliString::new(Complex::new(-3.0, 0.0))
            .with_op(2, Pauli::Z)
            .with_op(0, Pauli::Z), // Horizontal Z2Z0 (PBC)
        // Site (1,0) - Qubit 3
        PauliString::new(Complex::new(-0.1 * 4.0, 0.0)).with_op(3, Pauli::Z), // Field Z3
        PauliString::new(Complex::new(-3.5, 0.0))
            .with_op(3, Pauli::Z)
            .with_op(6, Pauli::Z), // Vertical Z3Z6
        PauliString::new(Complex::new(-4.0, 0.0))
            .with_op(3, Pauli::Z)
            .with_op(4, Pauli::Z), // Horizontal Z3Z4
        // Site (1,1) - Qubit 4
        PauliString::new(Complex::new(-0.1 * 5.0, 0.0)).with_op(4, Pauli::Z), // Field Z4
        PauliString::new(Complex::new(-4.5, 0.0))
            .with_op(4, Pauli::Z)
            .with_op(7, Pauli::Z), // Vertical Z4Z7
        PauliString::new(Complex::new(-5.0, 0.0))
            .with_op(4, Pauli::Z)
            .with_op(5, Pauli::Z), // Horizontal Z4Z5
        // Site (1,2) - Qubit 5
        PauliString::new(Complex::new(-0.1 * 6.0, 0.0)).with_op(5, Pauli::Z), // Field Z5
        PauliString::new(Complex::new(-5.5, 0.0))
            .with_op(5, Pauli::Z)
            .with_op(8, Pauli::Z), // Vertical Z5Z8
        PauliString::new(Complex::new(-6.0, 0.0))
            .with_op(5, Pauli::Z)
            .with_op(3, Pauli::Z), // Horizontal Z5Z3 (PBC)
        // Site (2,0) - Qubit 6
        PauliString::new(Complex::new(-0.1 * 7.0, 0.0)).with_op(6, Pauli::Z), // Field Z6
        PauliString::new(Complex::new(-6.5, 0.0))
            .with_op(6, Pauli::Z)
            .with_op(0, Pauli::Z), // Vertical Z6Z0 (PBC)
        PauliString::new(Complex::new(-7.0, 0.0))
            .with_op(6, Pauli::Z)
            .with_op(7, Pauli::Z), // Horizontal Z6Z7
        // Site (2,1) - Qubit 7
        PauliString::new(Complex::new(-0.1 * 8.0, 0.0)).with_op(7, Pauli::Z), // Field Z7
        PauliString::new(Complex::new(-7.5, 0.0))
            .with_op(7, Pauli::Z)
            .with_op(1, Pauli::Z), // Vertical Z7Z1 (PBC)
        PauliString::new(Complex::new(-8.0, 0.0))
            .with_op(7, Pauli::Z)
            .with_op(8, Pauli::Z), // Horizontal Z7Z8
        // Site (2,2) - Qubit 8
        PauliString::new(Complex::new(-0.1 * 9.0, 0.0)).with_op(8, Pauli::Z), // Field Z8
        PauliString::new(Complex::new(-8.5, 0.0))
            .with_op(8, Pauli::Z)
            .with_op(2, Pauli::Z), // Vertical Z8Z2 (PBC)
        PauliString::new(Complex::new(-9.0, 0.0))
            .with_op(8, Pauli::Z)
            .with_op(6, Pauli::Z), // Horizontal Z8Z6 (PBC)
    ];

    let state: State = State::new_basis_n(9, 511).unwrap();
    let result_state: State = result.apply(&state).unwrap();
    let expected_state: State = SumOp::new(expected_terms).apply(&state).unwrap();
    assert_eq!(result_state, expected_state);
}

#[test]
fn test_ising_2d_error() {
    // 1x1 qubit system (invalid case)
    let h: [[f64; 1]; 1] = [[1.0]];
    let j: [[[f64; 2]; 1]; 1] = [[[0.5, 1.0]]];
    let m: f64 = 0.1;

    let result: Result<SumOp, Error> = ising_2d(h, j, m);
    assert_eq!(result.unwrap_err(), Error::InvalidNumberOfInputs(1, 2));
}

#[test]
fn test_ising_2d_uniform_success() {
    // 3x3 qubit system
    let h: f64 = 1.0;
    let j: f64 = 2.0;
    let m: f64 = 0.1;

    let result: SumOp = ising_2d_uniform(3, 3, h, j, m).unwrap();
    assert_eq!(result.terms.len(), 27);

    let expected_terms: Vec<PauliString> = vec![
        // Field terms
        m * h * PauliString::new(Complex::new(-1.0, 0.0)).with_op(0, Pauli::Z),
        m * h * PauliString::new(Complex::new(-1.0, 0.0)).with_op(1, Pauli::Z),
        m * h * PauliString::new(Complex::new(-1.0, 0.0)).with_op(2, Pauli::Z),
        m * h * PauliString::new(Complex::new(-1.0, 0.0)).with_op(3, Pauli::Z),
        m * h * PauliString::new(Complex::new(-1.0, 0.0)).with_op(4, Pauli::Z),
        m * h * PauliString::new(Complex::new(-1.0, 0.0)).with_op(5, Pauli::Z),
        m * h * PauliString::new(Complex::new(-1.0, 0.0)).with_op(6, Pauli::Z),
        m * h * PauliString::new(Complex::new(-1.0, 0.0)).with_op(7, Pauli::Z),
        m * h * PauliString::new(Complex::new(-1.0, 0.0)).with_op(8, Pauli::Z),
        // Vertical coupling terms
        j * PauliString::new(Complex::new(-1.0, 0.0))
            .with_op(0, Pauli::Z)
            .with_op(3, Pauli::Z),
        j * PauliString::new(Complex::new(-1.0, 0.0))
            .with_op(1, Pauli::Z)
            .with_op(4, Pauli::Z),
        j * PauliString::new(Complex::new(-1.0, 0.0))
            .with_op(2, Pauli::Z)
            .with_op(5, Pauli::Z),
        j * PauliString::new(Complex::new(-1.0, 0.0))
            .with_op(3, Pauli::Z)
            .with_op(6, Pauli::Z),
        j * PauliString::new(Complex::new(-1.0, 0.0))
            .with_op(4, Pauli::Z)
            .with_op(7, Pauli::Z),
        j * PauliString::new(Complex::new(-1.0, 0.0))
            .with_op(5, Pauli::Z)
            .with_op(8, Pauli::Z),
        j * PauliString::new(Complex::new(-1.0, 0.0))
            .with_op(6, Pauli::Z)
            .with_op(0, Pauli::Z),
        j * PauliString::new(Complex::new(-1.0, 0.0))
            .with_op(7, Pauli::Z)
            .with_op(1, Pauli::Z),
        j * PauliString::new(Complex::new(-1.0, 0.0))
            .with_op(8, Pauli::Z)
            .with_op(2, Pauli::Z),
        // Horizontal coupling terms
        j * PauliString::new(Complex::new(-1.0, 0.0))
            .with_op(0, Pauli::Z)
            .with_op(1, Pauli::Z),
        j * PauliString::new(Complex::new(-1.0, 0.0))
            .with_op(1, Pauli::Z)
            .with_op(2, Pauli::Z),
        j * PauliString::new(Complex::new(-1.0, 0.0))
            .with_op(2, Pauli::Z)
            .with_op(0, Pauli::Z),
        j * PauliString::new(Complex::new(-1.0, 0.0))
            .with_op(3, Pauli::Z)
            .with_op(4, Pauli::Z),
        j * PauliString::new(Complex::new(-1.0, 0.0))
            .with_op(4, Pauli::Z)
            .with_op(5, Pauli::Z),
        j * PauliString::new(Complex::new(-1.0, 0.0))
            .with_op(5, Pauli::Z)
            .with_op(3, Pauli::Z),
        j * PauliString::new(Complex::new(-1.0, 0.0))
            .with_op(6, Pauli::Z)
            .with_op(7, Pauli::Z),
        j * PauliString::new(Complex::new(-1.0, 0.0))
            .with_op(7, Pauli::Z)
            .with_op(8, Pauli::Z),
        j * PauliString::new(Complex::new(-1.0, 0.0))
            .with_op(8, Pauli::Z)
            .with_op(6, Pauli::Z),
    ];

    let state: State = State::new_basis_n(9, 255).unwrap();
    // Round the result to 4 decimal places for comparison due to floating point precision issues
    let result_state: Vec<f64> = result
        .apply(&state)
        .unwrap()
        .state_vector
        .iter()
        .map(|c| f64::round(c.re * 10000.0) / 10000.0)
        .collect();
    let expected_state: Vec<f64> = SumOp::new(expected_terms)
        .apply(&state)
        .unwrap()
        .state_vector
        .iter()
        .map(|c| f64::round(c.re * 10000.0) / 10000.0)
        .collect();
    assert_eq!(result_state, expected_state);
}

#[test]
fn test_ising_2d_uniform_error() {
    // 1x1 qubit system (invalid case)
    let h: f64 = 1.0;
    let j: f64 = 2.0;
    let m: f64 = 0.1;

    let result: Result<SumOp, Error> = ising_2d_uniform(1, 1, h, j, m);
    assert_eq!(result.unwrap_err(), Error::InvalidNumberOfInputs(1, 2));
}