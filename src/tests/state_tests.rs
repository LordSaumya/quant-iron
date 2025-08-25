use crate::{errors::Error, components::state::{State}};
use num_complex::Complex;

#[test]
fn test_state_new_success() {
    // New valid state
    let state_vector: Vec<Complex<f64>> = vec![Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)];
    let state: State = State::new(state_vector.clone()).unwrap();

    assert!(state.num_qubits == 1);
    assert!(state.state_vector == state_vector);
}

#[test]
fn test_state_new_errors() {
    // Create invalid state (empty vector)
    let state_vector: Vec<Complex<f64>> = vec![];
    let state: Result<State, Error> = State::new(state_vector.clone());
    assert!(state.is_err());
    assert_eq!(state.unwrap_err(), Error::InvalidNumberOfQubits(0));

    // Create invalid state (non-normalised vector)
    let state_vector: Vec<Complex<f64>> = vec![Complex::new(1.0, 0.0), Complex::new(1.0, 0.0)];
    let state: Result<State, Error> = State::new(state_vector.clone());
    assert!(state.is_err());
    assert_eq!(state.unwrap_err(), Error::StateVectorNotNormalised);

    // Create invalid state (number of qubits not a power of 2)
    let state_vector: Vec<Complex<f64>> = vec![Complex::new(1.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)];
    let state: Result<State, Error> = State::new(state_vector.clone());
    assert!(state.is_err());
    assert_eq!(state.unwrap_err(), Error::InvalidNumberOfQubits(1));
}

#[test]
fn test_state_new_hartree_fock_success() {
    let hf_state: State = State::new_hartree_fock(2, 2).unwrap();
    let expected_state: State = State::new_basis_n(2, 3).unwrap(); // |11>
    assert_eq!(hf_state, expected_state);

    let hf_state: State = State::new_hartree_fock(4, 6).unwrap();
    let expected_state: State = State::new_basis_n(6, 60).unwrap(); // |111100>
    assert_eq!(hf_state, expected_state);

    let hf_state: State = State::new_hartree_fock(1, 4).unwrap();
    let expected_state: State = State::new_basis_n(4, 8).unwrap(); // |1000>
    assert_eq!(hf_state, expected_state);
}

#[test]
fn test_state_new_zero_success() {
    let expected_state_vector: Vec<Complex<f64>> = vec![Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)];
    let state: State = State::new_zero(1).unwrap();

    assert_eq!(state.num_qubits, 1);
    assert_eq!(state.state_vector, expected_state_vector);
}

#[test]
fn test_state_new_zero_errors() {
    let state: Result<State, Error> = State::new_zero(0);

    assert!(state.is_err());
    assert_eq!(state.unwrap_err(), Error::InvalidNumberOfQubits(0));
}

#[test]
fn test_state_new_basis_n_success() {
    let expected_state_vector: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0), Complex::new(1.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)];
    let state: State = State::new_basis_n(2, 1).unwrap();

    assert_eq!(state.num_qubits, 2);
    assert_eq!(state.state_vector, expected_state_vector);
}

#[test]
fn test_state_new_basis_n_errors() {
    // Invalid qubit index
    let state: Result<State, Error> = State::new_basis_n(2, 4);
    assert!(state.is_err());
    assert_eq!(state.unwrap_err(), Error::InvalidQubitIndex(4, 2));

    // Invalid number of qubits
    let state: Result<State, Error> = State::new_basis_n(0, 0);
    assert!(state.is_err());
    assert_eq!(state.unwrap_err(), Error::InvalidNumberOfQubits(0));
}

#[test]
fn test_state_new_plus_success() {
    let amplitude: f64 = 1.0 / (2.0_f64).sqrt();
    let expected_state_vector: Vec<Complex<f64>> = vec![Complex::new(amplitude, 0.0), Complex::new(amplitude, 0.0)];
    let state: State = State::new_plus(1).unwrap();

    assert_eq!(state.num_qubits, 1);
    assert_eq!(state.state_vector, expected_state_vector);
}

#[test]
fn test_state_new_plus_errors() {
    // Invalid number of qubits
    let state: Result<State, Error> = State::new_plus(0);

    assert!(state.is_err());
    assert_eq!(state.unwrap_err(), Error::InvalidNumberOfQubits(0));
}

#[test]
fn test_state_new_minus_success() {
    let amplitude: f64 = 1.0 / (2.0_f64).sqrt();
    let expected_state_vector: Vec<Complex<f64>> = vec![Complex::new(amplitude, 0.0), Complex::new(-amplitude, 0.0)];
    let state: State = State::new_minus(1).unwrap();

    assert_eq!(state.num_qubits, 1);
    assert_eq!(state.state_vector, expected_state_vector);
}

#[test]
fn test_state_new_minus_errors() {
    // Invalid number of qubits
    let state: Result<State, Error> = State::new_minus(0);

    assert!(state.is_err());
    assert_eq!(state.unwrap_err(), Error::InvalidNumberOfQubits(0));
}

#[test]
fn test_state_probability_success() {
    let state: State = State::new_plus(1).unwrap();
    let expected_probability: f64 = 0.5;

    let probability: f64 = state.probability(0).unwrap();
    assert!((probability - expected_probability).abs() < f64::EPSILON);
}

#[test]
fn test_state_probability_errors() {
    let state = State::new_plus(1).unwrap();

    // Invalid qubit index
    let probability: Result<f64, Error> = state.probability(2);
    assert!(probability.is_err());
    assert_eq!(probability.unwrap_err(), Error::InvalidQubitIndex(2, 1));
}

#[test]
fn test_state_tensor_product_success() {
    // |+>|-> = |+-> = 1/2 (|00> – |01> + |10> – |11>)
    let state1 = State::new_plus(1).unwrap();
    let state2 = State::new_minus(1).unwrap();

    // 1/2 (|00> – |01> + |10> – |11>)
    let expected: State = 0.5 * (
         State::new_basis_n(2, 0).unwrap()  // +|00>
       - State::new_basis_n(2, 1).unwrap()  // –|01>
       + State::new_basis_n(2, 2).unwrap()  // +|10>
       - State::new_basis_n(2, 3).unwrap()  // –|11>
    );

    let actual: State = state1.tensor_product(&state2).unwrap();
    assert_eq!(actual, expected);

    // |0>|1> = |01>
    let state1 = State::new_zero(1).unwrap();
    let state2 = State::new_basis_n(1, 1).unwrap();

    let expected: State = State::new_basis_n(2, 1).unwrap(); // |01>
    let actual: State = state1.tensor_product(&state2).unwrap();
    assert_eq!(actual, expected);
}