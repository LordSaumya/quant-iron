use crate::{components::state::State, errors::Error};
use num_complex::Complex;
use std::f64::consts::{PI, FRAC_1_SQRT_2};

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
fn test_state_new_ghz_success() {
    let n: usize = 6;
    let dim = (1 << n) - 1;
    let ones_state = State::new_basis_n(n, dim).unwrap();
    let zeros_state = State::new_zero(n).unwrap();
    let expected_state = FRAC_1_SQRT_2 * (ones_state + zeros_state);

    let ghz_state = State::new_ghz(n).unwrap();
    assert_eq!(ghz_state, expected_state);
}

#[test]
fn test_state_new_ghz_error() {
    // Invalid number of qubits
    let state: Result<State, Error> = State::new_ghz(0);
    assert!(state.is_err());
    assert_eq!(state.unwrap_err(), Error::InvalidNumberOfQubits(0));
}

#[test]
fn test_state_normalise_success() {
    let state_vector: Vec<Complex<f64>> = vec![Complex::new(3.0, 0.0), Complex::new(4.0, 0.0)];
    let state: State = State { state_vector, num_qubits: 2 };

    let normalised_state = state.normalise().unwrap();
    let expected_state_vector: Vec<Complex<f64>> = vec![Complex::new(0.6, 0.0), Complex::new(0.8, 0.0)];
    assert_eq!(normalised_state.state_vector, expected_state_vector);
}

#[test]
fn test_state_normalise_error() {
    let state_vector: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)];
    let state: State = State { state_vector, num_qubits: 2 };

    let result = state.normalise();
    assert!(result.is_err());
    assert_eq!(result.unwrap_err(), Error::ZeroNorm);
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

#[test]
fn test_state_conj() {
    let state_vector: Vec<Complex<f64>> = vec![Complex::new(1.0, 1.0), Complex::new(0.0, -1.0)];
    let conj_state_vector: Vec<Complex<f64>> = vec![Complex::new(1.0, -1.0), Complex::new(0.0, 1.0)];

    let state: State = State { state_vector, num_qubits: 1 };
    let conj_state: State = state.conj();
    let expected_state = State { state_vector: conj_state_vector, num_qubits: 1 };
    assert_eq!(conj_state, expected_state);
}

#[test]
fn test_state_fs_dist_success() {
    // Zero distance
    let zero_dist_state = State::new_plus(2).unwrap();
    assert!((zero_dist_state.fs_dist(&zero_dist_state).unwrap() - 0.0).abs() < f64::EPSILON);

    // Nonzero distance
    let state1_vector: Vec<Complex<f64>> = vec![Complex::new(1.0, 0.0), Complex::new(0.0, 1.0)];
    let state1 = State { state_vector: state1_vector, num_qubits: 2 };

    let state2_vector: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0), Complex::new(1.0, 0.0)];
    let state2 = State { state_vector: state2_vector, num_qubits: 2 };

    let fs_dist = state1.fs_dist(&state2).unwrap();

    assert!((fs_dist - PI/4.0).abs() < f64::EPSILON);

    // Check if D = arccos(sqrt(F))
    assert!((fs_dist.sqrt().acos() - 0.5) < f64::EPSILON);
}

#[test]
fn test_state_fs_dist_errors() {
    // Empty state vector (norm is 0)
    let empty_state_vector: Vec<Complex<f64>> = vec![];
    let state = State { state_vector: empty_state_vector, num_qubits: 0 };
    let result = state.fs_dist(&state);
    assert!(result.is_err());
    assert_eq!(result.unwrap_err(), Error::ZeroNorm);
    
    // Different number of qubits
    let state1 = State::new_plus(2).unwrap();
    let state2 = State::new_plus(1).unwrap();
    let result = state1.fs_dist(&state2);
    assert!(result.is_err());
    assert_eq!(result.unwrap_err(), Error::InvalidNumberOfQubits(2));

    // Zero norm error
    let zero_norm_state_vector: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)];
    let state = State { state_vector: zero_norm_state_vector, num_qubits: 2 };
    let state_2 = State::new_basis_n(2, 1).unwrap();
    let result = state.fs_dist(&state_2);
    assert!(result.is_err());
    assert_eq!(result.unwrap_err(), Error::ZeroNorm);
}

#[test]
fn test_state_fs_fidelity_success() {
    // Zero fidelity
    let state1_vector: Vec<Complex<f64>> = vec![Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)];
    let orthogonal_state_vector: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0), Complex::new(1.0, 0.0)];
    let state1 = State { state_vector: state1_vector, num_qubits: 2 };
    let state2 = State { state_vector: orthogonal_state_vector, num_qubits: 2 };
    let fidelity = state1.fs_fidelity(&state2).unwrap();
    assert!((fidelity - 0.0).abs() < f64::EPSILON);

    // Perfect fidelity
    let state = State::new_plus(2).unwrap();
    let fidelity = state.fs_fidelity(&state).unwrap();
    assert!((fidelity - 1.0).abs() < f64::EPSILON);

    // Nonzero fidelity
    let plus_state = State::new_plus(1).unwrap();
    let basis_state = State::new_basis_n(1, 1).unwrap();

    let fidelity = plus_state.fs_fidelity(&basis_state).unwrap();
    assert!((fidelity - 0.5).abs() < f64::EPSILON);

    // Check if F = cos^2(D)
    assert!((fidelity - (PI/4.0).cos().powi(2)).abs() < f64::EPSILON);
}

#[test]
fn test_state_new_bell_states() {
    let amplitude = Complex::new(FRAC_1_SQRT_2, 0.0);

    // Test |Φ+⟩
    let phi_plus = State::new_phi_plus();
    let expected_phi_plus_vec = vec![amplitude, Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), amplitude];
    assert_eq!(phi_plus.num_qubits, 2);
    assert_eq!(phi_plus.state_vector, expected_phi_plus_vec);

    // Test |Φ-⟩
    let phi_minus = State::new_phi_minus();
    let expected_phi_minus_vec = vec![amplitude, Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), -amplitude];
    assert_eq!(phi_minus.num_qubits, 2);
    assert_eq!(phi_minus.state_vector, expected_phi_minus_vec);

    // Test |Ψ+⟩
    let psi_plus = State::new_psi_plus();
    let expected_psi_plus_vec = vec![Complex::new(0.0, 0.0), amplitude, amplitude, Complex::new(0.0, 0.0)];
    assert_eq!(psi_plus.num_qubits, 2);
    assert_eq!(psi_plus.state_vector, expected_psi_plus_vec);

    // Test |Ψ-⟩
    let psi_minus = State::new_psi_minus();
    let expected_psi_minus_vec = vec![Complex::new(0.0, 0.0), amplitude, -amplitude, Complex::new(0.0, 0.0)];
    assert_eq!(psi_minus.num_qubits, 2);
    assert_eq!(psi_minus.state_vector, expected_psi_minus_vec);
}