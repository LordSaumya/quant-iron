use std::f64::consts::PI;

use crate::{errors::Error, components::{measurement::MeasurementResult, measurement::MeasurementBasis, state::{State, ChainableState}}};
use num_complex::Complex;

#[test]
fn test_measurement_measure_1_qubit_computational_success() {
    // Example 1 (simple)

    // Create a state vector with 2 qubits
    let amplitude: f64 = 1.0 / 2.0_f64.sqrt();
    let state_vector: Vec<Complex<f64>> = vec![Complex::new(amplitude, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(amplitude, 0.0)];
    let state: State = State::new(state_vector.clone()).unwrap();

    // Create a measurement result with 1 qubit and computational basis
    let measured_qubits: &[usize] = &[0];
    let measurement_result: MeasurementResult = state.measure(MeasurementBasis::Computational, measured_qubits).unwrap();

    // Check basis of measurement
    assert_eq!(measurement_result.get_basis(), &MeasurementBasis::Computational);

    // Check indices of measured qubits
    assert_eq!(measurement_result.get_indices(), measured_qubits);

    // Check new state vector based on outcome of measurement
    let outcome: u8 = measurement_result.get_outcomes()[0];
    let expected_state_vector: Vec<Complex<f64>> = match outcome {
        0 => vec![Complex::new(1.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)], // |00> is the only possible outcome
        1 => vec![Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(1.0, 0.0)], // |11> is the only possible outcome
        _ => unreachable!(),
    };

    assert_eq!(measurement_result.get_new_state().state_vector, expected_state_vector);
    assert_eq!(measurement_result.get_new_state().num_qubits, 2);


    // Example 2 (complex)

    // Create a state vector with 2 qubits
    let amplitude: f64 = 1.0 / 3.0_f64.sqrt();
    let state_vector: Vec<Complex<f64>> = vec![Complex::new(amplitude, 0.0), Complex::new(amplitude, 0.0), Complex::new(0.0, 0.0), Complex::new(amplitude, 0.0)];
    let state: State = State::new(state_vector.clone()).unwrap();

    // Create a measurement result with 1 qubit and computational basis
    let measured_qubits: &[usize] = &[0];
    let measurement_result: MeasurementResult = state.measure(MeasurementBasis::Computational, measured_qubits).unwrap();

    // Check basis of measurement
    assert_eq!(measurement_result.get_basis(), &MeasurementBasis::Computational);

    // Check indices of measured qubits
    assert_eq!(measurement_result.get_indices(), measured_qubits);

    // Check new state vector based on outcome of measurement
    let outcome: u8 = measurement_result.get_outcomes()[0];
    let expected_amplitude: f64 = 1.0 / 2.0_f64.sqrt();
    let expected_state_vector: Vec<Complex<f64>> = match outcome {
        0 => vec![Complex::new(1.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)], // |00> is the only possible outcome
        1 => vec![Complex::new(0.0, 0.0), Complex::new(expected_amplitude, 0.0), Complex::new(0.0, 0.0), Complex::new(expected_amplitude, 0.0)], // |11> and |01> are the only possible outcomes, with equal probability
        _ => unreachable!(),
    };

    // Compare expected state vector with the new state vector
    let expected_s = State::new(expected_state_vector).unwrap();
    assert_eq!(*measurement_result.get_new_state(), expected_s);

    assert_eq!(measurement_result.get_new_state().num_qubits, 2);
}

#[test]
fn test_measurement_measure_all_qubits_computational_success() {
    // Create a state vector with 2 qubits
    let amplitude: f64 = 1.0 / 4.0_f64.sqrt();
    let state_vector: Vec<Complex<f64>> = vec![Complex::new(amplitude, 0.0), Complex::new(amplitude, 0.0), Complex::new(amplitude, 0.0), Complex::new(amplitude, 0.0)];
    let state: State = State::new(state_vector.clone()).unwrap();

    // Create a measurement result with all qubits and computational basis
    let measured_qubits: &[usize] = &[];
    let measurement_result: MeasurementResult = state.measure(MeasurementBasis::Computational, measured_qubits).unwrap();

    // Check basis of measurement
    assert_eq!(measurement_result.get_basis(), &MeasurementBasis::Computational);

    // Check new state vector based on outcome of measurement
    let outcome_0: u8 = measurement_result.get_outcomes()[0];
    let outcome_1: u8 = measurement_result.get_outcomes()[1];

    let expected_state_vector: Vec<Complex<f64>> = match (outcome_1, outcome_0) {
        (0, 0) => vec![Complex::new(1.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)], // |00>
        (0, 1) => vec![Complex::new(0.0, 0.0), Complex::new(1.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)], // |01>
        (1, 0) => vec![Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)], // |10>
        (1, 1) => vec![Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(1.0, 0.0)], // |11>
        _ => unreachable!(),
    };

    assert_eq!(measurement_result.get_new_state().state_vector, expected_state_vector);
    assert_eq!(measurement_result.get_new_state().num_qubits, 2);
}

#[test]
fn test_measurement_measure_errors() {
    // Create a state vector with 2 qubits
    let amplitude: f64 = 1.0 / 4.0_f64.sqrt();
    let state_vector: Vec<Complex<f64>> = vec![Complex::new(amplitude, 0.0), Complex::new(amplitude, 0.0), Complex::new(amplitude, 0.0), Complex::new(amplitude, 0.0)];
    let state: State = State::new(state_vector.clone()).unwrap();

    // Invalid qubit index
    let measured_qubits: &[usize] = &[3]; // Invalid qubit index
    let measurement_result: Result<MeasurementResult, Error> = state.measure(MeasurementBasis::Computational, measured_qubits);
    assert!(measurement_result.is_err());
    assert_eq!(measurement_result.unwrap_err(), Error::InvalidQubitIndex(3, 2));

    // Invalid number of qubits
    let measured_qubits: &[usize] = &[0, 1, 2]; // More than 2 qubits are measured
    let measurement_result: Result<MeasurementResult, Error> = state.measure(MeasurementBasis::Computational, measured_qubits);
    assert!(measurement_result.is_err());
    assert_eq!(measurement_result.unwrap_err(), Error::InvalidNumberOfQubits(2));
}

#[test]
fn test_measurement_measure_n_1_qubit_computational_success() {
    // Create a state vector with 2 qubits
    let amplitude: f64 = 1.0 / 2.0_f64.sqrt();
    let state_vector: Vec<Complex<f64>> = vec![Complex::new(amplitude, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(amplitude, 0.0)];
    let state: State = State::new(state_vector.clone()).unwrap();

    // Create a measurement result with 1 qubit and computational basis
    let measured_qubits: &[usize] = &[0];
    let measurement_results: Vec<MeasurementResult> = state.measure_n(MeasurementBasis::Computational, measured_qubits, 5).unwrap();

    // Check basis of measurement
    assert_eq!(measurement_results[0].get_basis(), &MeasurementBasis::Computational);

    // Check indices of measured qubits
    assert_eq!(measurement_results[0].get_indices(), measured_qubits);

    // Check number of measurement results
    assert_eq!(measurement_results.len(), 5);

    // Check new state vectors based on outcome of measurement
    for measurement_result in measurement_results.iter() {
        let outcome: u8 = measurement_result.get_outcomes()[0];
        let expected_state_vector: Vec<Complex<f64>> = match outcome {
            0 => vec![Complex::new(1.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)], // |00> is the only possible outcome
            1 => vec![Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(1.0, 0.0)], // |11> is the only possible outcome
            _ => unreachable!(),
        };

        assert_eq!(measurement_result.get_new_state().state_vector, expected_state_vector);
        assert_eq!(measurement_result.get_new_state().num_qubits, 2);
    }
}

#[test]
fn test_measurement_measure_n_all_computational_success() {
    // Create a state vector with 2 qubits
    let amplitude: f64 = 1.0 / 4.0_f64.sqrt();
    let state_vector: Vec<Complex<f64>> = vec![Complex::new(amplitude, 0.0), Complex::new(amplitude, 0.0), Complex::new(amplitude, 0.0), Complex::new(amplitude, 0.0)];
    let state: State = State::new(state_vector.clone()).unwrap();

    // Create a measurement result with all qubits and computational basis
    let measured_qubits: &[usize] = &[];
    let measurement_results: Vec<MeasurementResult> = state.measure_n(MeasurementBasis::Computational, measured_qubits, 5).unwrap();

    // Check basis of measurement
    assert_eq!(measurement_results[0].get_basis(), &MeasurementBasis::Computational);

    // Check number of measurement results
    assert_eq!(measurement_results.len(), 5);

    // Check new state vectors based on outcome of measurement
    for measurement_result in measurement_results.iter() {
        let outcome_0: u8 = measurement_result.get_outcomes()[0];
        let outcome_1: u8 = measurement_result.get_outcomes()[1];

        let expected_state_vector: Vec<Complex<f64>> = match (outcome_1, outcome_0) {
            (0, 0) => vec![Complex::new(1.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)], // |00>
            (0, 1) => vec![Complex::new(0.0, 0.0), Complex::new(1.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)], // |01>
            (1, 0) => vec![Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)], // |10>
            (1, 1) => vec![Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(1.0, 0.0)], // |11>
            _ => unreachable!(),
        };

        assert_eq!(measurement_result.get_new_state().state_vector, expected_state_vector);
        assert_eq!(measurement_result.get_new_state().num_qubits, 2);
    }
}

#[test]
fn test_measurement_measure_n_errors() {
    // Create a state vector with 2 qubits
    let amplitude: f64 = 1.0 / 4.0_f64.sqrt();
    let state_vector: Vec<Complex<f64>> = vec![Complex::new(amplitude, 0.0), Complex::new(amplitude, 0.0), Complex::new(amplitude, 0.0), Complex::new(amplitude, 0.0)];
    let state: State = State::new(state_vector.clone()).unwrap();

    // Invalid qubit index
    let measured_qubits: &[usize] = &[3]; // Invalid qubit index
    let measurement_results: Result<Vec<MeasurementResult>, Error> = state.measure_n(MeasurementBasis::Computational, measured_qubits, 5);
    assert!(measurement_results.is_err());
    assert_eq!(measurement_results.unwrap_err(), Error::InvalidQubitIndex(3, 2));

    // Invalid number of qubits
    let measured_qubits: &[usize] = &[0, 1, 2]; // More than 2 qubits are measured
    let measurement_results: Result<Vec<MeasurementResult>, Error> = state.measure_n(MeasurementBasis::Computational, measured_qubits, 5);
    assert!(measurement_results.is_err());
    assert_eq!(measurement_results.unwrap_err(), Error::InvalidNumberOfQubits(2));

    // Invalid number of measurements
    let measured_qubits: &[usize] = &[0]; // Valid qubit index
    let measurement_results: Result<Vec<MeasurementResult>, Error> = state.measure_n(MeasurementBasis::Computational, measured_qubits, 0); // Invalid number of measurements
    assert!(measurement_results.is_err());
    assert_eq!(measurement_results.unwrap_err(), Error::InvalidNumberOfMeasurements(0));
}

#[test]
fn test_measurement_measure_1_qubit_x_basis_success() {
    // |+> * |0> = |+0>
    let state = State::new_plus(1).unwrap().tensor_product(&State::new_zero(1).unwrap()).unwrap(); // |+0> state

    // Measure the second qubit (|+>) in the X basis
    let measured_qubits: &[usize] = &[1]; // Measure the second qubit (|+>) in the X basis
    let measurement_result: MeasurementResult = state.measure(MeasurementBasis::X, measured_qubits).unwrap();

    assert_eq!(measurement_result.get_basis(), &MeasurementBasis::X);
    assert_eq!(measurement_result.get_indices(), measured_qubits);

    let outcome: u8 = measurement_result.get_outcomes()[0];
    let new_state_ref = measurement_result.get_new_state();

    // Outcome can only be 0
    assert_eq!(outcome, 0);
    
    // Expected state is the same as the original state (|+0>)
    assert_eq!(*new_state_ref, state);

    // Measure the first qubit (|0>) in the X basis
    let measured_qubits: &[usize] = &[0]; // Measure the first qubit (|0>) in the X basis
    let measurement_result: MeasurementResult = state.measure(MeasurementBasis::X, measured_qubits).unwrap();

    assert_eq!(measurement_result.get_basis(), &MeasurementBasis::X);
    assert_eq!(measurement_result.get_indices(), measured_qubits);

    let outcome: u8 = measurement_result.get_outcomes()[0];
    let new_state_ref = measurement_result.get_new_state();

    let expected_plus_state = State::new_plus(2).unwrap(); // |+> state
    let expected_minus_state = State::new_plus(1).unwrap().tensor_product(&State::new_minus(1).unwrap()).unwrap(); // |+-> state

    // Outcome can be either 0 or 1
    match outcome {
        0 => assert_eq!(*new_state_ref, expected_plus_state),
        1 => assert_eq!(*new_state_ref, expected_minus_state),
        _ => panic!("Unexpected outcome: {}", outcome),
    }
}

#[test]
fn test_measurement_measure_all_qubits_x_basis_success() {
    // |+> * |0> = |+0>
    let state = State::new_plus(1).unwrap().tensor_product(&State::new_zero(1).unwrap()).unwrap(); // |+0> state

    // Measure all qubits in the X basis
    let measured_qubits: &[usize] = &[]; // Measure all qubits in the X basis
    let measurement_result: MeasurementResult = state.measure(MeasurementBasis::X, measured_qubits).unwrap();

    assert_eq!(measurement_result.get_basis(), &MeasurementBasis::X);
    assert_eq!(measurement_result.get_indices(), &(0..state.num_qubits()).collect::<Vec<usize>>());

    let outcome: &Vec<u8> = measurement_result.get_outcomes();
    let new_state_ref = measurement_result.get_new_state();

    // Outcome can be (0, 0) or (1, 0)
    let expected_plus_state = State::new_plus(2).unwrap(); // |++> state
    let expected_minus_state = State::new_plus(1).unwrap().tensor_product(&State::new_minus(1).unwrap()).unwrap(); // |+-> state

    match outcome.as_slice() {
        [0, 0] => assert_eq!(*new_state_ref, expected_plus_state),
        [1, 0] => assert_eq!(*new_state_ref, expected_minus_state),
        _ => panic!("Unexpected outcome: {:?}", outcome),
    }
}

#[test]
fn test_measurement_measure_1_qubit_y_basis_success() {
    // |+> * |0> = |+0>
    let state = State::new_plus(1).unwrap().tensor_product(&State::new_zero(1).unwrap()).unwrap(); // |+0> state

    let measured_qubits: &[usize] = &[0]; // Measure the first qubit (|0>) in the Y basis
    let measurement_result: MeasurementResult = state.measure(MeasurementBasis::Y, measured_qubits).unwrap();

    assert_eq!(measurement_result.get_basis(), &MeasurementBasis::Y);
    assert_eq!(measurement_result.get_indices(), measured_qubits);

    let outcome: u8 = measurement_result.get_outcomes()[0];
    let new_state_ref = measurement_result.get_new_state();
    let i = Complex::new(0.0, 1.0);

    // Outcome can be 0 or 1
    let expected_state_0 = 0.5 * (
        State::new_basis_n(2, 0).unwrap() +
        i * State::new_basis_n(2, 1).unwrap() +
        State::new_basis_n(2, 2).unwrap() +
        i * State::new_basis_n(2, 3).unwrap()
    );

    let expected_state_1 = 0.5 * (
        State::new_basis_n(2, 0).unwrap() -
        i * State::new_basis_n(2, 1).unwrap() +
        State::new_basis_n(2, 2).unwrap() -
        i * State::new_basis_n(2, 3).unwrap()
    );

    match outcome {
        0 => assert_eq!(*new_state_ref, expected_state_0),
        1 => assert_eq!(*new_state_ref, expected_state_1),
        _ => panic!("Unexpected outcome: {}", outcome),
    }

    // Measure the second qubit (|+>) in the Y basis
    let measured_qubits: &[usize] = &[1]; // Measure the second qubit (|+>) in the Y basis
    let measurement_result: MeasurementResult = state.measure(MeasurementBasis::Y, measured_qubits).unwrap();

    assert_eq!(measurement_result.get_basis(), &MeasurementBasis::Y);
    assert_eq!(measurement_result.get_indices(), measured_qubits);

    let outcome: u8 = measurement_result.get_outcomes()[0];
    let new_state_ref: &State = measurement_result.get_new_state();
    let i = Complex::new(0.0, 1.0);

    // Outcome can be 0 or 1
    let expected_state_0 = 0.5 * (
        (1.0 - i) * State::new_basis_n(2, 0).unwrap() +
        0.0 * State::new_basis_n(2, 1).unwrap() +
        (1.0 + i) * State::new_basis_n(2, 2).unwrap() +
        0.0 * State::new_basis_n(2, 3).unwrap()
    );

    let expected_state_1 = 0.5 * (
        (1.0 + i) * State::new_basis_n(2, 0).unwrap() +
        0.0 * State::new_basis_n(2, 1).unwrap() +
        (1.0 - i) * State::new_basis_n(2, 2).unwrap() +
        0.0 * State::new_basis_n(2, 3).unwrap()
    );

    match outcome {
        0 => assert_eq!(*new_state_ref, expected_state_0),
        1 => assert_eq!(*new_state_ref, expected_state_1),
        _ => panic!("Unexpected outcome: {}", outcome),
    }
}

#[test]
fn test_measurement_measure_all_qubits_y_basis_success() {
    // |+> * |0> = |+0>
    let state = State::new_plus(1).unwrap().tensor_product(&State::new_zero(1).unwrap()).unwrap(); // |+0> state

    // Measure all qubits in the Y basis
    let measured_qubits: &[usize] = &[]; // Measure all qubits in the Y basis
    let measurement_result: MeasurementResult = state.measure(MeasurementBasis::Y, measured_qubits).unwrap();

    assert_eq!(measurement_result.get_basis(), &MeasurementBasis::Y);
    assert_eq!(measurement_result.get_indices(), &(0..state.num_qubits()).collect::<Vec<usize>>());

    let outcome: &Vec<u8> = measurement_result.get_outcomes();
    let new_state_ref = measurement_result.get_new_state();

    // Outcome can be (0, 0), (0, 1), (1, 0), or (1, 1)
    let expected_state_0_0 = State::new(vec![
        Complex::new(1.0, -1.0) / (2.0 * f64::sqrt(2.0)),
        Complex::new(1.0, 1.0) / (2.0 * f64::sqrt(2.0)),
        Complex::new(1.0, 1.0) / (2.0 * f64::sqrt(2.0)),
        Complex::new(-1.0, 1.0) / (2.0 * f64::sqrt(2.0)),
    ]).unwrap();
    
    // Corresponds to (e^{-i pi/4} |i+>) tensor |i->
    // State vector: (1/(2*sqrt(2))) * [ (1-i), -(1+i), (1+i), (1-i) ]
    let expected_state_0_1 = State::new(vec![
        Complex::new(1.0, -1.0) / (2.0 * f64::sqrt(2.0)),
        Complex::new(-1.0, -1.0) / (2.0 * f64::sqrt(2.0)),
        Complex::new(1.0, 1.0) / (2.0 * f64::sqrt(2.0)),
        Complex::new(1.0, -1.0) / (2.0 * f64::sqrt(2.0)),
    ]).unwrap();

    // Corresponds to (e^{i pi/4} |i->) tensor |i+>
    // State vector: (1/(2*sqrt(2))) * [ (1+i), (-1+i), (1-i), (1+i) ]
    let expected_state_1_0 = State::new(vec![
        Complex::new(1.0, 1.0) / (2.0 * f64::sqrt(2.0)),
        Complex::new(-1.0, 1.0) / (2.0 * f64::sqrt(2.0)),
        Complex::new(1.0, -1.0) / (2.0 * f64::sqrt(2.0)),
        Complex::new(1.0, 1.0) / (2.0 * f64::sqrt(2.0)),
    ]).unwrap();

    // Corresponds to (e^{i pi/4} |i->) tensor |i->
    // State vector: (1/(2*sqrt(2))) * [ (1+i), (1-i), (1-i), -(1+i) ]
    let expected_state_1_1 = State::new(vec![
        Complex::new(1.0, 1.0) / (2.0 * f64::sqrt(2.0)),
        Complex::new(1.0, -1.0) / (2.0 * f64::sqrt(2.0)),
        Complex::new(1.0, -1.0) / (2.0 * f64::sqrt(2.0)),
        Complex::new(-1.0, -1.0) / (2.0 * f64::sqrt(2.0)),
    ]).unwrap();

    match outcome.as_slice() {
        [0, 0] => assert_eq!(*new_state_ref, expected_state_0_0),
        [1, 0] => assert_eq!(*new_state_ref, expected_state_0_1),
        [0, 1] => assert_eq!(*new_state_ref, expected_state_1_0),
        [1, 1] => assert_eq!(*new_state_ref, expected_state_1_1),
        _ => panic!("Unexpected outcome: {:?}", outcome),
    }
}
