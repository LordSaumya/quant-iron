use crate::{errors::Error, components::{measurement::MeasurementResult, measurement::MeasurementBasis, state::State}};
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

    // Compare expected state vector with the new state vector using norm
    for (i, &value) in expected_state_vector.iter().enumerate() {
        assert!(measurement_result.get_new_state().state_vector[i].norm() - value.norm() < f64::EPSILON);
    }

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
    println!("Outcome 0: {}, Outcome 1: {}", outcome_0, outcome_1);
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