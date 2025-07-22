/// Create a new quantum state, apply gates, and measure
use quant_iron::{ChainableState, MeasurementBasis, State};

fn main() {
    // Initialise a 2-qubit |++> state
    let state = State::new_plus(2).expect("Could not create state")
        .h(0)               // Hadamard on qubit 0
        .x(1)               // Pauli-X on qubit 1
        .h_multi(&[0, 1])   // Hadamard on both qubits
        .cnot(0, 1);        // CNOT with control=0, target=1

    // Measure both qubits 100 times
    let measurement = state.measure_n(MeasurementBasis::Computational, &[0, 1], 100)
        .expect("Could not measure computation");

    println!("Measurement results:\n{:?}", measurement);    // Print the results of the measurement
}
