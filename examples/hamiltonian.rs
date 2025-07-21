use quant_iron::{Pauli, PauliString, State, SumOp};

/// Define a Hamiltonian and compute its expectation value
fn main() {
    // Define a Hamiltonian for a 2-qubit system
    let hamiltonian = SumOp::new(Vec::default())                                             // 2 X_0 + Y_1 + 0.5 Z_0 X_1
        .with_term(PauliString::new(2.0.into()).with_op(0, Pauli::X))                        // 2X_0
        .with_term(PauliString::new(1.0.into()).with_op(1, Pauli::Y))                        // Y_1
        .with_term(PauliString::new(0.5.into()).with_op(0, Pauli::Z).with_op(1, Pauli::X));  // 0.5Z_0 X_1

    // Initialise a |++> state
    let state = State::new_plus(2).expect("Could not initialize state");
    // Compute the expectation value for the given state
    let expectation_value = hamiltonian.expectation_value(&state)
        .expect("Could not get expectation value");
    // Print the expectation value for the Hamiltonian
    println!("Expectation value: {:?}", expectation_value);
}
