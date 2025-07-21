use quant_iron::{CircuitBuilder, MeasurementBasis, State, Subroutine};

/// Build a quantum circuit with a QFT subroutine and execute it on a state

fn main() {
    // Build a circuit with 3 qubits
    let circuit = CircuitBuilder::new(3)
        .h_gate(0)                                                  // Add a Hadamard gate on qubit 0
        .cnot_gate(0, 1)                                            // Add a CNOT gate with control=0 and target=1
        .x_gates(vec![1, 2])                                        // Add Pauli-X gates on qubits 1 and 2
        .add_subroutine(Subroutine::qft(vec![1, 2], 3))             // Add a QFT subroutine on qubits 1 and 2 for the 3-qubit system
        .measure_gate(MeasurementBasis::Computational, vec![0, 1])  // Measure qubits 0 and 1
        .build()
        .expect("Could not build Circuit");                         // Build the circuit

    let state = State::new_plus(3).expect("Could not create state");
    println!("Initial state:\n{:?}", state);            // Print the new state after execution

    let new_state = circuit.execute(&state)
        .expect("Could not execute circuit");          // Execute the circuit on the |++> state
    println!("New state:\n{:?}", new_state);            // Print the new state after execution
}
