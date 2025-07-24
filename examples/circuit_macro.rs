use quant_iron::circuit;

fn main() {
    // Use the `circuit!` macro to build a circuit
    let circuit = circuit! {
        qubits: 3,           // Define a circuit with 3 qubits
        h(0),                // Hadamard on qubit 0
        x([1, 2]),           // Pauli-X on qubits 1 and 2
        cnot(0, 1),          // CNOT with target = 0 and control = 1
        crx([1, 2], 0, 1.0), // Controlled RX rotation with targets = 1, 2, control = 0, angle = 1.0
        measurex([0, 1]),    // Measure qubits 0 and 1 in the X basis
        measurez(2)          // Measure qubit 2 in the Z basis
    }.expect("Failed to create circuit");

    println!("{:?}", circuit); // Print the created circuit
}