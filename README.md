# Quant-Iron

A high-performance, modular quantum computing library written in Rust.

Quant-Iron provides tools to represent quantum states, apply standard quantum gates, perform measurements, and build quantum circuits.

---

## Table of Contents

- [Features](#features)
- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Quickstart](#quickstart)
- [License](#license)
- [Future Plans](#future-plans)

---

## Features

- **Quantum State Representation**: Create and manipulate quantum states of arbitrary qubit count.

- **Standard Operations**: Hadamard (H), Pauli (X, Y, Z), CNOT, SWAP, Toffoli, and custom unitary operations.

- **Measurement**: Collapse wavefunction in computational basis with single or repeated measurements.

- **Speed**: Optimised for parallel execution and low memory overhead.

- **Extensibility**: Easily extensible for custom gates and measurement bases.

- **Error Handling**: Comprehensive error handling for invalid operations and state manipulations.

---

## Getting Started

### Installation

Add `quant-iron` to your Cargo.toml:

```toml
[dependencies]
quant-iron = "0.1.0"
```

Or via cargo:

```bash
cargo add quant-iron
```

### Quickstart

```rust

fn qubits() {
    // Initialise a 2-qubit |++> state
    let measurement = State::new_plus(2)?
        .h(0)               // Hadamard on qubit 0
        .x(1)               // Pauli-X on qubit 1
        .h_multi(&[0, 1])   // Hadamard on both qubits
        .cnot(0, 1)         // CNOT with control=0, target=1
        .measure_n(MeasurementBasis::Computational, &[0, 1], 100)?; // Measure both qubits 100 times

    println!("Measurement results: {:?}", measurement.outcomes); // Print the outcomes
    println!("New state: {:?}", measurement.new_state); // Print the new state after measurement
}

fn circuits() {
  // Build a circuit with 3 qubits
  let circuit = CircuitBuilder::new(3)
    .h_gate(0)                                                  // Add a Hadamard gate on qubit 0
    .cnot_gate(0, 1)                                            // Add a CNOT gate with control=0 and target=1
    .x_gates(vec![1, 2])                                        // Add Pauli-X gates on qubits 1 and 2
    .measure_gate(MeasurementBasis::Computational, vec![0, 1])  // Measure qubits 0 and 1
    .build();                                                   // Build the circuit

  let result = circuit.execute(State::new_plus(3)?);        // Execute the circuit on the |++> state
  println!("Circuit result: {:?}", result);                 // Print the result of the circuit execution
  println!("New state: {:?}", result.new_state);           // Print the new state after execution
}

```

---

## License

This project is licensed under the [GNU General Public License v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html).

---

## Future Plans

- **Measurement Bases**: Support for arbitrary measurement bases beyond the computational basis.
- **Hardware Acceleration**: GPU/CUDA or OpenCL support for large-scale state-vector simulations.
- **Density Matrix Support**: Extend to mixed states and density matrices for more complex quantum systems.
- **Circuit Visualisation**: Graphical representation of quantum circuits for better understanding and debugging.
- **Quantum Arithmetic & Algorithms**: Implement common subroutines (e.g., QFT, Grover, VQE). 
