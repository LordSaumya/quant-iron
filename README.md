# Quant-Iron v1.2.0

A high-performance, hardware-accelerated modular quantum computing library with a focus on physical applications.

Quant-Iron provides tools to represent quantum states, apply standard quantum gates, perform measurements, build quantum circuits, and implement quantum algorithms.

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

- **Quantum State Representation**: Create and manipulate predefined or custom quantum states of arbitrary qubit count.

- **Standard Operations**: Hadamard (H), Pauli (X, Y, Z), CNOT, SWAP, Toffoli, Phase shifts, Rotations, and custom unitary operations.

- **Hardware Acceleration**: Optimised for parallel execution (CPU and GPU) and low memory overhead, with OpenCL-accelerated operations for enhanced performance on compatible hardware. (Requires `gpu` feature flag).
- **Circuit Builder**: High-level interface for constructing quantum circuits with a fluent API and the `circuit!` macro with support for subroutines.

- **Measurement**: Collapse wavefunction in the measurement basis with single or repeated measurements in the `Computational`, `X`, `Y`, and custom bases.

- **Pauli String Algebra**:
  - Represent products of Pauli operators with complex coefficients (`PauliString`).

  - Construct sums of Pauli strings (`SumOp`) to define Hamiltonians and other observables.

  - Apply Pauli strings and their sums to quantum states.

  - Calculate expectation values of `SumOp` with respect to a quantum state.

  - Apply exponentials of `PauliString` instances to states.

- **Predefined Quantum Models**:
  - **Heisenberg Model**: Generate Hamiltonians for 1D and 2D anisotropic Heisenberg models using `SumOp`.
  - **Ising Model**: Generate Hamiltonians for 1D and 2D Ising models with configurable site-specific or uniform interactions and fields using `SumOp`.

- **Predefined Quantum Algorithms**:
  - **Quantum Fourier Transform (QFT)**: Efficiently compute the QFT for a given number of qubits.
  - **Inverse Quantum Fourier Transform (IQFT)**: Efficiently compute the inverse QFT for a given number of qubits.

- **OpenQASM 3.0 Support**: Convert circuits to OpenQASM 3.0 for interoperability with other quantum computing platforms.

- **Extensibility**: Easily extensible for custom gates and measurement bases.

- **Error Handling**: Comprehensive error handling for invalid operations and state manipulations.

- **Quality of Life**: Implementation of `std` and arithmetic traits for easy, intuitive usage.

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

**Create a new quantum state, apply gates, and measure:**

```rust

fn qubits() {
    // Initialise a 2-qubit |++> state
    let measurement = State::new_plus(2)?
        .h(0)               // Hadamard on qubit 0
        .x(1)               // Pauli-X on qubit 1
        .h_multi(&[0, 1])   // Hadamard on both qubits
        .cnot(0, 1)         // CNOT with control=0, target=1
        .measure_n(MeasurementBasis::Computational, &[0, 1], 100)?; // Measure both qubits 100 times

    println!("Measurement results:\n{:?}", measurement);            // Print the results of the measurement
}
```

**Build a quantum circuit with a QFT subroutine and execute it on a state:**

```rust
fn circuits() {
  // Build a circuit with 3 qubits
  let circuit = CircuitBuilder::new(3)
    .h_gate(0)                                                  // Add a Hadamard gate on qubit 0
    .cnot_gate(0, 1)                                            // Add a CNOT gate with control=0 and target=1
    .x_gates(vec![1, 2])                                        // Add Pauli-X gates on qubits 1 and 2
    .add_subroutine(Subroutine::qft(vec![1, 2], 3))             // Add a QFT subroutine on qubits 1 and 2 for the 3 qubit system
    .measure_gate(MeasurementBasis::Computational, vec![0, 1])  // Measure qubits 0 and 1
    .build();                                                   // Build the circuit

  let result = circuit.execute(State::new_plus(3)?);        // Execute the circuit on the |++> state
  println!("Circuit result: {:?}", result);                 // Print the result of the circuit execution
}
```

**Use the `circuit!` macro to build a circuit:**

```rust
fn circuit_macro() {
    // Use the `circuit!` macro to build a circuit
    let circuit = circuit! {
        qubits: 3,           // Define a circuit with 3 qubits
        h(0),                // Hadamard on qubit 0
        x([1, 2]),           // Pauli-X on qubits 1 and 2
        cnot(0, 1),          // CNOT with target = 1 and control = 0
        crx([1, 2], 0, 1.0), // Controlled RX rotation with control = 1, target = 2, angle = 1.0
        measurex([0, 1]),    // Measure qubits 0 and 1 in the X basis
        measurez(2)          // Measure qubit 2 in the Z basis
    }.expect("Failed to create circuit");

    println!("{:?}", circuit); // Print the created circuit
}
```

**Define a Hamiltonian and compute its expectation value:**

```rust
fn hamiltonian() {
  // Define a Hamiltonian for a 2-qubit system
  let hamiltonian = SumOp::new()                                                  // 2 X_0 + Y_1 + 0.5 Z_0 X_1
    .with_term(PauliString::new(2.0).with_op(0, Pauli::X))                        // 2X_0
    .with_term(PauliString::new(1.0).with_op(1, Pauli::Y))                        // Y_1
    .with_term(PauliString::new(0.5).with_op(0, Pauli::Z).with_op(1, Pauli::X));  // 0.5Z_0 X_1

  let state = State::new_plus(2)?;                                // Initialise a |++> state
  let expectation_value = hamiltonian.expectation_value(&state)?; // Compute the expectation value for the given state

  println!("Expectation value: {:?}", expectation_value);         // Print the expectation value for the Hamiltonian
}
```

**Create a Hamiltonian for the 1D Heisenberg model and execute it on a state:**

```rust
fn heisenberg() {
  // Define a Hamiltonian for the 1D Heisenberg model
  let number_of_spins = 3;
  let coupling_constant_x = 1.0;
  let coupling_constant_y = 2.0;
  let coupling_constant_z = 3.0;
  let field_strength = 0.5;
  let magnetic_field = 0.1;

  let hamiltonian = heisenberg_1d(number_of_spins, coupling_constant_x, 
  coupling_constant_y, coupling_constant_z, field_strength, magnetic_field)?;

  let state = State::new_plus(3)?;                  // Initialise a |+++> state
  let modified_state = hamiltonian.apply(&state)?;  // Apply the Hamiltonian to the state
  println!("Modified state: {:?}", modified_state); // Print the modified state
}
```

**Create a circuit and convert it to OpenQASM 3.0:**

```rust
use std::f64::consts::PI;

fn qasm() {
    // Create a circuit
    let circuit = CircuitBuilder::new(5)
    .h_gate(0)
    .cx_gates(vec![1, 2], vec![0])
    .p_gates(vec![0,1,2], PI)
    .measure_gate(MeasurementBasis::X, vec![0, 1])
    .swap_gate(3, 4)
    .ctdag_gates(vec![2, 3], vec![0, 3])
    .crz_gates(vec![4], vec![0, 1], PI)
    .measure_gate(MeasurementBasis::Custom([[0.0.into(), 1.0.into()], [1.0.into(), 0.0.into()]]), vec![2, 3, 4])
    .build_final();

    // Convert circuit to OpenQASM 3.0
    println!("{}", circuit.to_qasm(None::<&str>).expect("Could not convert circuit to QASM"));
}
```

The code examples above demonstrate common use cases of Quant-Iron, and can be run using the `cargo run --example {example-name}` command.

## License

This project is licensed under the [MIT License](LICENSE).

---

## Future Plans

- **Density Matrix Support**: Extend to mixed states and density matrices for more complex quantum systems.
- **Circuit Visualisation**: Graphical representation of quantum circuits for better understanding and debugging.
- **Quantum Arithmetic & Algorithms**: Implement common subroutines (e.g. Grover's algorithm, Variational Quantum Eigensolver (VQE)).
