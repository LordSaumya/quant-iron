use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId, black_box};
use num_complex::Complex;

// Imports based on src/lib.rs and README examples
use quant_iron::{State, MeasurementBasis, Pauli}; // Re-exported in lib.rs
use quant_iron::circuit::CircuitBuilder;
use quant_iron::subroutine::Subroutine;
use quant_iron::components::pauli_string::{PauliString, SumOp}; // Corrected path
use quant_iron::models::heisenberg::heisenberg_1d;
use quant_iron::errors::Error as QuantIronError; // Alias to avoid conflict if Criterion has an Error type

// --- Adapted README functions ---

fn run_qubits_logic(num_qubits_for_state: usize) -> Result<(), QuantIronError> {
    // For this example, the operations are on specific qubits (0, 1).
    // We'll ensure num_qubits_for_state is at least 2.
    // If num_qubits_for_state is larger, the operations still apply to qubits 0 and 1.
    let actual_num_qubits = std::cmp::max(2, num_qubits_for_state);
    let measured_qubits: Vec<usize> = (0..std::cmp::min(actual_num_qubits, 2)).collect();


    let _measurement_result = State::new_plus(actual_num_qubits)?
        .h(0)?
        .x(1)?
        .h_multi(&[0, 1])?
        .cnot(0, 1)?
        .measure_n(MeasurementBasis::Computational, &measured_qubits, 100)?; // Measure 100 times
    Ok(())
}

fn run_circuits_logic(num_qubits_for_circuit: usize) -> Result<(), QuantIronError> {
    // This circuit example uses 3 qubits specifically for QFT on [1,2] and measures [0,1].
    // For scaling, we'll adapt it. If num_qubits_for_circuit is N:
    // - H on 0
    // - CNOT 0, 1
    // - X on 1, 2 (if N >= 3)
    // - QFT on [1, N-1] (if N >= 2 for QFT target, N-1 must be >= 1)
    // - Measure [0, min(1, N-1)]
    let actual_num_qubits = std::cmp::max(3, num_qubits_for_circuit); // Ensure at least 3 for original logic

    let mut builder = CircuitBuilder::new(actual_num_qubits);
    builder.h_gate(0).cnot_gate(0, 1); // Qubits 0 and 1 must exist.

    if actual_num_qubits >= 3 {
        builder.x_gates(vec![1, 2]); // Original X gates
    }

    if actual_num_qubits >= 2 { // QFT needs at least 1 target qubit
        let qft_targets: Vec<usize> = (1..actual_num_qubits).collect();
        if !qft_targets.is_empty() {
             builder.add_subroutine(Subroutine::qft(qft_targets, actual_num_qubits));
        }
    }
    
    let mut measure_targets = vec![0];
    if actual_num_qubits >= 2 {
        measure_targets.push(1);
    }
    builder.measure_gate(MeasurementBasis::Computational, measure_targets);

    let circuit = builder.build()?;
    let initial_state = State::new_plus(actual_num_qubits)?;
    let _final_state = circuit.execute(&initial_state)?; // execute returns Result<State, Error>
    Ok(())
}

fn run_hamiltonian_logic(num_qubits_for_hamiltonian: usize) -> Result<(), QuantIronError> {
    // Original example is for 2 qubits.
    // For scaling, we'll keep the structure but ensure qubit indices are valid.
    let actual_num_qubits = std::cmp::max(2, num_qubits_for_hamiltonian);

    // 2 X_0 + Y_1 + 0.5 Z_0 X_1
    let term1 = PauliString::new(Complex::new(2.0, 0.0))
        .with_op(0, Pauli::X);
    let term2 = PauliString::new(Complex::new(1.0, 0.0))
        .with_op(1, Pauli::Y); // Qubit 1 must exist
    let term3 = PauliString::new(Complex::new(0.5, 0.0))
        .with_op(0, Pauli::Z)
        .with_op(1, Pauli::X); // Qubit 1 must exist

    let hamiltonian = SumOp::new(vec![term1, term2, term3]);

    let state = State::new_plus(actual_num_qubits)?;
    let _expectation_value = hamiltonian.expectation_value(&state)?;
    Ok(())
}

fn run_heisenberg_logic(number_of_spins: usize) -> Result<(), QuantIronError> {
    let actual_n_spins = std::cmp::max(2, number_of_spins); // heisenberg_1d needs n >= 2

    let coupling_constant_x = 1.0;
    let coupling_constant_y = 2.0;
    let coupling_constant_z = 3.0;
    let field_strength = 0.5; // h in the formula
    let magnetic_moment = 0.1; // mu in the formula

    let hamiltonian = heisenberg_1d(
        actual_n_spins,
        coupling_constant_x,
        coupling_constant_y,
        coupling_constant_z,
        field_strength,
        magnetic_moment,
    )?;

    let state = State::new_plus(actual_n_spins)?;
    let _modified_state = hamiltonian.apply(&state)?;
    Ok(())
}

// --- Benchmark Functions ---

fn benchmark_readme_examples(c: &mut Criterion) {
    let mut group = c.benchmark_group("README Examples");

    // Qubits example (original size implied by ops on 0,1)
    group.bench_function(BenchmarkId::new("qubits_logic", "2_qubits"), |b| {
        b.iter(|| black_box(run_qubits_logic(2).unwrap()))
    });

    // Circuits example (original size 3 qubits)
    group.bench_function(BenchmarkId::new("circuits_logic", "3_qubits"), |b| {
        b.iter(|| black_box(run_circuits_logic(3).unwrap()))
    });

    // Hamiltonian example (original size 2 qubits)
    group.bench_function(BenchmarkId::new("hamiltonian_logic", "2_qubits"), |b| {
        b.iter(|| black_box(run_hamiltonian_logic(2).unwrap()))
    });
    
    // Heisenberg example (original size 3 spins)
    group.bench_function(BenchmarkId::new("heisenberg_logic", "3_spins"), |b| {
        b.iter(|| black_box(run_heisenberg_logic(3).unwrap()))
    });

    group.finish();
}

fn benchmark_scaled_examples(c: &mut Criterion) {
    let mut group = c.benchmark_group("Scaled Examples");
    let qubit_counts = [11, 20]; // For Rayon and GPU thresholds

    for &n_qubits in &qubit_counts {
        let mut bench_group = &mut group;

        if n_qubits == 20 {
            bench_group = bench_group.sample_size(10);
        }

        bench_group.bench_function(BenchmarkId::new("qubits_logic_scaled", n_qubits), |b| {
            b.iter(|| black_box(run_qubits_logic(n_qubits).unwrap()))
        });

        // For now, we scale the number of qubits for the state and adapt circuit construction.
        bench_group.bench_function(BenchmarkId::new("circuits_logic_scaled", n_qubits), |b| {
            b.iter(|| black_box(run_circuits_logic(n_qubits).unwrap()))
        });
        
        // Hamiltonian example scales with the state, PauliStrings still on 0,1
        bench_group.bench_function(BenchmarkId::new("hamiltonian_logic_scaled", n_qubits), |b| {
            b.iter(|| black_box(run_hamiltonian_logic(n_qubits).unwrap()))
        });

        // Heisenberg model scales directly with number of spins
        if n_qubits >= 2 { // heisenberg_1d requires at least 2 spins
            bench_group.bench_function(BenchmarkId::new("heisenberg_logic_scaled", n_qubits), |b| {
                b.iter(|| black_box(run_heisenberg_logic(n_qubits).unwrap()))
            });
        }
    }
    group.finish();
}

criterion_group!(
    benches,
    benchmark_readme_examples,
    benchmark_scaled_examples
);
criterion_main!(benches);
