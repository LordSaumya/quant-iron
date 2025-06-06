use crate::{
    algorithms::time_evolution::{
        trotter_evolve_state, first_order_trotter_step, second_order_trotter_step, TrotterOrder,
    },
    components::{
        operator::Pauli,
        pauli_string::{PauliString, SumOp},
        state::State,
    },
    errors::Error,
};
use num_complex::Complex;

#[test]
fn test_first_order_trotter_step_success() {
    // Create test Hamiltonian: H = X1 + Y2
    let ps1: PauliString = PauliString::new(1.0.into()).with_op(0, Pauli::X);
    let ps2: PauliString = PauliString::new(1.0.into()).with_op(1, Pauli::Y);

    let hamiltonian: SumOp = SumOp::new(vec![
        ps1.clone(),
        ps2.clone(),
    ]);

    // Initial state |10>
    let initial_state = State::new_basis_n(2, 2).unwrap();

    // Time step
    let dt = 0.1;

    // Perform first-order Trotter step
    let result = first_order_trotter_step(&hamiltonian, &initial_state, dt).unwrap();

    // Expected state after applying X1 and Y2
    let mut expected_state = State::new_basis_n(2, 2).unwrap();
    expected_state = ps1.apply_exp_factor(&expected_state, Complex::new(0.0, -dt)).unwrap();
    expected_state = ps2.apply_exp_factor(&expected_state, Complex::new(0.0, -dt)).unwrap();

    // Check if the result matches the expected state
    assert_eq!(result, expected_state, "First-order Trotter step failed");
}

#[test]
fn test_first_order_trotter_step_empty_hamiltonian_error() {
    // Create an empty Hamiltonian
    let hamiltonian: SumOp = SumOp::new(vec![]);

    // Initial state |00>
    let initial_state = State::new_basis_n(2, 0).unwrap();

    // Time step
    let dt = 0.1;

    // Perform first-order Trotter step and expect an error
    let result = first_order_trotter_step(&hamiltonian, &initial_state, dt);
    
    assert!(result.is_err(), "Expected error for empty Hamiltonian");
}

#[test]
fn test_first_order_trotter_invalid_indices_error() {
    // Create a Hamiltonian with an invalid Pauli string
    let ps1: PauliString = PauliString::new(1.0.into()).with_op(0, Pauli::X);
    let ps2: PauliString = PauliString::new(1.0.into()).with_op(2, Pauli::Y); // Invalid index

    let hamiltonian: SumOp = SumOp::new(vec![ps1, ps2]);

    // Initial state |10>
    let initial_state = State::new_basis_n(2, 2).unwrap();

    // Time step
    let dt = 0.1;

    // Perform first-order Trotter step and expect an error
    let result = first_order_trotter_step(&hamiltonian, &initial_state, dt);
    
    assert!(result.is_err(), "Expected error for invalid indices in Hamiltonian");
}

#[test]
fn test_second_order_trotter_step_success() {
    // Create test Hamiltonian: H = X1 + Y2
    let ps1: PauliString = PauliString::new(1.0.into()).with_op(0, Pauli::X);
    let ps2: PauliString = PauliString::new(1.0.into()).with_op(1, Pauli::Y);

    let hamiltonian: SumOp = SumOp::new(vec![
        ps1.clone(),
        ps2.clone(),
    ]);

    // Initial state |10>
    let initial_state = State::new_basis_n(2, 2).unwrap();

    // Time step
    let dt = 0.1;

    // Perform second-order Trotter step
    let result = second_order_trotter_step(&hamiltonian, &initial_state, dt).unwrap();

    // Expected state after applying X1 and Y2 (apply PS1 with dt/2, then PS2 with dt, then PS1 again with dt/2)
    let mut expected_state = State::new_basis_n(2, 2).unwrap();
    expected_state = ps1.apply_exp_factor(&expected_state, Complex::new(0.0, -dt / 2.0)).unwrap();
    expected_state = ps2.apply_exp_factor(&expected_state, Complex::new(0.0, -dt)).unwrap();
    expected_state = ps1.apply_exp_factor(&expected_state, Complex::new(0.0, -dt / 2.0)).unwrap();

    // Check if the result matches the expected state
    assert_eq!(result, expected_state, "Second-order Trotter step failed");
}

#[test]
fn test_second_order_trotter_step_empty_hamiltonian_error() {
    // Create an empty Hamiltonian
    let hamiltonian: SumOp = SumOp::new(vec![]);

    // Initial state |00>
    let initial_state = State::new_basis_n(2, 0).unwrap();

    // Time step
    let dt = 0.1;

    // Perform second-order Trotter step and expect an error
    let result = second_order_trotter_step(&hamiltonian, &initial_state, dt);
    
    assert!(result.is_err(), "Expected error for empty Hamiltonian");
}

#[test]
fn test_second_order_trotter_invalid_indices_error() {
    // Create a Hamiltonian with an invalid Pauli string
    let ps1: PauliString = PauliString::new(1.0.into()).with_op(0, Pauli::X);
    let ps2: PauliString = PauliString::new(1.0.into()).with_op(2, Pauli::Y); // Invalid index

    let hamiltonian: SumOp = SumOp::new(vec![ps1, ps2]);

    // Initial state |10>
    let initial_state = State::new_basis_n(2, 2).unwrap();

    // Time step
    let dt = 0.1;

    // Perform second-order Trotter step and expect an error
    let result = second_order_trotter_step(&hamiltonian, &initial_state, dt);
    
    assert!(result.is_err(), "Expected error for invalid indices in Hamiltonian");
}

#[test]
fn test_trotter_evolve_state_first_order_success() {
    // Create test Hamiltonian: H = X1 + Y2
    let ps1: PauliString = PauliString::new(1.0.into()).with_op(0, Pauli::X);
    let ps2: PauliString = PauliString::new(1.0.into()).with_op(1, Pauli::Y);

    let hamiltonian: SumOp = SumOp::new(vec![
        ps1.clone(),
        ps2.clone(),
    ]);

    // Initial state |10>
    let initial_state = State::new_basis_n(2, 2).unwrap();

    // Time step
    let dt = 0.1;

    // Perform Trotter evolution
    let result = trotter_evolve_state(&hamiltonian, &initial_state, dt, 3, TrotterOrder::First).unwrap();

    // Expected state after applying X1 and Y2 (apply 1st order step 3 times)
    let mut expected_state = State::new_basis_n(2, 2).unwrap();
    for _ in 0..3 {
        expected_state = ps1.apply_exp_factor(&expected_state, Complex::new(0.0, -dt)).unwrap();
        expected_state = ps2.apply_exp_factor(&expected_state, Complex::new(0.0, -dt)).unwrap();
    }
    
    // Check if the result matches the expected state
    assert_eq!(result, expected_state, "Trotter evolution failed for first order");
}

#[test]
fn test_trotter_evolve_state_second_order_success() {
    // Create test Hamiltonian: H = X1 + Y2
    let ps1: PauliString = PauliString::new(1.0.into()).with_op(0, Pauli::X);
    let ps2: PauliString = PauliString::new(1.0.into()).with_op(1, Pauli::Y);

    let hamiltonian: SumOp = SumOp::new(vec![
        ps1.clone(),
        ps2.clone(),
    ]);

    // Initial state |10>
    let initial_state = State::new_basis_n(2, 2).unwrap();

    // Time step
    let dt = 0.1;

    // Perform Trotter evolution
    let result = trotter_evolve_state(&hamiltonian, &initial_state, dt, 3, TrotterOrder::Second).unwrap();

    // Expected state after applying X1 and Y2 (apply 2nd order step 3 times)
    let mut expected_state = State::new_basis_n(2, 2).unwrap();
    for _ in 0..3 {
        expected_state = ps1.apply_exp_factor(&expected_state, Complex::new(0.0, -dt / 2.0)).unwrap();
        expected_state = ps2.apply_exp_factor(&expected_state, Complex::new(0.0, -dt)).unwrap();
        expected_state = ps1.apply_exp_factor(&expected_state, Complex::new(0.0, -dt / 2.0)).unwrap();
    }
    
    // Check if the result matches the expected state
    assert_eq!(result, expected_state, "Trotter evolution failed for second order");
}

#[test]
fn test_trotter_evolve_state_empty_hamiltonian_error() {
    // Create an empty Hamiltonian
    let hamiltonian: SumOp = SumOp::new(vec![]);

    // Initial state |00>
    let initial_state = State::new_basis_n(2, 0).unwrap();

    // Time step
    let dt = 0.1;

    // Perform Trotter evolution and expect an error
    let result = trotter_evolve_state(&hamiltonian, &initial_state, dt, 3, TrotterOrder::First);
    
    assert!(result.is_err(), "Expected error for empty Hamiltonian in Trotter evolution");
}

#[test]
fn test_trotter_evolve_state_invalid_indices_error() {
    // Create a Hamiltonian with an invalid Pauli string
    let ps1: PauliString = PauliString::new(1.0.into()).with_op(0, Pauli::X);
    let ps2: PauliString = PauliString::new(1.0.into()).with_op(2, Pauli::Y); // Invalid index

    let hamiltonian: SumOp = SumOp::new(vec![ps1, ps2]);

    // Initial state |10>
    let initial_state = State::new_basis_n(2, 2).unwrap();

    // Time step
    let dt = 0.1;

    // Perform Trotter evolution and expect an error
    let result = trotter_evolve_state(&hamiltonian, &initial_state, dt, 3, TrotterOrder::First);
    
    assert!(result.is_err(), "Expected error for invalid indices in Hamiltonian during Trotter evolution");
}