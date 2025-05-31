use crate::{
    components::{
        pauli_string::SumOp,
        state::State,
    },
    errors::Error,
};
use num_complex::Complex;

/// Trotter decomposition orders.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrotterOrder {
    /// First-order Trotter decomposition.
    ///
    /// For H = Σⱼ Hⱼ, approximates exp(-iHt) ≈ Πⱼ exp(-iHⱼt).
    /// This is the simplest and fastest method but has O(t²) error per step.
    First,
    
    /// Second-order Trotter decomposition.
    ///
    /// Uses the symmetric Suzuki formula for improved accuracy with O(t³) error per step.
    /// Requires twice as many exponential operations as first-order.
    Second,
}

/// Performs a single first-order Trotter step for time evolution.
///
/// Implements the first-order Trotter formula: exp(-iHdt) ≈ Πⱼ exp(-iHⱼdt),
/// where H = Σⱼ Hⱼ is the Hamiltonian decomposed into a sum of terms.
///
/// # Arguments
///
/// * `hamiltonian` - The Hamiltonian operator represented as a `SumOp` (sum of Pauli strings)
/// * `initial_state` - The quantum state to evolve
/// * `dt` - The time step for evolution
///
/// # Returns
///
/// * `Result<State, Error>` - The evolved state after applying the Trotter step, or an error if the operation fails
///
/// # Errors
///
/// * Returns an error if any Pauli string operations fail (e.g., invalid qubit indices)
/// * Returns an error if the Hamiltonian contains no terms
pub fn first_order_trotter_step(
    hamiltonian: &SumOp,
    initial_state: &State,
    dt: f64,
) -> Result<State, Error> {
    // Validate inputs
    if hamiltonian.num_terms() == 0 {
        return Err(Error::InvalidNumberOfQubits(0));
    }

    // Complex factor for -i*dt
    let factor = Complex::new(0.0, -dt);
    
    // Apply each term in the Hamiltonian sequentially
    let mut current_state = initial_state.clone();
    
    for term in &hamiltonian.terms {
        current_state = term.apply_exp_factor(&current_state, factor)?;
    }
    
    Ok(current_state)
}

/// Performs a single second-order Trotter step for time evolution.
///
/// Implements the second-order symmetric Trotter formula for improved accuracy.
/// The symmetric decomposition reduces the leading error term, providing O(t³)
/// accuracy compared to O(t²) for first-order.
///
/// # Arguments
///
/// * `hamiltonian` - The Hamiltonian operator represented as a sum of Pauli strings
/// * `initial_state` - The quantum state to evolve
/// * `dt` - The time step for evolution
///
/// # Returns
///
/// * `Result<State, Error>` - The evolved state after applying the second-order Trotter step,
///   or an error if the operation fails
///
/// # Errors
///
/// * Returns an error if any Pauli string operations fail
/// * Returns an error if the Hamiltonian contains no terms
pub fn second_order_trotter_step(
    hamiltonian: &SumOp,
    initial_state: &State,
    dt: f64,
) -> Result<State, Error> {
    // Validate inputs
    if hamiltonian.num_terms() == 0 {
        return Err(Error::InvalidNumberOfQubits(0));
    }

    // Complex factor for -i*dt/2
    let half_factor = Complex::new(0.0, -dt / 2.0);
    
    let mut current_state = initial_state.clone();
    
    // First half: apply each term with dt/2
    for term in &hamiltonian.terms {
        current_state = term.apply_exp_factor(&current_state, half_factor)?;
    }
    
    // Second half: apply each term in reverse order with dt/2
    for term in hamiltonian.terms.iter().rev() {
        current_state = term.apply_exp_factor(&current_state, half_factor)?;
    }
    
    Ok(current_state)
}

/// Evolves a quantum state under a given Hamiltonian using Trotter decomposition.
///
/// This is the main time evolution function that applies the specified Trotter method
/// for the given number of steps with the specified time step size.
/// The total evolution approximates exp(-iH*dt*num_steps)|ψ⟩.
///
/// # Arguments
///
/// * `hamiltonian` - The Hamiltonian operator as a sum of Pauli strings
/// * `initial_state` - The initial quantum state to evolve
/// * `dt` - The time step for each evolution step
/// * `num_steps` - The number of discrete time steps to use
/// * `order` - The order of Trotter decomposition to apply
///
/// # Returns
///
/// * `Result<State, Error>` - The final evolved state after the complete time evolution,
///   or an error if any step fails
///
/// # Errors
///
/// * Returns an error if any individual Trotter step fails
/// * Returns an error if the Hamiltonian does not contain any terms
pub fn trotter_evolve_state(
    hamiltonian: &SumOp,
    initial_state: &State,
    dt: f64,
    num_steps: usize,
    order: TrotterOrder,
) -> Result<State, Error> {
    // Validate inputs
    if hamiltonian.num_terms() == 0 {
        return Err(Error::InvalidNumberOfQubits(0));
    }
    
    // Apply Trotter steps iteratively
    let mut current_state = initial_state.clone();
    
    for _ in 0..num_steps {
        current_state = match order {
            TrotterOrder::First => {
                first_order_trotter_step(hamiltonian, &current_state, dt)?
            },
            TrotterOrder::Second => {
                second_order_trotter_step(hamiltonian, &current_state, dt)?
            },
        };
    }
    
    Ok(current_state)
}