use crate::{errors::Error, components::state::State};
use num_complex::Complex;

/// A trait defining the interface for all operators.
pub trait Operator {
    /// Applies the operator to the given state's target qubits, using the control qubits if required.
    /// 
    /// # Arguments:
    /// 
    /// * `state` - The state to apply the operator to.
    /// 
    /// * `target_qubits` - The target qubits to apply the operator to. If no target qubits are specified, the operator will be applied to all qubits in the state.
    /// 
    /// * `control_qubits` - The control qubits to apply the operator to. This is an optional argument, and is ignored if the operator does not require control qubits.
    /// 
    /// # Returns:
    /// 
    /// * The new state after applying the operator.
    fn apply(&self, state: &State, target_qubits: &[usize], control_qubits: Option<&[usize]>) -> Result<State, Error>;

    /// Returns the number of qubits that the operator acts on.
    /// 
    /// # Returns:
    /// 
    /// * The number of qubits that the operator acts on.
    fn base_qubits(&self) -> usize;
}


/// Defines a Hadamard operator.
/// 
/// A single-qubit operator that transforms the state of a qubit into a superposition of its basis states.
#[derive(Debug, Clone, Copy)]
pub struct Hadamard;

impl Operator for Hadamard {
    /// Applies the Hadamard operator to the given state's target qubit.
    /// 
    /// # Arguments:
    /// 
    /// * `state` - The state to apply the operator to.
    /// 
    /// * `target_qubits` - The target qubits to apply the operator to. This should be a single qubit.
    /// 
    /// * `control_qubits` - The control qubits to apply the operator to. This is an optional argument, and is ignored as Hadamard does not require control qubits.
    /// 
    /// # Returns:
    /// 
    /// * The new state after applying the Hadamard operator.
    /// 
    /// # Errors:
    /// 
    /// * `Error::InvalidNumberOfQubits` - If the target qubits is not 1.
    /// 
    /// * `Error::InvalidQubitIndex` - If the target qubit index is invalid for the number of qubits in the state.
    fn apply(&self, state: &State, target_qubits: &[usize], _control_qubits: Option<&[usize]>) -> Result<State, Error> {
        // Validation
        if target_qubits.len() != 1 {
            return Err(Error::InvalidNumberOfQubits(target_qubits.len()));
        }

        let target_qubit: usize = target_qubits[0];

        if target_qubit >= state.num_qubits() {
            return Err(Error::InvalidQubitIndex(target_qubit, state.num_qubits()));
        }

        // Apply Hadamard operator
        let sqrt_2_inv: f64 = 1.0 / (2.0f64).sqrt();
        let dim: usize = 1 << state.num_qubits();
        let mut new_state: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); dim];

        for i in 0..dim {
            if i & (1 << target_qubit) == 0 {
                let j: usize = i | (1 << target_qubit);
                let amp_i: Complex<f64> = state.amplitude(i)?;
                let amp_j: Complex<f64> = state.amplitude(j)?;

                new_state[i as usize] = sqrt_2_inv * (amp_i + amp_j);
                new_state[j as usize] = sqrt_2_inv * (amp_i - amp_j);
            }
            // Handle pairs of indices, so skip when bit = 1.
        }

        Ok(State {
            state_vector: new_state,
            num_qubits: state.num_qubits(),
        })
    }

    fn base_qubits(&self) -> usize {
        1 // Hadamard acts on 1 qubit only
    }
}

/// Defines the Pauli operators: X, Y, Z.
#[derive(Debug, Clone, Copy)]
pub enum Pauli {
    /// Pauli-X operator (NOT gate)
    X,
    /// Pauli-Y operator
    Y,
    /// Pauli-Z operator
    Z,
}

impl Operator for Pauli {
    /// Applies the Pauli operator to the given state's target qubit.
    /// 
    /// # Arguments:
    /// 
    /// * `state` - The state to apply the operator to.
    /// 
    /// * `target_qubits` - The target qubits to apply the operator to. This should be a single qubit.
    /// 
    /// * `control_qubits` - The control qubits to apply the operator to. This is an optional argument, and is ignored as Pauli does not require control qubits.
    /// 
    /// # Returns:
    /// 
    /// * The new state after applying the Pauli operator.
    /// 
    /// # Errors:
    /// 
    /// * `Error::InvalidNumberOfQubits` - If the target qubits is not 1.
    /// 
    /// * `Error::InvalidQubitIndex` - If the target qubit index is invalid for the number of qubits in the state.
    fn apply(&self, state: &State, target_qubits: &[usize], _control_qubits: Option<&[usize]>) -> Result<State, Error> {
        // Validation
        if target_qubits.len() != 1 {
            return Err(Error::InvalidNumberOfQubits(target_qubits.len()));
        }

        let target_qubit: usize = target_qubits[0];

        if target_qubit >= state.num_qubits() {
            return Err(Error::InvalidQubitIndex(target_qubit, state.num_qubits()));
        }

        // Apply Pauli operator
        let dim: usize = 1 << state.num_qubits();
        let mut new_state: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); dim as usize];

        for i in 0..dim {
            let target_bit: usize = 1 << target_qubit;

            match self {
                Pauli::X => {
                    // Swap the amplitudes of the target qubit
                    let j: usize = i ^ (1 << target_qubit); // Index with the target bit flipped
                    new_state[j] = state.state_vector[i]; // Swap the amplitudes
                }

                Pauli::Y => {
                    // Flip target qubit and apply phase
                    let j: usize = i ^ (1 << target_qubit); // Index with the target bit flipped
                    if target_bit == 0 {
                        new_state[j] = state.state_vector[i] * Complex::new(0.0, 1.0);
                    } else {
                        new_state[j] = state.state_vector[i] * Complex::new(0.0, -1.0);
                    }
                }

                Pauli::Z => {
                    // Apply a phase to the target qubit if it is 1
                    if target_bit == 0 {
                        new_state[i] = state.state_vector[i]; // No change
                    } else {
                        new_state[i] = -state.state_vector[i]; // Phase flip
                    }
                }
            }
        }

        Ok(State {
            state_vector: new_state,
            num_qubits: state.num_qubits(),
        })
    }

    fn base_qubits(&self) -> usize {
        1 // Pauli operators act on 1 qubit only
    }
}

/// Defines a CNOT operator.
/// 
/// A two-qubit operator that flips the target qubit if the control qubit is in the |1> state.
#[derive(Debug, Clone, Copy)]
pub struct CNOT;

impl Operator for CNOT {
    /// Applies the CNOT operator to the given state's target qubit, using the control qubit.
    /// 
    /// # Arguments:
    /// 
    /// * `state` - The state to apply the operator to.
    /// 
    /// * `target_qubits` - The target qubits to apply the operator to. This should be a single qubit.
    /// 
    /// * `control_qubits` - The control qubits to apply the operator to. This should be a single qubit.
    /// 
    /// # Returns:
    /// 
    /// * The new state after applying the CNOT operator.
    /// 
    /// # Errors:
    /// 
    /// * `Error::InvalidNumberOfQubits` - If the target or control qubits is not 1.
    /// 
    /// * `Error::InvalidQubitIndex` - If the target or control qubit index is invalid for the number of qubits in the state.
    fn apply(&self, state: &State, target_qubits: &[usize], control_qubits: Option<&[usize]>) -> Result<State, Error> {
        // Validation
        if target_qubits.len() != 1 || control_qubits.is_none() || control_qubits.unwrap().len() != 1 {
            return Err(Error::InvalidNumberOfQubits(target_qubits.len()));
        }

        let target_qubit: usize = target_qubits[0];
        let control_qubit: usize = control_qubits.unwrap()[0];

        if target_qubit >= state.num_qubits() || control_qubit >= state.num_qubits() {
            return Err(Error::InvalidQubitIndex(target_qubit, state.num_qubits()));
        }

        // Apply CNOT operator
        let dim: usize = 1 << state.num_qubits();
        let mut new_state: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); dim as usize];

        for i in 0..dim {
            let target_bit: usize = 1 << target_qubit;
            let control_bit: usize = (i >> control_qubit) & 1;

            if control_bit == 1 {
                // Flip the target qubit if the control qubit is 1
                let j: usize = i ^ (1 << target_bit); // Index with the target bit flipped
                new_state[j] = state.state_vector[i]; // Swap the amplitudes
            } else {
                // No change if control qubit is 0
                new_state[i] = state.state_vector[i];
            }
        }

        Ok(State {
            state_vector: new_state,
            num_qubits: state.num_qubits(),
        })
    }

    fn base_qubits(&self) -> usize {
        2 // CNOT acts on 2 qubits (1 control, 1 target)
    }
}

/// Defines a SWAP operator.
/// 
/// A two-qubit operator that swaps the states of the two qubits.
#[derive(Debug, Clone, Copy)]
pub struct SWAP;

impl Operator for SWAP {
    /// Applies the SWAP operator to the given state's target qubits.
    /// 
    /// # Arguments:
    /// 
    /// * `state` - The state to apply the operator to.
    /// 
    /// * `target_qubits` - The target qubits to apply the operator to. This should be two qubits.
    /// 
    /// * `control_qubits` - The control qubits to apply the operator to. This is an optional argument, and is ignored as SWAP does not require control qubits.
    /// 
    /// # Returns:
    /// 
    /// * The new state after applying the SWAP operator.
    /// 
    /// # Errors:
    /// 
    /// * `Error::InvalidNumberOfQubits` - If the target qubits are not 2 different qubits.
    /// 
    /// * `Error::InvalidQubitIndex` - If the target qubit indices are invalid for the number of qubits in the state.
    /// 
    /// * `Error::InvalidQubitIndex` - If the target qubit indices are not different.
    fn apply(&self, state: &State, target_qubits: &[usize], _control_qubits: Option<&[usize]>) -> Result<State, Error> {
        // Validation
        if target_qubits.len() != 2 {
            return Err(Error::InvalidNumberOfQubits(target_qubits.len()));
        }

        let target_qubit_1: usize = target_qubits[0];
        let target_qubit_2: usize = target_qubits[1];

        if target_qubit_1 >= state.num_qubits() || target_qubit_2 >= state.num_qubits() {
            return Err(Error::InvalidQubitIndex(target_qubit_1, state.num_qubits()));
        }
        if target_qubit_1 == target_qubit_2 {
            return Err(Error::InvalidQubitIndex(target_qubit_1, state.num_qubits()));
        }

        // Apply SWAP operator
        let dim: usize = 1 << state.num_qubits();
        let mut new_state: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); dim as usize];

        for i in 0..dim {
            let target_bit_1: usize = i >> (target_qubit_1) & 1;
            let target_bit_2: usize = i >> (target_qubit_2) & 1;

            if target_bit_1 == target_bit_2 {
                // No change if both target bits are the same
                new_state[i] = state.state_vector[i];
                continue;
            }

            // Swap the amplitudes of the target qubits if they are different
            let j: usize = i ^ (1 << target_qubit_1) ^ (1 << target_qubit_2); // Index with the target bits flipped
            new_state[j] = state.state_vector[i]; // Swap the amplitudes
        }

        Ok(State {
            state_vector: new_state,
            num_qubits: state.num_qubits(),
        })
    }

    fn base_qubits(&self) -> usize {
        2 // SWAP acts on 2 qubits
    }
}

/// Defines a Toffoli operator.
/// 
/// A three-qubit operator that flips the target qubit if both control qubits are in the |1> state. Also known as CCNOT (Controlled-Controlled-NOT).
#[derive(Debug, Clone, Copy)]
pub struct Toffoli;

impl Operator for Toffoli {
    /// Applies the Toffoli operator to the given state's target qubit, using the control qubits.
    /// 
    /// # Arguments:
    /// 
    /// * `state` - The state to apply the operator to.
    /// 
    /// * `target_qubits` - The target qubit to apply the operator to. This should be a single qubit.
    /// 
    /// * `control_qubits` - The control qubits to apply the operator to. This should be two qubits.
    /// 
    /// # Returns:
    /// 
    /// * The new state after applying the Toffoli operator.
    /// 
    /// # Errors:
    /// 
    /// * `Error::InvalidNumberOfQubits` - If the target or control qubits are not 1 and 2 respectively, or if the control qubits are not different.
    /// 
    /// * `Error::InvalidQubitIndex` - If the target or control qubit indices are invalid for the number of qubits in the state.
    fn apply(&self, state: &State, target_qubits: &[usize], control_qubits: Option<&[usize]>) -> Result<State, Error> {
        // Validation
        if target_qubits.len() != 1 || control_qubits.is_none() || control_qubits.unwrap().len() != 2 {
            return Err(Error::InvalidNumberOfQubits(target_qubits.len()));
        }

        let target_qubit: usize = target_qubits[0];
        let control_qubit_1: usize = control_qubits.unwrap()[0];
        let control_qubit_2: usize = control_qubits.unwrap()[1];

        if target_qubit >= state.num_qubits() || control_qubit_1 >= state.num_qubits() || control_qubit_2 >= state.num_qubits() {
            return Err(Error::InvalidQubitIndex(target_qubit, state.num_qubits()));
        }
        if control_qubit_1 == control_qubit_2 {
            return Err(Error::InvalidQubitIndex(control_qubit_1, state.num_qubits()));
        }

        let dim: usize = 1 << state.num_qubits();
        let mut new_state: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); dim as usize];

        for i in 0..dim {
            let control_bit_1: usize = (i >> control_qubit_1) & 1;
            let control_bit_2: usize = (i >> control_qubit_2) & 1;

            if control_bit_1 == 1 && control_bit_2 == 1 {
                // Flip the target qubit if both control qubits are 1
                let j: usize = i ^ (1 << target_qubit); // Index with the target bit flipped
                new_state[j] = state.state_vector[i]; // Swap the amplitudes
            } else {
                // No change if both control qubits are not 1
                new_state[i] = state.state_vector[i];
            }
        }

        Ok(State {
            state_vector: new_state,
            num_qubits: state.num_qubits(),
        })
    }

    fn base_qubits(&self) -> usize {
        3 // Toffoli acts on 3 qubits (2 control, 1 target)
    }
}