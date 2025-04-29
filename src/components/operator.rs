use crate::{components::state::State, errors::Error};
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
    /// * `control_qubits` - The control qubits to apply the operator to.
    ///
    /// # Returns:
    ///
    /// * The new state after applying the operator.
    fn apply(
        &self,
        state: &State,
        target_qubits: &[usize],
        control_qubits: &[usize],
    ) -> Result<State, Error>;

    /// Returns the number of qubits that the operator acts on.
    ///
    /// # Returns:
    ///
    /// * The number of qubits that the operator acts on.
    fn base_qubits(&self) -> usize;
}

/// Helper function to check if all control qubits are in the |1> state for a given basis state index.
fn check_controls(index: usize, control_qubits: &[usize]) -> bool {
    for &control_qubit in control_qubits {
        if (index >> control_qubit) & 1 == 0 {
            return false; // At least one control qubit is |0>
        }
    }
    true // All control qubits are |1>
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
    /// * `control_qubits` - The control qubits for the operator. If not empty, the operator will be applied conditionally based on the control qubits. Otherwise, it will be applied unconditionally.
    ///
    /// # Returns:
    ///
    /// * The new state after applying the Hadamard operator.
    ///
    /// # Errors:
    ///
    /// * `Error::InvalidNumberOfQubits` - If the target qubits is not 1.
    ///
    /// * `Error::InvalidQubitIndex` - If the target qubit or control qubit index is invalid for the number of qubits in the state.
    ///
    /// * `Error::OverlappingControlAndTargetQubits` - If the control qubit and target qubit indices overlap.
    fn apply(
        &self,
        state: &State,
        target_qubits: &[usize],
        control_qubits: &[usize],
    ) -> Result<State, Error> {
        // Validation
        if target_qubits.len() != 1 {
            return Err(Error::InvalidNumberOfQubits(target_qubits.len()));
        }

        let target_qubit: usize = target_qubits[0];

        if target_qubit >= state.num_qubits() {
            return Err(Error::InvalidQubitIndex(target_qubit, state.num_qubits()));
        }

        for &control_qubit in control_qubits {
            if control_qubit >= state.num_qubits() {
                return Err(Error::InvalidQubitIndex(control_qubit, state.num_qubits()));
            }
            if control_qubit == target_qubit {
                return Err(Error::OverlappingControlAndTargetQubits(
                    control_qubit,
                    target_qubit,
                ));
            }
        }

        // Apply potentially controlled Hadamard operator
        let sqrt_2_inv: f64 = 1.0 / (2.0f64).sqrt();
        let dim: usize = 1 << state.num_qubits();
        let mut new_state: Vec<Complex<f64>> = state.state_vector.clone(); // Start with a copy

        for i in 0..dim {
            // Process pairs (i, j) where target bit differs.
            if (i >> target_qubit) & 1 == 0 {
                // Process pairs starting from |..0..>
                let j = i | (1 << target_qubit); // Index where target qubit is |1>

                // Check controls for both i and j.
                let i_controls_met: bool = check_controls(i, control_qubits);
                let j_controls_met: bool = check_controls(j, control_qubits);

                if i_controls_met && j_controls_met {
                    // Apply Hadamard logic if controls met for both i and j
                    let amp_i: Complex<f64> = state.amplitude(i)?; // Use original state amplitudes
                    let amp_j: Complex<f64> = state.amplitude(j)?;

                    new_state[i] = sqrt_2_inv * (amp_i + amp_j);
                    new_state[j] = sqrt_2_inv * (amp_i - amp_j);
                }
            }
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
    /// * `control_qubits` - The control qubits for the operator. If not empty, the operator will be applied conditionally based on the control qubits. Otherwise, it will be applied unconditionally.
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
    ///
    /// * `Error::OverlappingControlAndTargetQubits` - If the control qubit and target qubit indices overlap.
    fn apply(
        &self,
        state: &State,
        target_qubits: &[usize],
        control_qubits: &[usize],
    ) -> Result<State, Error> {
        // Validation
        if target_qubits.len() != 1 {
            return Err(Error::InvalidNumberOfQubits(target_qubits.len()));
        }
        let target_qubit: usize = target_qubits[0];
        if target_qubit >= state.num_qubits() {
            return Err(Error::InvalidQubitIndex(target_qubit, state.num_qubits()));
        }
        for &control in control_qubits {
            if control >= state.num_qubits() {
                return Err(Error::InvalidQubitIndex(control, state.num_qubits()));
            }
            if control == target_qubit {
                return Err(Error::OverlappingControlAndTargetQubits(
                    control,
                    target_qubit,
                ));
            }
        }

        // Apply potentially controlled Pauli operator
        let dim: usize = 1 << state.num_qubits();
        let mut new_state: Vec<Complex<f64>> = state.state_vector.clone();

        let i_complex = Complex::new(0.0, 1.0);

        for i in 0..dim {
            // Check if controls are met for this basis state index
            if check_controls(i, control_qubits) {
                match self {
                    Pauli::X => {
                        // Process pairs (i, j) where target bit differs.
                        if (i >> target_qubit) & 1 == 0 {
                            // Process pairs starting from |..0..>
                            let j = i | (1 << target_qubit); // Index where target qubit is |1>
                            if check_controls(j, control_qubits) {
                                let amp_i: Complex<f64> = state.amplitude(i)?;
                                let amp_j: Complex<f64> = state.amplitude(j)?;
                                new_state[i] = amp_j; // Swap amplitudes
                                new_state[j] = amp_i;
                            }
                        }
                    }

                    Pauli::Y => {
                        // Process pairs (i, j) where target bit differs.
                        if (i >> target_qubit) & 1 == 0 {
                            // Process pairs starting from |..0..>
                            let j = i | (1 << target_qubit); // Index where target qubit is |1>
                            if check_controls(j, control_qubits) {
                                let amp_i = state.amplitude(i)?;
                                let amp_j = state.amplitude(j)?;
                                new_state[i] = -i_complex * amp_j; // Apply Y logic
                                new_state[j] = i_complex * amp_i;
                            }
                        }
                    }

                    Pauli::Z => {
                        // Apply phase flip only if target qubit is |1>
                        if (i >> target_qubit) & 1 == 1 {
                            // Controls already checked for index 'i' at the start of the loop
                            new_state[i] = -state.amplitude(i)?; // Phase flip
                        }
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
    /// * `control_qubits` - The control qubits for the operator. This should be a single qubit.
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
    ///
    /// * `Error::OverlappingControlAndTargetQubits` - If the control qubit and target qubit indices overlap.
    fn apply(
        &self,
        state: &State,
        target_qubits: &[usize],
        control_qubits: &[usize],
    ) -> Result<State, Error> {
        // Validation
        if target_qubits.len() != 1 || control_qubits.len() != 1 {
            return Err(Error::InvalidNumberOfQubits(target_qubits.len()));
        }
        if control_qubits.len() != 1 {
            return Err(Error::InvalidNumberOfQubits(control_qubits.len()));
        }
        let target_qubit: usize = target_qubits[0];
        let control_qubit: usize = control_qubits[0];

        if target_qubit >= state.num_qubits() {
            return Err(Error::InvalidQubitIndex(target_qubit, state.num_qubits()));
        }
        if control_qubit >= state.num_qubits() {
            return Err(Error::InvalidQubitIndex(control_qubit, state.num_qubits()));
        }
        if target_qubit == control_qubit {
            return Err(Error::OverlappingControlAndTargetQubits(
                control_qubit,
                target_qubit,
            ));
        }

        // Apply CNOT operator (same as Pauli-X with 1 control qubit)
        Pauli::X.apply(state, target_qubits, &[control_qubit])
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
    /// * `control_qubits` - The control qubits. If empty, the swap is unconditional. Otherwise, the swap occurs only if all control qubits are |1> for the relevant basis states.
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
    ///
    /// * `Error::OverlappingControlAndTargetQubits` - If the control qubit and target qubit indices overlap.
    fn apply(
        &self,
        state: &State,
        target_qubits: &[usize],
        control_qubits: &[usize],
    ) -> Result<State, Error> {
        // Validation: Target Qubits
        if target_qubits.len() != 2 {
            return Err(Error::InvalidNumberOfQubits(target_qubits.len()));
        }
        let target_qubit_1: usize = target_qubits[0];
        let target_qubit_2: usize = target_qubits[1];
        let num_qubits: usize = state.num_qubits();

        if target_qubit_1 >= num_qubits {
            return Err(Error::InvalidQubitIndex(target_qubit_1, num_qubits));
        }
        if target_qubit_2 >= num_qubits {
            return Err(Error::InvalidQubitIndex(target_qubit_2, num_qubits));
        }
        if target_qubit_1 == target_qubit_2 {
            // Indicate duplicate target qubits
            return Err(Error::InvalidQubitIndex(target_qubit_1, num_qubits));
        }

        // Validation: Control Qubits
        for &control_qubit in control_qubits {
            if control_qubit >= num_qubits {
                return Err(Error::InvalidQubitIndex(control_qubit, num_qubits));
            }
            if control_qubit == target_qubit_1 {
                return Err(Error::OverlappingControlAndTargetQubits(
                    control_qubit,
                    target_qubit_1,
                ));
            }
            if control_qubit == target_qubit_2 {
                return Err(Error::OverlappingControlAndTargetQubits(
                    control_qubit,
                    target_qubit_2,
                ));
            }
        }

        // Apply potentially controlled SWAP operator
        let dim: usize = 1 << num_qubits;
        let mut new_state = state.state_vector.clone(); // Start with a copy

        for i in 0..dim {
            // Check if target bits are different for this basis state index 'i'
            let target_bit_1 = (i >> target_qubit_1) & 1;
            let target_bit_2 = (i >> target_qubit_2) & 1;

            if target_bit_1 != target_bit_2 {
                // Calculate the index 'j' with target bits swapped
                let j = i ^ (1 << target_qubit_1) ^ (1 << target_qubit_2);

                // Only perform the swap once per pair (i, j) by convention (e.g., when i < j)
                // And only if controls are met for *both* states involved in the swap
                if i < j {
                    let i_controls_met = check_controls(i, control_qubits);
                    let j_controls_met = check_controls(j, control_qubits);

                    if i_controls_met && j_controls_met {
                        // Use original state amplitudes for the swap
                        let amp_i = state.amplitude(i)?;
                        let amp_j = state.amplitude(j)?;
                        new_state[i] = amp_j;
                        new_state[j] = amp_i;
                    }
                    // else: Controls not met for the pair, leave amplitudes unchanged (already handled by clone)
                }
            }
            // else: Target bits are the same, no swap needed for this 'i' (already handled by clone)
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
    /// * `control_qubits` - The control qubits for the operator. This should be two qubits.
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
    ///
    /// * `Error::OverlappingControlAndTargetQubits` - If the control qubit and target qubit indices overlap.
    fn apply(
        &self,
        state: &State,
        target_qubits: &[usize],
        control_qubits: &[usize],
    ) -> Result<State, Error> {
        // Validation
        if target_qubits.len() != 1 {
            return Err(Error::InvalidNumberOfQubits(target_qubits.len()));
        }
        if control_qubits.len() != 2 {
            return Err(Error::InvalidNumberOfQubits(control_qubits.len()));
        }
        let target_qubit: usize = target_qubits[0];
        let control_qubit_1: usize = control_qubits[0];
        let control_qubit_2: usize = control_qubits[1];

        let num_qubits: usize = state.num_qubits();
        if target_qubit >= num_qubits {
            return Err(Error::InvalidQubitIndex(target_qubit, num_qubits));
        }
        if control_qubit_1 >= num_qubits {
            return Err(Error::InvalidQubitIndex(control_qubit_1, num_qubits));
        }
        if control_qubit_2 >= num_qubits {
            return Err(Error::InvalidQubitIndex(control_qubit_2, num_qubits));
        }
        if control_qubit_1 == control_qubit_2 {
            return Err(Error::InvalidNumberOfQubits(control_qubits.len()));
        }
        if control_qubit_1 == target_qubit {
            return Err(Error::OverlappingControlAndTargetQubits(
                control_qubit_1,
                target_qubit,
            ));
        }
        if control_qubit_2 == target_qubit {
            return Err(Error::OverlappingControlAndTargetQubits(
                control_qubit_2,
                target_qubit,
            ));
        }

        Pauli::X.apply(state, target_qubits, control_qubits)
    }

    fn base_qubits(&self) -> usize {
        3 // Toffoli acts on 3 qubits (2 control, 1 target)
    }
}

/// Defines an identity operator
///
/// A single-qubit operator that does not change the state of the qubit.
#[derive(Debug, Clone, Copy)]
pub struct Identity;

impl Operator for Identity {
    /// Applies the identity operator to the given state's target qubit.
    ///
    /// # Arguments:
    ///
    /// * `state` - The state to apply the operator to.
    ///
    /// * `target_qubits` - The target qubits to apply the operator to. This should be a single qubit.
    ///
    /// * `control_qubits` - The control qubits for the operator. If not empty, the operator will be applied conditionally based on the control qubits. Otherwise, it will be applied unconditionally.
    ///
    /// # Returns:
    ///
    /// * The new state after applying the identity operator.
    fn apply(
        &self,
        state: &State,
        target_qubits: &[usize],
        control_qubits: &[usize],
    ) -> Result<State, Error> {
        // Validation
        if target_qubits.len() != 1 {
            return Err(Error::InvalidNumberOfQubits(target_qubits.len()));
        }

        let target_qubit: usize = target_qubits[0];

        if target_qubit >= state.num_qubits() {
            return Err(Error::InvalidQubitIndex(target_qubit, state.num_qubits()));
        }

        for &control_qubit in control_qubits {
            if control_qubit >= state.num_qubits() {
                return Err(Error::InvalidQubitIndex(control_qubit, state.num_qubits()));
            }
            if control_qubit == target_qubit {
                return Err(Error::OverlappingControlAndTargetQubits(
                    control_qubit,
                    target_qubit,
                ));
            }
        }

        // Apply identity operator (no change)
        Ok(state.clone())
    }

    fn base_qubits(&self) -> usize {
        1 // Identity acts on 1 qubit only
    }
}

/// Defines a Phase S operator.
///
/// A single-qubit operator that applies a phase shift to the |1> state. Also known as the S gate or Phase gate.
#[derive(Debug, Clone, Copy)]
pub struct PhaseS;

impl Operator for PhaseS {
    /// Applies the Phase S operator to the given state's target qubit.
    ///
    /// # Arguments:
    ///
    /// * `state` - The state to apply the operator to.
    ///
    /// * `target_qubits` - The target qubits to apply the operator to. This should be a single qubit.
    ///
    /// * `control_qubits` - The control qubits for the operator. If not empty, the operator will be applied conditionally based on the control qubits. Otherwise, it will be applied unconditionally.
    ///
    /// # Returns:
    ///
    /// * The new state after applying the Phase S operator.
    fn apply(
        &self,
        state: &State,
        target_qubits: &[usize],
        control_qubits: &[usize],
    ) -> Result<State, Error> {
        // Validation
        if target_qubits.len() != 1 {
            return Err(Error::InvalidNumberOfQubits(target_qubits.len()));
        }
        if target_qubits[0] >= state.num_qubits() {
            return Err(Error::InvalidQubitIndex(
                target_qubits[0],
                state.num_qubits(),
            ));
        }
        let target_qubit: usize = target_qubits[0];
        if target_qubit >= state.num_qubits() {
            return Err(Error::InvalidQubitIndex(target_qubit, state.num_qubits()));
        }
        for &control_qubit in control_qubits {
            if control_qubit >= state.num_qubits() {
                return Err(Error::InvalidQubitIndex(control_qubit, state.num_qubits()));
            }
            if control_qubit == target_qubit {
                return Err(Error::OverlappingControlAndTargetQubits(
                    control_qubit,
                    target_qubit,
                ));
            }
        }

        // Apply potentially controlled Phase S operator
        let dim: usize = 1 << state.num_qubits();
        // Start with a copy, only modify elements where the operation applies
        let mut new_state: Vec<Complex<f64>> = state.state_vector.clone();
        let phase_factor = Complex::new(0.0, 1.0); // Phase shift of pi/2 (i)

        for i in 0..dim {
            let target_bit_is_one = (i >> target_qubit) & 1 == 1;

            // Apply phase shift only if target is |1> AND controls are met
            if target_bit_is_one && check_controls(i, control_qubits) {
                new_state[i] = state.amplitude(i)? * phase_factor;
            }
            // else: No change if target is |0> or controls are not met (already handled by clone)
        }

        Ok(State {
            state_vector: new_state,
            num_qubits: state.num_qubits(),
        })
    }

    fn base_qubits(&self) -> usize {
        1 // Phase S acts on 1 qubit only
    }
}

/// Defines a Phase T operator.
///
/// A single-qubit operator that applies a phase shift to the |1> state. Also known as the T gate or π/8 gate.s
pub struct PhaseT;

impl Operator for PhaseT {
    /// Applies the Phase T operator to the given state's target qubit.
    ///
    /// # Arguments:
    ///
    /// * `state` - The state to apply the operator to.
    ///
    /// * `target_qubits` - The target qubits to apply the operator to. This should be a single qubit.
    ///
    /// * `control_qubits` - The control qubits for the operator. If not empty, the operator will be applied conditionally based on the control qubits. Otherwise, it will be applied unconditionally.
    ///
    /// # Returns:
    ///
    /// * The new state after applying the Phase T operator.
    fn apply(
        &self,
        state: &State,
        target_qubits: &[usize],
        control_qubits: &[usize],
    ) -> Result<State, Error> {
        // Validation: Target Qubit
        if target_qubits.len() != 1 {
            return Err(Error::InvalidNumberOfQubits(target_qubits.len()));
        }
        let target_qubit = target_qubits[0];
        let num_qubits = state.num_qubits();

        if target_qubit >= num_qubits {
            return Err(Error::InvalidQubitIndex(target_qubit, num_qubits));
        }

        // Validation: Control Qubits
        for &control_qubit in control_qubits {
            // Use control_qubits
            if control_qubit >= num_qubits {
                return Err(Error::InvalidQubitIndex(control_qubit, num_qubits));
            }
            if control_qubit == target_qubit {
                return Err(Error::OverlappingControlAndTargetQubits(
                    control_qubit,
                    target_qubit,
                ));
            }
        }

        // Apply potentially controlled Phase T operator
        let dim: usize = 1 << num_qubits;
        // Start with a copy, only modify elements where the operation applies
        let mut new_state: Vec<Complex<f64>> = state.state_vector.clone();
        let invsqrt2: f64 = 1.0 / (2.0f64).sqrt();
        let phase_factor = Complex::new(invsqrt2, invsqrt2); // Phase shift of pi/4 (exp(i*pi/4))

        for i in 0..dim {
            let target_bit_is_one = (i >> target_qubit) & 1 == 1;

            // Apply phase shift only if target is |1> AND controls are met
            if target_bit_is_one && check_controls(i, control_qubits) {
                new_state[i] = state.amplitude(i)? * phase_factor;
            }
            // else: No change if target is |0> or controls are not met (already handled by clone)
        }

        Ok(State {
            state_vector: new_state,
            num_qubits: state.num_qubits(),
        })
    }

    fn base_qubits(&self) -> usize {
        1 // Phase T acts on 1 qubit only
    }
}

/// Defines a Phase Sdag operator.
///
/// A single-qubit operator that applies a phase shift to the |1> state. Also known as the S† gate or Phase† gate. Inverse of S gate.
#[derive(Debug, Clone, Copy)]
pub struct PhaseSdag;

impl Operator for PhaseSdag {
    /// Applies the Phase Sdag operator to the given state's target qubit.
    ///
    /// # Arguments:
    ///
    /// * `state` - The state to apply the operator to.
    ///
    /// * `target_qubits` - The target qubits to apply the operator to. This should be a single qubit.
    ///
    /// * `control_qubits` - The control qubits for the operator. If not empty, the operator will be applied conditionally based on the control qubits. Otherwise, it will be applied unconditionally.
    ///
    /// # Returns:
    ///
    /// * The new state after applying the Phase Sdag operator.
    fn apply(
        &self,
        state: &State,
        target_qubits: &[usize],
        control_qubits: &[usize],
    ) -> Result<State, Error> {
        // Validation: Target Qubit
        if target_qubits.len() != 1 {
            return Err(Error::InvalidNumberOfQubits(target_qubits.len()));
        }
        let target_qubit = target_qubits[0];
        let num_qubits = state.num_qubits();

        if target_qubit >= num_qubits {
            return Err(Error::InvalidQubitIndex(target_qubit, num_qubits));
        }

        // Validation: Control Qubits
        for &control_qubit in control_qubits {
            // Use control_qubits
            if control_qubit >= num_qubits {
                return Err(Error::InvalidQubitIndex(control_qubit, num_qubits));
            }
            if control_qubit == target_qubit {
                return Err(Error::OverlappingControlAndTargetQubits(
                    control_qubit,
                    target_qubit,
                ));
            }
        }

        // Apply potentially controlled Phase Sdag operator
        let dim: usize = 1 << num_qubits;
        // Start with a copy, only modify elements where the operation applies
        let mut new_state: Vec<Complex<f64>> = state.state_vector.clone();
        let phase_factor = Complex::new(0.0, -1.0); // Phase shift of -pi/2 (-i)

        for i in 0..dim {
            let target_bit_is_one = (i >> target_qubit) & 1 == 1;

            // Apply phase shift only if target is |1> AND controls are met
            if target_bit_is_one && check_controls(i, control_qubits) {
                new_state[i] = state.amplitude(i)? * phase_factor;
            }
            // else: No change if target is |0> or controls are not met (already handled by clone)
        }

        Ok(State {
            state_vector: new_state,
            num_qubits: state.num_qubits(),
        })
    }

    fn base_qubits(&self) -> usize {
        1 // Phase Sdag acts on 1 qubit only
    }
}

/// Defines a Phase Tdag operator.
///
/// A single-qubit operator that applies a phase shift to the |1> state. Also known as the T† gate or π/8† gate. Inverse of T gate.
pub struct PhaseTdag;

impl Operator for PhaseTdag {
    /// Applies the Phase Tdag operator to the given state's target qubit.
    ///
    /// # Arguments:
    ///
    /// * `state` - The state to apply the operator to.
    ///
    /// * `target_qubits` - The target qubits to apply the operator to. This should be a single qubit.
    ///
    /// * `control_qubits` - The control qubits for the operator. If not empty, the operator will be applied conditionally based on the control qubits. Otherwise, it will be applied unconditionally.
    ///
    /// # Returns:
    ///
    /// * The new state after applying the Phase Tdag operator.
    fn apply(
        &self,
        state: &State,
        target_qubits: &[usize],
        control_qubits: &[usize],
    ) -> Result<State, Error> {
        // Validation: Target Qubit
        if target_qubits.len() != 1 {
            return Err(Error::InvalidNumberOfQubits(target_qubits.len()));
        }
        let target_qubit = target_qubits[0];
        let num_qubits = state.num_qubits();

        if target_qubit >= num_qubits {
            return Err(Error::InvalidQubitIndex(target_qubit, num_qubits));
        }

        // Validation: Control Qubits
        for &control_qubit in control_qubits {
            // Use control_qubits
            if control_qubit >= num_qubits {
                return Err(Error::InvalidQubitIndex(control_qubit, num_qubits));
            }
            if control_qubit == target_qubit {
                return Err(Error::OverlappingControlAndTargetQubits(
                    control_qubit,
                    target_qubit,
                ));
            }
        }

        // Apply potentially controlled Phase Tdag operator
        let dim: usize = 1 << num_qubits;
        // Start with a copy, only modify elements where the operation applies
        let mut new_state: Vec<Complex<f64>> = state.state_vector.clone();
        let invsqrt2: f64 = 1.0 / (2.0f64).sqrt();
        // Phase shift of -pi/4 (exp(-i*pi/4) = cos(-pi/4) + i*sin(-pi/4) = cos(pi/4) - i*sin(pi/4))
        let phase_factor = Complex::new(invsqrt2, -invsqrt2);

        for i in 0..dim {
            let target_bit_is_one = (i >> target_qubit) & 1 == 1;

            // Apply phase shift only if target is |1> AND controls are met
            if target_bit_is_one && check_controls(i, control_qubits) {
                new_state[i] = state.amplitude(i)? * phase_factor;
            }
            // else: No change if target is |0> or controls are not met (already handled by clone)
        }

        Ok(State {
            state_vector: new_state,
            num_qubits: state.num_qubits(),
        })
    }

    fn base_qubits(&self) -> usize {
        1 // Phase Tdag acts on 1 qubit only
    }
}

/// Defines the phase shift operator
///
/// A single-qubit operator that applies a phase shift of the provided angle to the |1> state. Also known as the phase shift gate.
pub struct PhaseShift {
    angle: f64,
}

impl PhaseShift {
    /// Creates a new PhaseShift operator with the given angle.
    ///
    /// # Arguments:
    ///
    /// * `angle` - The angle of the phase shift in radians.
    pub fn new(angle: f64) -> Self {
        PhaseShift { angle }
    }
}

impl Operator for PhaseShift {
    /// Applies the phase shift operator to the given state's target qubit.
    ///
    /// # Arguments:
    ///
    /// * `state` - The state to apply the operator to.
    ///
    /// * `target_qubits` - The target qubits to apply the operator to. This should be a single qubit.
    ///
    /// * `control_qubits` - The control qubits for the operator. If not empty, the operator will be applied conditionally based on the control qubits. Otherwise, it will be applied unconditionally.
    ///
    /// # Returns:
    ///
    /// * The new state after applying the phase shift operator.
    ///
    /// # Errors:
    ///
    /// * `Error::InvalidNumberOfQubits` - If the target qubits is not 1.
    ///
    /// * `Error::InvalidQubitIndex` - If the target qubit index or control qubit index is invalid for the number of qubits in the state.
    ///
    /// * `Error::OverlappingControlAndTargetQubits` - If the control qubit and target qubit indices overlap.
    fn apply(
        &self,
        state: &State,
        target_qubits: &[usize],
        control_qubits: &[usize],
    ) -> Result<State, Error> {
        // Validation: Target Qubit
        if target_qubits.len() != 1 {
            return Err(Error::InvalidNumberOfQubits(target_qubits.len()));
        }
        let target_qubit = target_qubits[0];
        let num_qubits = state.num_qubits();

        if target_qubit >= num_qubits {
            return Err(Error::InvalidQubitIndex(target_qubit, num_qubits));
        }

        // Validation: Control Qubits
        for &control_qubit in control_qubits {
            // Use control_qubits
            if control_qubit >= num_qubits {
                return Err(Error::InvalidQubitIndex(control_qubit, num_qubits));
            }
            if control_qubit == target_qubit {
                return Err(Error::OverlappingControlAndTargetQubits(
                    control_qubit,
                    target_qubit,
                ));
            }
        }

        // Apply potentially controlled Phase Shift operator
        let dim: usize = 1 << num_qubits;
        // Start with a copy, only modify elements where the operation applies
        let mut new_state: Vec<Complex<f64>> = state.state_vector.clone();
        // Calculate phase factor: exp(i * angle) = cos(angle) + i * sin(angle)
        let phase_factor = Complex::new(self.angle.cos(), self.angle.sin());

        for i in 0..dim {
            let target_bit_is_one = (i >> target_qubit) & 1 == 1;

            // Apply phase shift only if target is |1> AND controls are met
            if target_bit_is_one && check_controls(i, control_qubits) {
                new_state[i] = state.amplitude(i)? * phase_factor;
            }
            // else: No change if target is |0> or controls are not met (already handled by clone)
        }

        Ok(State {
            state_vector: new_state,
            num_qubits: state.num_qubits(),
        })
    }

    fn base_qubits(&self) -> usize {
        1 // Phase shift acts on 1 qubit only
    }
}

/// Defines the rotate-X operator
///
/// A single-qubit operator that applies a rotation around the X axis of the Bloch sphere by the given angle. Also known as the RX gate.
pub struct RotateX {
    angle: f64,
}

impl RotateX {
    /// Creates a new RotateX operator with the given angle.
    ///
    /// # Arguments:
    ///
    /// * `angle` - The angle of rotation in radians.
    pub fn new(angle: f64) -> Self {
        RotateX { angle }
    }
}

impl Operator for RotateX {
    /// Applies the RotateX operator to the given state's target qubit.
    ///
    /// # Arguments:
    ///
    /// * `state` - The state to apply the operator to.
    ///
    /// * `target_qubits` - The target qubits to apply the operator to. This should be a single qubit.
    ///
    /// * `control_qubits` - The control qubits for the operator. If not empty, the operator will be applied conditionally based on the control qubits. Otherwise, it will be applied unconditionally.
    ///
    /// # Returns:
    ///
    /// * The new state after applying the RotateX operator.
    fn apply(
        &self,
        state: &State,
        target_qubits: &[usize],
        control_qubits: &[usize],
    ) -> Result<State, Error> {
        // Validation: Target Qubit
        if target_qubits.len() != 1 {
            return Err(Error::InvalidNumberOfQubits(target_qubits.len()));
        }
        let target_qubit = target_qubits[0];
        let num_qubits = state.num_qubits();

        if target_qubit >= num_qubits {
            return Err(Error::InvalidQubitIndex(target_qubit, num_qubits));
        }

        // Validation: Control Qubits
        for &control_qubit in control_qubits { // Use control_qubits
            if control_qubit >= num_qubits {
                return Err(Error::InvalidQubitIndex(control_qubit, num_qubits));
            }
            if control_qubit == target_qubit {
                return Err(Error::OverlappingControlAndTargetQubits(
                    control_qubit,
                    target_qubit,
                ));
            }
        }

        // Apply potentially controlled RotateX operator
        let dim: usize = 1 << num_qubits;
        let mut new_state: Vec<Complex<f64>> = state.state_vector.clone();
        let half_angle: f64 = self.angle / 2.0;
        let cos_half: f64 = half_angle.cos();
        let sin_half: f64 = half_angle.sin();
        let i_complex: Complex<f64> = Complex::new(0.0, 1.0);

        for i in 0..dim {
            // Process pairs (i, j) where target bit differs.
            if (i >> target_qubit) & 1 == 0 { // Process pairs starting from |..0..>
                let j: usize = i | (1 << target_qubit); // Index where target qubit is |1>

                // Check controls for both i and j.
                let i_controls_met: bool = check_controls(i, control_qubits);
                let j_controls_met: bool = check_controls(j, control_qubits);

                if i_controls_met && j_controls_met {
                    // Apply RotateX logic if controls met for both i and j
                    let amp_i: Complex<f64> = state.amplitude(i)?; // Use original state amplitudes
                    let amp_j: Complex<f64> = state.amplitude(j)?;

                    new_state[i] = cos_half * amp_i - i_complex * sin_half * amp_j;
                    new_state[j] = -i_complex * sin_half * amp_i + cos_half * amp_j;
                }
            }
            // Skip when target bit is 1, as the pair was handled when it was 0.
        }

        Ok(State {
            state_vector: new_state,
            num_qubits: state.num_qubits(),
        })
    }

    fn base_qubits(&self) -> usize {
        1 // RotateX acts on 1 qubit only
    }
}

/// Defines the rotate-Y operator
///
/// A single-qubit operator that applies a rotation around the Y axis of the Bloch sphere by the given angle. Also known as the RY gate.
pub struct RotateY {
    angle: f64,
}

impl RotateY {
    /// Creates a new RotateY operator with the given angle.
    ///
    /// # Arguments:
    ///
    /// * `angle` - The angle of rotation in radians.
    pub fn new(angle: f64) -> Self {
        RotateY { angle }
    }
}

impl Operator for RotateY {
    /// Applies the RotateY operator to the given state's target qubit.
    ///
    /// # Arguments:
    ///
    /// * `state` - The state to apply the operator to.
    ///
    /// * `target_qubits` - The target qubits to apply the operator to. This should be a single qubit.
    ///
    /// * `control_qubits` - The control qubits for the operator. If not empty, the operator will be applied conditionally based on the control qubits. Otherwise, it will be applied unconditionally.
    ///
    /// # Returns:
    ///
    /// * The new state after applying the RotateY operator.
    fn apply(
        &self,
        state: &State,
        target_qubits: &[usize],
        control_qubits: &[usize],
    ) -> Result<State, Error> {
        // Validation: Target Qubit
        if target_qubits.len() != 1 {
            return Err(Error::InvalidNumberOfQubits(target_qubits.len()));
        }
        let target_qubit = target_qubits[0];
        let num_qubits = state.num_qubits();

        if target_qubit >= num_qubits {
            return Err(Error::InvalidQubitIndex(target_qubit, num_qubits));
        }

        // Validation: Control Qubits
        for &control_qubit in control_qubits { // Use control_qubits
            if control_qubit >= num_qubits {
                return Err(Error::InvalidQubitIndex(control_qubit, num_qubits));
            }
            if control_qubit == target_qubit {
                return Err(Error::OverlappingControlAndTargetQubits(
                    control_qubit,
                    target_qubit,
                ));
            }
        }

        // Apply potentially controlled RotateY operator
        let dim: usize = 1 << num_qubits;
        let mut new_state: Vec<Complex<f64>> = state.state_vector.clone(); // Start with a copy
        let half_angle: f64 = self.angle / 2.0;
        let cos_half: f64 = half_angle.cos();
        let sin_half: f64 = half_angle.sin();

        for i in 0..dim {
            // Process pairs (i, j) where target bit differs.
            if (i >> target_qubit) & 1 == 0 { // Process pairs starting from |..0..>
                let j = i | (1 << target_qubit); // Index where target qubit is |1>

                // Check controls for both i and j.
                let i_controls_met: bool = check_controls(i, control_qubits);
                let j_controls_met: bool = check_controls(j, control_qubits);

                if i_controls_met && j_controls_met {
                    // Apply RotateY logic if controls met for both i and j
                    let amp_i: Complex<f64> = state.amplitude(i)?; // Use original state amplitudes
                    let amp_j: Complex<f64> = state.amplitude(j)?;

                    new_state[i] = cos_half * amp_i - sin_half * amp_j;
                    new_state[j] = sin_half * amp_i + cos_half * amp_j;
                }
            }
            // Skip when target bit is 1, as the pair was handled when it was 0.
        }

        Ok(State {
            state_vector: new_state,
            num_qubits: state.num_qubits(),
        })
    }

    fn base_qubits(&self) -> usize {
        1 // RotateY acts on 1 qubit only
    }
}

/// Defines the rotate-Z operator
///
/// A single-qubit operator that applies a rotation around the Z axis of the Bloch sphere by the given angle. Also known as the RZ gate.
pub struct RotateZ {
    angle: f64,
}

impl RotateZ {
    /// Creates a new RotateZ operator with the given angle.
    ///
    /// # Arguments:
    ///
    /// * `angle` - The angle of rotation in radians.
    pub fn new(angle: f64) -> Self {
        RotateZ { angle }
    }
}

impl Operator for RotateZ {
    /// Applies the RotateZ operator to the given state's target qubit.
    ///
    /// # Arguments:
    ///
    /// * `state` - The state to apply the operator to.
    ///
    /// * `target_qubits` - The target qubits to apply the operator to. This should be a single qubit.
    ///
    /// * `control_qubits` - The control qubits for the operator. If not empty, the operator will be applied conditionally based on the control qubits. Otherwise, it will be applied unconditionally.
    ///
    /// # Returns:
    ///
    /// * The new state after applying the RotateZ operator.
    fn apply(
        &self,
        state: &State,
        target_qubits: &[usize],
        control_qubits: &[usize],
    ) -> Result<State, Error> {
        // Validation: Target Qubit
        if target_qubits.len() != 1 {
            return Err(Error::InvalidNumberOfQubits(target_qubits.len()));
        }
        let target_qubit = target_qubits[0];
        let num_qubits = state.num_qubits();

        if target_qubit >= num_qubits {
            return Err(Error::InvalidQubitIndex(target_qubit, num_qubits));
        }

        // Validation: Control Qubits
        for &control_qubit in control_qubits { // Use control_qubits
            if control_qubit >= num_qubits {
                return Err(Error::InvalidQubitIndex(control_qubit, num_qubits));
            }
            if control_qubit == target_qubit {
                return Err(Error::OverlappingControlAndTargetQubits(
                    control_qubit,
                    target_qubit,
                ));
            }
        }

        // Apply potentially controlled RotateZ operator
        let dim: usize = 1 << num_qubits;
        let mut new_state: Vec<Complex<f64>> = state.state_vector.clone(); // Start with a copy
        let half_angle = self.angle / 2.0;
        // Phase factor for |0> state: exp(-i * angle / 2)
        let phase_0 = Complex::new(half_angle.cos(), -half_angle.sin());
        // Phase factor for |1> state: exp(i * angle / 2)
        let phase_1 = Complex::new(half_angle.cos(), half_angle.sin());

        for i in 0..dim {
            // Apply phase shift only if controls are met
            if check_controls(i, control_qubits) {
                let target_bit_is_one = (i >> target_qubit) & 1 == 1;

                if target_bit_is_one {
                    // Apply phase_1 to |1> component
                    new_state[i] = state.amplitude(i)? * phase_1;
                } else {
                    // Apply phase_0 to |0> component
                    new_state[i] = state.amplitude(i)? * phase_0;
                }
            }
            // else: Controls not met, leave amplitude unchanged (already handled by clone)
        }

        Ok(State {
            state_vector: new_state,
            num_qubits: state.num_qubits(),
        })
    }

    fn base_qubits(&self) -> usize {
        1 // RotateZ acts on 1 qubit only
    }
}
