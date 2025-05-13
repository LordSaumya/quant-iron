use crate::{components::state::State, errors::Error};
use num_complex::Complex;
use rayon::prelude::*;
#[cfg(feature = "gpu")]
use ocl::{ProQue, Buffer,flags, prm::Float2}; // Added ocl imports

const PARALLEL_THRESHOLD_NUM_QUBITS: usize = 10;
const OPENCL_THRESHOLD_NUM_QUBITS: usize = 15; // Threshold for using OpenCL

// Kernel sources - assumes .cl files are in src/components/kernels/
const HADAMARD_KERNEL_SRC: &str = include_str!("kernels/hadamard.cl");
const PAULI_X_KERNEL_SRC: &str = include_str!("kernels/pauli_x.cl");
const PAULI_Y_KERNEL_SRC: &str = include_str!("kernels/pauli_y.cl");
const PAULI_Z_KERNEL_SRC: &str = include_str!("kernels/pauli_z.cl");

#[cfg(feature = "gpu")]
fn execute_on_gpu(
    state: &State,
    target_qubit: usize,
    control_qubits: &[usize],
    kernel_name: &str,
    kernel_src: &str,
) -> Result<Vec<Complex<f64>>, Error> {
    let num_qubits = state.num_qubits();
    let pro_que = ProQue::builder()
        .src(kernel_src)
        .dims(state.state_vector.len())
        .build()
        .map_err(|e| Error::OpenCLError(format!("Failed to build ProQue: {}", e)))?;

    let state_vector_f32: Vec<Float2> = state.state_vector.iter()
        .map(|c| Float2::new(c.re as f32, c.im as f32))
        .collect();

    let state_buffer = Buffer::builder()
        .queue(pro_que.queue().clone())
        .flags(flags::MEM_READ_WRITE | flags::MEM_COPY_HOST_PTR)
        .len(state_vector_f32.len())
        .copy_host_slice(&state_vector_f32)
        .build()
        .map_err(|e| Error::OpenCLError(format!("Failed to create state buffer: {}", e)))?;

    let control_qubits_i32: Vec<i32> = control_qubits.iter().map(|&q| q as i32).collect();
    let control_buffer: Buffer<i32> = if !control_qubits_i32.is_empty() {
        Buffer::builder()
            .queue(pro_que.queue().clone())
            .flags(flags::MEM_READ_ONLY | flags::MEM_COPY_HOST_PTR)
            .len(control_qubits_i32.len())
            .copy_host_slice(&control_qubits_i32)
            .build()
            .map_err(|e| Error::OpenCLError(format!("Failed to create control buffer: {}", e)))?
    } else {
        let dummy_control_data = [0i32];
        Buffer::builder()
            .queue(pro_que.queue().clone())
            .flags(flags::MEM_READ_ONLY | flags::MEM_COPY_HOST_PTR)
            .len(dummy_control_data.len())
            .copy_host_slice(&dummy_control_data)
            .build()
            .map_err(|e| Error::OpenCLError(format!("Failed to create dummy control buffer: {}", e)))?
    };

    let kernel = pro_que.kernel_builder(kernel_name)
        .global_work_size((1 << (num_qubits - 1)) as usize)
        .arg(&state_buffer)
        .arg(num_qubits as i32)
        .arg(target_qubit as i32)
        .arg(&control_buffer)
        .arg(control_qubits_i32.len() as i32)
        .build()
        .map_err(|e| Error::OpenCLError(format!("Failed to build kernel: {}", e)))?;

    unsafe {
        kernel.enq().map_err(|e| Error::OpenCLError(format!("Failed to enqueue kernel: {}", e)))?;
    }

    let mut state_vector_ocl_result = vec![Float2::new(0.0, 0.0); state_vector_f32.len()];
    state_buffer.read(&mut state_vector_ocl_result).enq()
        .map_err(|e| Error::OpenCLError(format!("Failed to read state buffer: {}", e)))?;

    Ok(state_vector_ocl_result.iter()
        .map(|f2| Complex::new(f2[0] as f64, f2[1] as f64))
        .collect())
}

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
    control_qubits
        .iter()
        .all(|&qubit| (index >> qubit) & 1 == 1)
}

/// Helper function to validate target and control qubits
///
/// # Arguments:
///
/// * `state` - The quantum state that contains information about the number of qubits.
/// * `target_qubits` - The target qubits to validate.
/// * `control_qubits` - The control qubits to validate.
/// * `expected_targets` - The expected number of target qubits.
///
/// # Returns:
///
/// * `Ok(())` if all validations pass.
/// * `Err(Error)` if any validation fails.
fn validate_qubits(
    state: &State,
    target_qubits: &[usize],
    control_qubits: &[usize],
    expected_targets: usize,
) -> Result<(), Error> {
    // Check if we have the expected number of target qubits
    if target_qubits.len() != expected_targets {
        return Err(Error::InvalidNumberOfQubits(target_qubits.len()));
    }

    let num_qubits = state.num_qubits();

    // Check if all target qubits are valid indices
    for &target_qubit in target_qubits {
        if target_qubit >= num_qubits {
            return Err(Error::InvalidQubitIndex(target_qubit, num_qubits));
        }
    }

    // Check if all control qubits are valid indices and don't overlap with target qubits
    for &control_qubit in control_qubits {
        if control_qubit >= num_qubits {
            return Err(Error::InvalidQubitIndex(control_qubit, num_qubits));
        }

        for &target_qubit in target_qubits {
            if control_qubit == target_qubit {
                return Err(Error::OverlappingControlAndTargetQubits(
                    control_qubit,
                    target_qubit,
                ));
            }
        }
    }

    // Special check for multiple target qubits to ensure no duplicates
    if expected_targets > 1 {
        for i in 0..target_qubits.len() {
            for j in i + 1..target_qubits.len() {
                if target_qubits[i] == target_qubits[j] {
                    return Err(Error::InvalidQubitIndex(target_qubits[i], num_qubits));
                }
            }
        }
    }

    Ok(())
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
        validate_qubits(state, target_qubits, control_qubits, 1)?;

        let target_qubit: usize = target_qubits[0];
        let num_qubits: usize = state.num_qubits();

        // Apply potentially controlled Hadamard operator
        let sqrt_2_inv: f64 = 1.0 / (2.0f64).sqrt();
        let dim: usize = 1 << num_qubits;
        let mut new_state_vec: Vec<Complex<f64>>;
        let gpu_enabled: bool = cfg!(feature = "gpu");

        if num_qubits >= OPENCL_THRESHOLD_NUM_QUBITS && gpu_enabled {
            #[cfg(feature = "gpu")]
            {
                new_state_vec = execute_on_gpu(
                    state,
                    target_qubit,
                    control_qubits,
                    "hadamard_kernel",
                    HADAMARD_KERNEL_SRC,
                )?;
            }
        } else if num_qubits >= PARALLEL_THRESHOLD_NUM_QUBITS {
            // Rayon CPU Parallel implementation
            new_state_vec = state.state_vector.clone(); // Initialise for CPU path
            if control_qubits.is_empty() {
                // Parallel uncontrolled Hadamard
                let updates: Vec<(usize, Complex<f64>)> = (0..(1 << (num_qubits - 1)))
                    .into_par_iter()
                    .flat_map(|k| {
                        let i0 = (k >> target_qubit << (target_qubit + 1))
                            | (k & ((1 << target_qubit) - 1));
                        let i1 = i0 | (1 << target_qubit);
                        let amp0 = state.state_vector[i0];
                        let amp1 = state.state_vector[i1];
                        vec![
                            (i0, sqrt_2_inv * (amp0 + amp1)),
                            (i1, sqrt_2_inv * (amp0 - amp1)),
                        ]
                    })
                    .collect();
                for (idx, val) in updates {
                    new_state_vec[idx] = val;
                }
            } else {
                // Rayon CPU Parallel controlled Hadamard
                let updates: Vec<(usize, Complex<f64>)> = (0..dim)
                    .into_par_iter()
                    .filter_map(|i| {
                        if (i >> target_qubit) & 1 == 0 { // Process pairs (i, j) where i has 0 at target_qubit
                            let j = i | (1 << target_qubit); // j has 1 at target_qubit
                            if check_controls(i, control_qubits) { // Check controls based on i
                                let amp_i = state.state_vector[i];
                                let amp_j = state.state_vector[j];
                                Some(vec![
                                    (i, sqrt_2_inv * (amp_i + amp_j)),
                                    (j, sqrt_2_inv * (amp_i - amp_j)),
                                ])
                            } else {
                                None // Controls not met for this pair
                            }
                        } else {
                            None // Already processed as part of a pair starting with 0 at target_qubit
                        }
                    })
                    .flatten()
                    .collect();
                for (idx, val) in updates {
                    new_state_vec[idx] = val;
                }
            }
        } else {
            // Sequential CPU implementation
            new_state_vec = state.state_vector.clone(); // initialise for CPU path
            if control_qubits.is_empty() {
                // Sequential uncontrolled Hadamard
                for k in 0..(1 << (num_qubits - 1)) {
                    let i0 =
                        (k >> target_qubit << (target_qubit + 1)) | (k & ((1 << target_qubit) - 1));
                    let i1 = i0 | (1 << target_qubit);
                    let amp0 = state.state_vector[i0];
                    let amp1 = state.state_vector[i1];
                    new_state_vec[i0] = sqrt_2_inv * (amp0 + amp1);
                    new_state_vec[i1] = sqrt_2_inv * (amp0 - amp1);
                }
            } else {
                // Sequential controlled Hadamard
                for i in 0..dim {
                    if (i >> target_qubit) & 1 == 0 {
                        let j = i | (1 << target_qubit);
                        if check_controls(i, control_qubits) {
                            let amp_i = state.state_vector[i];
                            let amp_j = state.state_vector[j];
                            new_state_vec[i] = sqrt_2_inv * (amp_i + amp_j);
                            new_state_vec[j] = sqrt_2_inv * (amp_i - amp_j);
                        }
                    }
                }
            }
        }

        Ok(State {
            state_vector: new_state_vec,
            num_qubits,
        })
    }

    fn base_qubits(&self) -> usize {
        1 // Hadamard acts on 1 qubit only
    }
}

/// Defines the Pauli operators: X, Y, Z.
#[derive(Debug, Clone, Copy, PartialEq)]
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
        validate_qubits(state, target_qubits, control_qubits, 1)?;

        let target_qubit: usize = target_qubits[0];
        let num_qubits: usize = state.num_qubits();

        // Apply potentially controlled Pauli operator
        let dim: usize = 1 << num_qubits;
        let mut new_state_vec: Vec<Complex<f64>> = state.state_vector.clone();
        let i_complex: Complex<f64> = Complex::new(0.0, 1.0);
        let gpu_enabled: bool = cfg!(feature = "gpu");

        if num_qubits >= OPENCL_THRESHOLD_NUM_QUBITS && gpu_enabled {
            #[cfg(feature = "gpu")]
            {
                let kernel = match self {
                    Pauli::X => ("pauli_x_kernel", PAULI_X_KERNEL_SRC),
                    Pauli::Y => ("pauli_y_kernel", PAULI_Y_KERNEL_SRC),
                    Pauli::Z => ("pauli_z_kernel", PAULI_Z_KERNEL_SRC),
                };
                new_state_vec = execute_on_gpu(
                    state,
                    target_qubit,
                    control_qubits,
                    kernel.0,
                    kernel.1,
                )?;
            }
        } else if num_qubits >= PARALLEL_THRESHOLD_NUM_QUBITS {
            // Parallel implementation
            match self {
                Pauli::X => {
                    let updates: Vec<(usize, Complex<f64>)> = (0..dim)
                        .into_par_iter()
                        .filter_map(|i| {
                            if check_controls(i, control_qubits) && ((i >> target_qubit) & 1 == 0) {
                                let j = i | (1 << target_qubit);
                                let amp_i = state.state_vector[i];
                                let amp_j = state.state_vector[j];
                                Some(vec![(i, amp_j), (j, amp_i)])
                            } else {
                                None
                            }
                        })
                        .flatten()
                        .collect();
                    for (idx, val) in updates {
                        new_state_vec[idx] = val;
                    }
                }
                Pauli::Y => {
                    let updates: Vec<(usize, Complex<f64>)> = (0..dim)
                        .into_par_iter()
                        .filter_map(|i| {
                            if check_controls(i, control_qubits) && ((i >> target_qubit) & 1 == 0) {
                                let j = i | (1 << target_qubit);
                                let amp_i = state.state_vector[i];
                                let amp_j = state.state_vector[j];
                                Some(vec![(i, -i_complex * amp_j), (j, i_complex * amp_i)])
                            } else {
                                None
                            }
                        })
                        .flatten()
                        .collect();
                    for (idx, val) in updates {
                        new_state_vec[idx] = val;
                    }
                }
                Pauli::Z => {
                    new_state_vec
                        .par_iter_mut()
                        .enumerate()
                        .for_each(|(i, current_amp_ref)| {
                            if check_controls(i, control_qubits) && ((i >> target_qubit) & 1 == 1) {
                                *current_amp_ref = -state.state_vector[i];
                            }
                        });
                }
            }
        } else {
            // Sequential implementation
            for i in 0..dim {
                if check_controls(i, control_qubits) {
                    match self {
                        Pauli::X => {
                            if (i >> target_qubit) & 1 == 0 {
                                let j = i | (1 << target_qubit);
                                let amp_i = state.state_vector[i];
                                let amp_j = state.state_vector[j];
                                new_state_vec[i] = amp_j;
                                new_state_vec[j] = amp_i;
                            }
                        }
                        Pauli::Y => {
                            if (i >> target_qubit) & 1 == 0 {
                                let j = i | (1 << target_qubit);
                                let amp_i = state.state_vector[i];
                                let amp_j = state.state_vector[j];
                                new_state_vec[i] = -i_complex * amp_j;
                                new_state_vec[j] = i_complex * amp_i;
                            }
                        }
                        Pauli::Z => {
                            if (i >> target_qubit) & 1 == 1 {
                                new_state_vec[i] = -state.state_vector[i];
                            }
                        }
                    }
                }
            }
        }

        Ok(State {
            state_vector: new_state_vec,
            num_qubits: state.num_qubits(),
        })
    }

    fn base_qubits(&self) -> usize {
        1 // Pauli operators act on 1 qubit only
    }
}

impl std::fmt::Display for Pauli {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Pauli::X => write!(f, "X"),
            Pauli::Y => write!(f, "Y"),
            Pauli::Z => write!(f, "Z"),
        }
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
        validate_qubits(state, target_qubits, control_qubits, 1)?;

        // Additional validation for CNOT: exactly one control qubit
        if control_qubits.len() != 1 {
            return Err(Error::InvalidNumberOfQubits(control_qubits.len()));
        }

        let control_qubit: usize = control_qubits[0];

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
        // Validation
        validate_qubits(state, target_qubits, control_qubits, 2)?;

        let target_qubit_1: usize = target_qubits[0];
        let target_qubit_2: usize = target_qubits[1];
        let num_qubits: usize = state.num_qubits();

        // Apply potentially controlled SWAP operator
        let dim: usize = 1 << num_qubits;
        let mut new_state_vec = state.state_vector.clone(); // Start with a copy

        if num_qubits >= PARALLEL_THRESHOLD_NUM_QUBITS {
            // Parallel implementation
            let updates: Vec<(usize, Complex<f64>)> = (0..dim)
                .into_par_iter()
                .filter_map(|i| {
                    let target_bit_1 = (i >> target_qubit_1) & 1;
                    let target_bit_2 = (i >> target_qubit_2) & 1;

                    if target_bit_1 != target_bit_2 {
                        let j = i ^ (1 << target_qubit_1) ^ (1 << target_qubit_2);
                        if i < j && check_controls(i, control_qubits) {
                            let amp_i = state.state_vector[i];
                            let amp_j = state.state_vector[j];
                            Some(vec![(i, amp_j), (j, amp_i)])
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                })
                .flatten()
                .collect();
            for (idx, val) in updates {
                new_state_vec[idx] = val;
            }
        } else {
            // Sequential implementation
            for i in 0..dim {
                let target_bit_1 = (i >> target_qubit_1) & 1;
                let target_bit_2 = (i >> target_qubit_2) & 1;

                if target_bit_1 != target_bit_2 {
                    let j = i ^ (1 << target_qubit_1) ^ (1 << target_qubit_2);
                    if i < j {
                        if check_controls(i, control_qubits) {
                            let amp_i = state.state_vector[i];
                            let amp_j = state.state_vector[j];
                            new_state_vec[i] = amp_j;
                            new_state_vec[j] = amp_i;
                        }
                    }
                }
            }
        }

        Ok(State {
            state_vector: new_state_vec,
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
        validate_qubits(state, target_qubits, control_qubits, 1)?;

        // Additional validation for Toffoli: exactly two control qubits
        if control_qubits.len() != 2 {
            return Err(Error::InvalidNumberOfQubits(control_qubits.len()));
        }

        // Additional validation for Toffoli: control qubits must be different
        if control_qubits[0] == control_qubits[1] {
            return Err(Error::InvalidNumberOfQubits(control_qubits.len()));
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
        validate_qubits(state, target_qubits, control_qubits, 1)?;

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
        validate_qubits(state, target_qubits, control_qubits, 1)?;

        let target_qubit: usize = target_qubits[0];
        let num_qubits: usize = state.num_qubits();

        // Apply potentially controlled Phase S operator
        let dim: usize = 1 << num_qubits;
        let mut new_state_vec: Vec<Complex<f64>> = state.state_vector.clone();
        let phase_factor = Complex::new(0.0, 1.0); // Phase shift of pi/2 (i)

        if num_qubits >= PARALLEL_THRESHOLD_NUM_QUBITS {
            new_state_vec
                .par_iter_mut()
                .enumerate()
                .for_each(|(i, current_amp_ref)| {
                    if ((i >> target_qubit) & 1 == 1) && check_controls(i, control_qubits) {
                        *current_amp_ref = state.state_vector[i] * phase_factor;
                    }
                });
        } else {
            for i in 0..dim {
                let target_bit_is_one = (i >> target_qubit) & 1 == 1;
                if target_bit_is_one && check_controls(i, control_qubits) {
                    new_state_vec[i] = state.state_vector[i] * phase_factor;
                }
            }
        }

        Ok(State {
            state_vector: new_state_vec,
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
        // Validation
        validate_qubits(state, target_qubits, control_qubits, 1)?;

        let target_qubit = target_qubits[0];
        let num_qubits = state.num_qubits();

        // Apply potentially controlled Phase T operator
        let dim: usize = 1 << num_qubits;
        let mut new_state_vec: Vec<Complex<f64>> = state.state_vector.clone();
        let invsqrt2: f64 = 1.0 / (2.0f64).sqrt();
        let phase_factor = Complex::new(invsqrt2, invsqrt2); // Phase shift of pi/4 (exp(i*pi/4))

        if num_qubits >= PARALLEL_THRESHOLD_NUM_QUBITS {
            new_state_vec
                .par_iter_mut()
                .enumerate()
                .for_each(|(i, current_amp_ref)| {
                    if ((i >> target_qubit) & 1 == 1) && check_controls(i, control_qubits) {
                        *current_amp_ref = state.state_vector[i] * phase_factor;
                    }
                });
        } else {
            for i in 0..dim {
                let target_bit_is_one = (i >> target_qubit) & 1 == 1;
                if target_bit_is_one && check_controls(i, control_qubits) {
                    new_state_vec[i] = state.state_vector[i] * phase_factor;
                }
            }
        }

        Ok(State {
            state_vector: new_state_vec,
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
        // Validation
        validate_qubits(state, target_qubits, control_qubits, 1)?;

        let target_qubit = target_qubits[0];
        let num_qubits = state.num_qubits();

        // Apply potentially controlled Phase Sdag operator
        let dim: usize = 1 << num_qubits;
        let mut new_state_vec: Vec<Complex<f64>> = state.state_vector.clone();
        let phase_factor = Complex::new(0.0, -1.0); // Phase shift of -pi/2 (-i)

        if num_qubits >= PARALLEL_THRESHOLD_NUM_QUBITS {
            new_state_vec
                .par_iter_mut()
                .enumerate()
                .for_each(|(i, current_amp_ref)| {
                    if ((i >> target_qubit) & 1 == 1) && check_controls(i, control_qubits) {
                        *current_amp_ref = state.state_vector[i] * phase_factor;
                    }
                });
        } else {
            for i in 0..dim {
                let target_bit_is_one = (i >> target_qubit) & 1 == 1;
                if target_bit_is_one && check_controls(i, control_qubits) {
                    new_state_vec[i] = state.state_vector[i] * phase_factor;
                }
            }
        }

        Ok(State {
            state_vector: new_state_vec,
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
        // Validation
        validate_qubits(state, target_qubits, control_qubits, 1)?;

        let target_qubit = target_qubits[0];
        let num_qubits = state.num_qubits();

        // Apply potentially controlled Phase Tdag operator
        let dim: usize = 1 << num_qubits;
        let mut new_state_vec: Vec<Complex<f64>> = state.state_vector.clone();
        let invsqrt2: f64 = 1.0 / (2.0f64).sqrt();
        // Phase shift of -pi/4 (exp(-i*pi/4) = cos(-pi/4) + i*sin(-pi/4) = cos(pi/4) - i*sin(pi/4))
        let phase_factor = Complex::new(invsqrt2, -invsqrt2);

        if num_qubits >= PARALLEL_THRESHOLD_NUM_QUBITS {
            new_state_vec
                .par_iter_mut()
                .enumerate()
                .for_each(|(i, current_amp_ref)| {
                    if ((i >> target_qubit) & 1 == 1) && check_controls(i, control_qubits) {
                        *current_amp_ref = state.state_vector[i] * phase_factor;
                    }
                });
        } else {
            for i in 0..dim {
                let target_bit_is_one = (i >> target_qubit) & 1 == 1;
                if target_bit_is_one && check_controls(i, control_qubits) {
                    new_state_vec[i] = state.state_vector[i] * phase_factor;
                }
            }
        }

        Ok(State {
            state_vector: new_state_vec,
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
#[derive(Debug, Clone, Copy)]
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
        // Validation
        validate_qubits(state, target_qubits, control_qubits, 1)?;

        let target_qubit = target_qubits[0];
        let num_qubits = state.num_qubits();

        // Apply potentially controlled Phase Shift operator
        let dim: usize = 1 << num_qubits;
        let mut new_state_vec: Vec<Complex<f64>> = state.state_vector.clone();
        // Calculate phase factor: exp(i * angle) = cos(angle) + i * sin(angle)
        let phase_factor = Complex::new(self.angle.cos(), self.angle.sin());

        if num_qubits >= PARALLEL_THRESHOLD_NUM_QUBITS {
            new_state_vec
                .par_iter_mut()
                .enumerate()
                .for_each(|(i, current_amp_ref)| {
                    if ((i >> target_qubit) & 1 == 1) && check_controls(i, control_qubits) {
                        *current_amp_ref = state.state_vector[i] * phase_factor;
                    }
                });
        } else {
            for i in 0..dim {
                let target_bit_is_one = (i >> target_qubit) & 1 == 1;
                if target_bit_is_one && check_controls(i, control_qubits) {
                    new_state_vec[i] = state.state_vector[i] * phase_factor;
                }
            }
        }

        Ok(State {
            state_vector: new_state_vec,
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
#[derive(Debug, Clone, Copy)]
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
        // Validation
        validate_qubits(state, target_qubits, control_qubits, 1)?;

        let target_qubit = target_qubits[0];
        let num_qubits = state.num_qubits();

        // Apply potentially controlled RotateX operator
        let dim: usize = 1 << num_qubits;
        let mut new_state_vec: Vec<Complex<f64>> = state.state_vector.clone();
        let half_angle: f64 = self.angle / 2.0;
        let cos_half: f64 = half_angle.cos();
        let sin_half: f64 = half_angle.sin();
        let i_complex: Complex<f64> = Complex::new(0.0, 1.0);

        if num_qubits >= PARALLEL_THRESHOLD_NUM_QUBITS {
            // Parallel implementation
            let updates: Vec<(usize, Complex<f64>)> = (0..dim)
                .into_par_iter()
                .filter_map(|i| {
                    if ((i >> target_qubit) & 1 == 0) && check_controls(i, control_qubits) {
                        let j = i | (1 << target_qubit);
                        let amp_i = state.state_vector[i];
                        let amp_j = state.state_vector[j];
                        Some(vec![
                            (i, cos_half * amp_i - i_complex * sin_half * amp_j),
                            (j, -i_complex * sin_half * amp_i + cos_half * amp_j),
                        ])
                    } else {
                        None
                    }
                })
                .flatten()
                .collect();
            for (idx, val) in updates {
                new_state_vec[idx] = val;
            }
        } else {
            // Sequential implementation
            for i in 0..dim {
                if (i >> target_qubit) & 1 == 0 {
                    let j = i | (1 << target_qubit);
                    if check_controls(i, control_qubits) {
                        let amp_i = state.state_vector[i];
                        let amp_j = state.state_vector[j];
                        new_state_vec[i] = cos_half * amp_i - i_complex * sin_half * amp_j;
                        new_state_vec[j] = -i_complex * sin_half * amp_i + cos_half * amp_j;
                    }
                }
            }
        }

        Ok(State {
            state_vector: new_state_vec,
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
#[derive(Debug, Clone, Copy)]
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
        // Validation
        validate_qubits(state, target_qubits, control_qubits, 1)?;

        let target_qubit = target_qubits[0];
        let num_qubits = state.num_qubits();

        // Apply potentially controlled RotateY operator
        let dim: usize = 1 << num_qubits;
        let mut new_state_vec: Vec<Complex<f64>> = state.state_vector.clone(); // Start with a copy
        let half_angle: f64 = self.angle / 2.0;
        let cos_half: f64 = half_angle.cos();
        let sin_half: f64 = half_angle.sin();

        if num_qubits >= PARALLEL_THRESHOLD_NUM_QUBITS {
            // Parallel implementation
            let updates: Vec<(usize, Complex<f64>)> = (0..dim)
                .into_par_iter()
                .filter_map(|i| {
                    if ((i >> target_qubit) & 1 == 0) && check_controls(i, control_qubits) {
                        let j = i | (1 << target_qubit);
                        let amp_i = state.state_vector[i];
                        let amp_j = state.state_vector[j];
                        Some(vec![
                            (i, cos_half * amp_i - sin_half * amp_j),
                            (j, sin_half * amp_i + cos_half * amp_j),
                        ])
                    } else {
                        None
                    }
                })
                .flatten()
                .collect();
            for (idx, val) in updates {
                new_state_vec[idx] = val;
            }
        } else {
            // Sequential implementation
            for i in 0..dim {
                if (i >> target_qubit) & 1 == 0 {
                    let j = i | (1 << target_qubit);
                    if check_controls(i, control_qubits) {
                        let amp_i = state.state_vector[i];
                        let amp_j = state.state_vector[j];
                        new_state_vec[i] = cos_half * amp_i - sin_half * amp_j;
                        new_state_vec[j] = sin_half * amp_i + cos_half * amp_j;
                    }
                }
            }
        }

        Ok(State {
            state_vector: new_state_vec,
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
#[derive(Debug, Clone, Copy)]
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
        // Validation
        validate_qubits(state, target_qubits, control_qubits, 1)?;

        let target_qubit = target_qubits[0];
        let num_qubits = state.num_qubits();

        // Apply potentially controlled RotateZ operator
        let dim: usize = 1 << num_qubits;
        let mut new_state_vec: Vec<Complex<f64>> = state.state_vector.clone(); // Start with a copy
        let half_angle = self.angle / 2.0;
        // Phase factor for |0> state: exp(-i * angle / 2)
        let phase_0 = Complex::new(half_angle.cos(), -half_angle.sin());
        // Phase factor for |1> state: exp(i * angle / 2)
        let phase_1 = Complex::new(half_angle.cos(), half_angle.sin());

        if num_qubits >= PARALLEL_THRESHOLD_NUM_QUBITS {
            new_state_vec
                .par_iter_mut()
                .enumerate()
                .for_each(|(i, current_amp_ref)| {
                    if check_controls(i, control_qubits) {
                        let target_bit_is_one = (i >> target_qubit) & 1 == 1;
                        if target_bit_is_one {
                            *current_amp_ref = state.state_vector[i] * phase_1;
                        } else {
                            *current_amp_ref = state.state_vector[i] * phase_0;
                        }
                    }
                });
        } else {
            for i in 0..dim {
                if check_controls(i, control_qubits) {
                    let target_bit_is_one = (i >> target_qubit) & 1 == 1;
                    if target_bit_is_one {
                        new_state_vec[i] = state.state_vector[i] * phase_1;
                    } else {
                        new_state_vec[i] = state.state_vector[i] * phase_0;
                    }
                }
            }
        }

        Ok(State {
            state_vector: new_state_vec,
            num_qubits: state.num_qubits(),
        })
    }

    fn base_qubits(&self) -> usize {
        1 // RotateZ acts on 1 qubit only
    }
}

/// An arbitrary 2×2 unitary operator.
///
/// This operator can be applied to a single qubit in a quantum state. It is represented by a 2×2 unitary matrix.
#[derive(Debug, Clone, Copy)]
pub struct Unitary2 {
    /// The 2×2 unitary matrix representing the operator.
    matrix: [[Complex<f64>; 2]; 2],
}

impl Unitary2 {
    /// Creates a new Unitary2 operator with the given 2×2 unitary matrix.
    ///
    /// # Arguments:
    ///
    /// * `matrix` - A 2×2 unitary matrix represented as a 2D array of complex numbers.
    ///
    /// # Returns:
    ///
    /// * `Result<Self, Error>` - A result containing the new Unitary2 operator or an error if the matrix is not unitary.
    ///
    /// # Errors:
    ///
    /// * `Error::NonUnitaryMatrix` - If the provided matrix is not unitary.
    pub fn new(matrix: [[Complex<f64>; 2]; 2]) -> Result<Self, Error> {
        // Faster 2×2 unitary check: U U_dagger = I (rows are orthonormal)
        let tol: f64 = f64::EPSILON * 2.0; // Tolerance for floating point comparisons
        let a: Complex<f64> = matrix[0][0]; // U_00
        let b: Complex<f64> = matrix[0][1]; // U_01
        let c: Complex<f64> = matrix[1][0]; // U_10
        let d: Complex<f64> = matrix[1][1]; // U_11

        // Check if each row has norm 1
        // Row 0: |a|^2 + |b|^2 == 1
        if ((a.norm_sqr() + b.norm_sqr()) - 1.0).abs() > tol {
            return Err(Error::NonUnitaryMatrix);
        }
        // Row 1: |c|^2 + |d|^2 == 1
        if ((c.norm_sqr() + d.norm_sqr()) - 1.0).abs() > tol {
            return Err(Error::NonUnitaryMatrix);
        }

        // Check if rows are orthogonal
        // Row 0 dot Row 1_conj: a*c_conj + b*d_conj == 0
        if (a * c.conj() + b * d.conj()).norm_sqr() > tol * tol {
            // Compare norm_sqr with tol^2
            return Err(Error::NonUnitaryMatrix);
        }

        Ok(Unitary2 { matrix })
    }
}

impl Operator for Unitary2 {
    /// Applies the Unitary2 operator to the given state's target qubit.
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
    /// * The new state after applying the Unitary2 operator.
    fn apply(
        &self,
        state: &State,
        target_qubits: &[usize],
        control_qubits: &[usize],
    ) -> Result<State, Error> {
        // Validation
        validate_qubits(state, target_qubits, control_qubits, 1)?;

        let t: usize = target_qubits[0];
        let nq: usize = state.num_qubits();

        // Apply 2×2 block on each basis‐pair
        let dim = 1 << nq;
        let mut new_state_vec = state.state_vector.clone();

        if nq >= PARALLEL_THRESHOLD_NUM_QUBITS {
            // Parallel implementation
            let updates: Vec<(usize, Complex<f64>)> = (0..dim)
                .into_par_iter()
                .filter_map(|i| {
                    if ((i >> t) & 1 == 0) && check_controls(i, control_qubits) {
                        let j = i | (1 << t);
                        let ai = state.state_vector[i];
                        let aj = state.state_vector[j];
                        Some(vec![
                            (i, self.matrix[0][0] * ai + self.matrix[0][1] * aj),
                            (j, self.matrix[1][0] * ai + self.matrix[1][1] * aj),
                        ])
                    } else {
                        None
                    }
                })
                .flatten()
                .collect();
            for (idx, val) in updates {
                new_state_vec[idx] = val;
            }
        } else {
            // Sequential implementation
            for i in 0..dim {
                if (i >> t) & 1 == 0 {
                    let j = i | (1 << t);
                    if check_controls(i, control_qubits) {
                        let ai = state.state_vector[i];
                        let aj = state.state_vector[j];
                        new_state_vec[i] = self.matrix[0][0] * ai + self.matrix[0][1] * aj;
                        new_state_vec[j] = self.matrix[1][0] * ai + self.matrix[1][1] * aj;
                    }
                }
            }
        }

        Ok(State {
            state_vector: new_state_vec,
            num_qubits: nq,
        })
    }

    fn base_qubits(&self) -> usize {
        1
    }
}
