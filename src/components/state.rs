use crate::components::{
    measurement::{MeasurementBasis, MeasurementResult},
    operator::{
        Hadamard, Identity, Matchgate, Operator, Pauli, PhaseS, PhaseSdag, PhaseShift, PhaseT, PhaseTdag, RotateX, RotateY, RotateZ, Toffoli, Unitary2, CNOT, SWAP
    },
};
use crate::errors::Error;
use num_complex::Complex;
use rand::Rng;
use rayon::prelude::*;
use std::ops::{Add, Mul, Sub};

// Helper function to calculate the adjoint (conjugate transpose) of a 2x2 matrix
fn calculate_adjoint(matrix: &[[Complex<f64>; 2]; 2]) -> [[Complex<f64>; 2]; 2] {
    [
        [matrix[0][0].conj(), matrix[1][0].conj()],
        [matrix[0][1].conj(), matrix[1][1].conj()],
    ]
}

#[derive(Clone)]
/// Represents the state of a quantum register.
///
/// The state is represented as a complex vector, where each element corresponds to a probability amplitude for a particular basis state.
/// The number of qubits in the system is also stored, which determines the length of the state vector (2^num_qubits).
pub struct State {
    /// The state vector of the system, represented as a complex vector.
    /// Each element of the vector represents a probability amplitude for a particular state.
    pub state_vector: Vec<Complex<f64>>,

    /// The number of qubits in the system.
    pub num_qubits: usize,
}

impl State {
    /// Creates a new state object with the given state vector.
    ///
    /// # Arguments
    ///
    /// * `state_vector` - The state vector of the system, represented as a complex vector.
    ///
    /// # Returns
    ///
    /// * `state` - A result containing the state object if successful, or an error if the state vector is invalid.
    ///
    /// # Errors
    ///
    /// * Returns an error if the state vector is empty.
    /// * Returns an error if the state vector is not normalised (i.e., the square norm is not 1).
    /// * Returns an error if the number of qubits is invalid (i.e., the length of the state vector is not a power of 2).
    pub fn new(state_vector: Vec<Complex<f64>>) -> Result<Self, Error> {
        let len = state_vector.len();

        if len == 0 {
            return Err(Error::InvalidNumberOfQubits(0));
        }

        // Check if the length of the state vector is a power of 2
        if !len.is_power_of_two() {
            // For error reporting, num_qubits can be approximated or a specific error used.
            return Err(Error::InvalidNumberOfQubits(
                (len as f64).log2().floor() as usize
            ));
        }
        // num_qubits can be safely calculated as len is non-zero and a power of two.
        let num_qubits = len.trailing_zeros() as usize;

        // Check if the square norm (probability) of the state vector is 1
        let norm: f64 = state_vector.par_iter().map(|x| x.norm_sqr()).sum();
        let tol: f64 = f64::EPSILON * len as f64; // Using len directly
        if (norm - 1.0).abs() > tol {
            return Err(Error::StateVectorNotNormalised);
        }

        Ok(Self {
            state_vector,
            num_qubits,
        })
    }

    /// Creates a new Hartree-Fock state object with the given number of electrons and orbitals.
    /// 
    /// # Arguments
    ///
    /// * `num_electrons` - The number of electrons in the system.
    /// 
    /// * `num_orbitals` - The number of orbitals in the system.
    ///
    /// # Returns
    ///
    /// * `state` - A result containing the state object if successful, or an error if the input is invalid.
    pub fn new_hartree_fock(num_electrons: usize, num_orbitals: usize) -> Result<Self, Error> {
        // Validate input (num_orbitals must be > 0 and >= num_electrons)
        if num_orbitals == 0 || num_orbitals < num_electrons {
            return Err(Error::InvalidInputValue(num_orbitals));
        }

        let n = ((1_usize.wrapping_shl(num_electrons as u32)) - 1).wrapping_shl((num_orbitals - num_electrons) as u32);

        Ok(State::new_basis_n(num_orbitals, n)?)
    }

    /// Checks the phase-independent equality of two states
    /// 
    /// # Arguments
    /// 
    /// * `other` - The other state to compare with.
    /// 
    /// # Returns
    /// 
    /// * `true` if the states are equal (ignoring phase), `false` otherwise.
    pub fn equals_without_phase(&self, other: &Self) -> bool {
        if self.num_qubits != other.num_qubits {
            return false;
        }
        // Safe to unwrap since number of qubits is the same
        (self.inner_product(other).unwrap().norm() - 1.0).abs() < f32::EPSILON.into()
    }

    /// Creates a new state object with the given number of qubits initialised to the |0...0> state.
    ///
    /// # Arguments
    ///
    /// * `num_qubits` - The number of qubits in the system.
    ///
    /// # Returns
    ///
    /// * `state` - A result containing the state object if successful, or an error if the number of qubits is invalid.
    ///
    /// # Errors
    ///
    /// * Returns an error if the number of qubits is 0.
    pub fn new_zero(num_qubits: usize) -> Result<Self, Error> {
        if num_qubits == 0 {
            return Err(Error::InvalidNumberOfQubits(num_qubits));
        }

        let dim: usize = 1 << num_qubits;
        let mut state_vector: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); dim];
        state_vector[0] = Complex::new(1.0, 0.0); // |0...0> state has amplitude 1.0 at index 0

        Ok(Self {
            state_vector,
            num_qubits,
        })
    }

    /// Creates a new state object with the given number of qubits initialised to the `n`-th basis state.
    ///
    /// # Arguments
    ///
    /// * `num_qubits` - The number of qubits in the system.
    /// * `n` - The index of the basis state to initialise to.
    ///
    /// # Returns
    ///
    /// * `state` - A result containing a new state object with the specified number of qubits, initialised to the `n`-th basis state or an error if `n` is out of bounds or if the number of qubits is invalid.
    ///
    /// # Errors
    ///
    /// * Returns an error if the number of qubits is 0 or if `n` is out of bounds for the given number of qubits.
    pub fn new_basis_n(num_qubits: usize, n: usize) -> Result<Self, Error> {
        let dim: usize = 1 << num_qubits;
        if n >= dim {
            return Err(Error::InvalidQubitIndex(n, num_qubits));
        }
        if num_qubits == 0 {
            return Err(Error::InvalidNumberOfQubits(num_qubits));
        }

        let mut state_vector: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); dim];
        state_vector[n] = Complex::new(1.0, 0.0); // |n> state has amplitude 1.0 at index n

        Ok(Self {
            state_vector,
            num_qubits,
        })
    }

    /// Creates a new plus state object with the given number of qubits initialised to the |+...+> state.
    ///
    /// # Arguments
    ///
    /// * `num_qubits` - The number of qubits in the system.
    ///
    /// # Returns
    ///
    /// * `state` - A result containing a new state object with the specified number of qubits, initialised to the |+...+> state or an error if the number of qubits is invalid.
    ///
    /// # Errors
    ///
    /// * Returns an error if the number of qubits is 0.
    pub fn new_plus(num_qubits: usize) -> Result<Self, Error> {
        if num_qubits == 0 {
            return Err(Error::InvalidNumberOfQubits(num_qubits));
        }
        let dim: usize = 1 << num_qubits;
        let amplitude = Complex::new(1.0 / (dim as f64).sqrt(), 0.0);
        let state_vector: Vec<Complex<f64>> = vec![amplitude; dim];

        Ok(Self {
            state_vector,
            num_qubits,
        })
    }

    /// Creates a new minus state object with the given number of qubits initialised to the |-...-> state.
    ///
    /// # Arguments
    ///
    /// * `num_qubits` - The number of qubits in the system.
    ///
    /// # Returns
    ///
    /// * `state` - A result containing a new state object with the specified number of qubits, initialised to the |-...-> state or an error if the number of qubits is invalid.
    ///
    /// # Errors
    ///
    /// * Returns an error if the number of qubits is 0.
    pub fn new_minus(num_qubits: usize) -> Result<Self, Error> {
        if num_qubits == 0 {
            return Err(Error::InvalidNumberOfQubits(num_qubits));
        }
        let dim: usize = 1 << num_qubits;
        let amplitude: Complex<f64> = Complex::new(1.0 / (dim as f64).sqrt(), 0.0);
        const PARALLEL_THRESHOLD: usize = 1 << 6; // Threshold for parallelization

        let state_vector: Vec<Complex<f64>> = if dim > PARALLEL_THRESHOLD {
            (0..dim)
                .into_par_iter()
                .map(|i| {
                    let num_ones = i.count_ones() as usize;
                    if num_ones % 2 == 0 {
                        amplitude
                    } else {
                        -amplitude
                    }
                })
                .collect()
        } else {
            let mut sv = Vec::with_capacity(dim);
            for i in 0..dim {
                let num_ones = i.count_ones() as usize;
                sv.push(if num_ones % 2 == 0 {
                    amplitude
                } else {
                    -amplitude
                });
            }
            sv
        };

        Ok(Self {
            state_vector,
            num_qubits,
        })
    }

    /// Returns the probability of a basis state at index `n` in the state vector.
    ///
    /// # Arguments
    ///
    /// * `n` - The index of the basis state to get the probability for.
    ///
    /// # Returns
    ///
    /// * `probability` - The probability of the basis state at index `n`.
    ///
    /// # Errors
    ///
    /// * Returns an error if `n` is out of bounds for the state vector.
    pub fn probability(&self, n: usize) -> Result<f64, Error> {
        if n >= self.state_vector.len() {
            return Err(Error::InvalidQubitIndex(n, self.num_qubits));
        }
        let amplitude: Complex<f64> = self.state_vector[n];
        Ok(amplitude.norm_sqr())
    }

    /// Returns the number of qubits in the state vector.
    ///
    /// # Returns
    ///
    /// * `num_qubits` - The number of qubits in the state vector.
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Returns the amplitude of the basis state at index `n` in the state vector.
    ///     
    /// # Arguments
    ///
    /// * `n` - The index of the basis state to get the amplitude for.
    ///
    /// # Returns
    ///
    /// * `amplitude` - The amplitude of the basis state at index `n`.
    ///
    /// # Errors
    ///
    /// * Returns an error if `n` is out of bounds for the state vector.
    pub fn amplitude(&self, n: usize) -> Result<Complex<f64>, Error> {
        if n >= self.state_vector.len() {
            return Err(Error::InvalidQubitIndex(n, self.num_qubits));
        }
        Ok(self.state_vector[n])
    }

    // ***** MEASUREMENT FUNCTIONS *****

    fn _measure_computational(
        &self,
        measured_qubits: &[usize],
    ) -> Result<MeasurementResult, Error> {
        self.measure(MeasurementBasis::Computational, measured_qubits)
    }

    /// Measures the state vector in the specified basis and returns the measurement result.
    ///
    /// # Arguments
    ///
    /// * `basis` - The basis to measure in.
    /// * `indices` - The indices of the qubits to measure. If `indices` is empty, all qubits are measured.
    ///
    /// # Returns
    ///
    /// * `result` - A result containing the measurement result if successful, or an error if the measurement fails.
    ///
    /// # Errors
    ///
    /// * Returns an error if the measurement fails.
    /// * Returns an error if the number of qubits is invalid.
    /// * Returns an error if the indices are out of bounds for the state vector.
    pub fn measure(
        &self,
        basis: MeasurementBasis,
        measured_qubits: &[usize],
    ) -> Result<MeasurementResult, Error> {
        // If no indices are provided, measure all qubits
        let all_qubits: Vec<usize> = if measured_qubits.is_empty() {
            (0..self.num_qubits).collect()
        } else {
            Vec::new()
        };

        let actual_measured_qubits: &[usize] = if measured_qubits.is_empty() {
            &all_qubits
        } else {
            measured_qubits
        };

        // Check for valid indices
        let num_measured: usize = actual_measured_qubits.len();
        if num_measured > self.num_qubits {
            return Err(Error::InvalidNumberOfQubits(self.num_qubits));
        }
        for &index in actual_measured_qubits {
            if index >= self.num_qubits {
                return Err(Error::InvalidQubitIndex(index, self.num_qubits));
            }
        }

        match basis {
            MeasurementBasis::Computational => {
                let num_outcomes: usize = 1 << num_measured;

                // Calculate probabilities for each outcome (outcome as a single integer 0..num_outcomes-1)
                let probabilities: Vec<f64> = self
                    .state_vector
                    .par_iter()
                    .enumerate()
                    .fold(
                        || vec![0.0; num_outcomes], // Thread-local accumulator
                        |mut acc_probs, (idx, amplitude)| {
                            let mut outcome_val_for_this_state = 0;
                            // Extract the bits corresponding to measured_qubits to form the outcome value
                            for (bit_idx, &qubit_pos) in actual_measured_qubits.iter().enumerate() {
                                if (idx >> qubit_pos) & 1 != 0 {
                                    outcome_val_for_this_state |= 1 << bit_idx;
                                }
                            }
                            // Ensure outcome_val_for_this_state is within bounds (0 to num_outcomes-1)
                            if outcome_val_for_this_state < num_outcomes {
                                acc_probs[outcome_val_for_this_state] += amplitude.norm_sqr();
                            }
                            acc_probs
                        },
                    )
                    .reduce(
                        || vec![0.0; num_outcomes], // Initialiser for combining thread-local results
                        |mut total_probs, local_probs| {
                            for i in 0..num_outcomes {
                                total_probs[i] += local_probs[i];
                            }
                            total_probs
                        },
                    );

                // Normalise probabilities
                let total_probability: f64 = probabilities.iter().sum();
                if total_probability < f64::EPSILON {
                    return Err(Error::UnknownError);
                }
                let normalised_probabilities: Vec<f64> = probabilities
                    .iter()
                    .map(|&prob| prob / total_probability)
                    .collect();

                // Sample an outcome based on the probabilities
                let mut rng = rand::rng();
                let random_value: f64 = rng.random_range(0.0..1.0);

                let mut cumulative_probability: f64 = 0.0;
                let mut sampled_outcome_int: usize = 0;

                // Sample loop
                for (i, &prob) in normalised_probabilities.iter().enumerate() {
                    cumulative_probability += prob;
                    // Sample if random_value falls into the cumulative probability bin
                    if random_value < cumulative_probability {
                        sampled_outcome_int = i;
                        break; // Found the outcome, exit loop
                    }
                }
                // If, due to floating point issues, no outcome was selected, select the last one.
                if random_value >= cumulative_probability && !normalised_probabilities.is_empty() {
                    sampled_outcome_int = normalised_probabilities.len() - 1;
                }

                // Collapse the state vector into a new vector
                let mut collapsed_state_data: Vec<Complex<f64>> =
                    vec![Complex::new(0.0, 0.0); self.state_vector.len()];

                let normalisation_sq: f64 = collapsed_state_data
                    .par_iter_mut()
                    .enumerate()
                    .zip(self.state_vector.par_iter()) // Zip with original state vector amplitudes
                    .map(|((idx, collapsed_amp_ref), &original_amp)| {
                        let mut current_outcome_val_for_this_state = 0;
                        // Extract the bits corresponding to measured_qubits
                        for (bit_idx, &qubit_pos) in actual_measured_qubits.iter().enumerate() {
                            if (idx >> qubit_pos) & 1 != 0 {
                                current_outcome_val_for_this_state |= 1 << bit_idx;
                            }
                        }

                        if current_outcome_val_for_this_state == sampled_outcome_int {
                            *collapsed_amp_ref = original_amp;
                            original_amp.norm_sqr() // Contribution to normalisation_sq
                        } else {
                            // *collapsed_amp_ref remains Complex::new(0.0, 0.0)
                            0.0 // No contribution
                        }
                    })
                    .sum(); // Sums up all contributions to get total normalisation_sq

                // Renormalise the new collapsed state vector
                if normalisation_sq > f64::EPSILON {
                    let norm_factor: f64 = normalisation_sq.sqrt();
                    for amplitude in collapsed_state_data.iter_mut() {
                        *amplitude /= norm_factor;
                    }
                }

                // Convert the sampled integer outcome to a Vec<u8>
                let mut outcome_binary_vec: Vec<u8> = vec![0; num_measured];
                for (i, qubit_pos) in outcome_binary_vec.iter_mut().enumerate() {
                    *qubit_pos = ((sampled_outcome_int >> i) & 1) as u8;
                }

                // Create the measurement result
                Ok(MeasurementResult {
                    basis,
                    indices: actual_measured_qubits.to_vec(),
                    outcomes: outcome_binary_vec,
                    new_state: State::new(collapsed_state_data)?,
                })
            }
            MeasurementBasis::X => {
                // Apply Hadamard to measured qubits
                let transformed_state = self.h_multi(actual_measured_qubits)?;
                // Measure in computational basis
                let computational_measurement_result =
                    transformed_state._measure_computational(actual_measured_qubits)?;
                // Transform the new state back by applying Hadamard again
                let final_state = computational_measurement_result
                    .new_state
                    .h_multi(actual_measured_qubits)?;
                Ok(MeasurementResult {
                    basis: MeasurementBasis::X,
                    indices: computational_measurement_result.indices,
                    outcomes: computational_measurement_result.outcomes,
                    new_state: final_state,
                })
            }
            MeasurementBasis::Y => {
                // Apply Sdg then H to measured qubits
                let state_after_sdag = self.s_dag_multi(actual_measured_qubits)?;
                let transformed_state = state_after_sdag.h_multi(actual_measured_qubits)?;
                // Measure in computational basis
                let computational_measurement_result =
                    transformed_state._measure_computational(actual_measured_qubits)?;
                // Transform the new state back by applying H then S
                let state_after_h = computational_measurement_result
                    .new_state
                    .h_multi(actual_measured_qubits)?;
                let final_state = state_after_h.s_multi(actual_measured_qubits)?;
                Ok(MeasurementResult {
                    basis: MeasurementBasis::Y,
                    indices: computational_measurement_result.indices,
                    outcomes: computational_measurement_result.outcomes,
                    new_state: final_state,
                })
            }
            MeasurementBasis::Custom(u_matrix) => {
                // Apply the custom unitary U to measured qubits
                let transformed_state = self.unitary_multi(actual_measured_qubits, u_matrix)?;

                // Measure in computational basis
                let computational_measurement_result =
                    transformed_state._measure_computational(actual_measured_qubits)?;

                // Calculate U_dagger (adjoint of u_matrix)
                let u_dagger_matrix = calculate_adjoint(&u_matrix);

                // Transform the new state back by applying U_dagger
                let final_state = computational_measurement_result
                    .new_state
                    .unitary_multi(actual_measured_qubits, u_dagger_matrix)?;

                Ok(MeasurementResult {
                    basis: MeasurementBasis::Custom(u_matrix),
                    indices: computational_measurement_result.indices,
                    outcomes: computational_measurement_result.outcomes,
                    new_state: final_state,
                })
            }
        }
    }

    /// Measures the state vector `n` times in the specified basis and returns the measurement results.
    ///
    /// # Arguments
    ///
    /// * `basis` - The basis to measure in.
    /// * `indices` - The indices of the qubits to measure. If `indices` is empty, all qubits are measured.
    /// * `n` - The number of measurements to perform.
    ///
    /// # Returns
    ///
    /// * `results` - A result containing a vector of measurement results if successful, or an error if the measurement fails.
    ///
    /// # Errors
    ///
    /// * Returns an error if the measurement fails.
    /// * Returns an error if the number of qubits is invalid.
    /// * Returns an error if the indices are out of bounds for the state vector.
    /// * Returns an error if `n` is 0.
    pub fn measure_n(
        &self,
        basis: MeasurementBasis,
        measured_qubits: &[usize],
        n: usize,
    ) -> Result<Vec<MeasurementResult>, Error> {
        if n == 0 {
            return Err(Error::InvalidNumberOfMeasurements(0));
        }

        // If no indices are provided, measure all qubits
        let all_indices: Vec<usize> = (0..self.num_qubits).collect();
        let actual_measured_qubits: &[usize] = if measured_qubits.is_empty() {
            &all_indices
        } else {
            measured_qubits
        };

        // Check for valid indices
        let num_measured: usize = actual_measured_qubits.len();
        if num_measured > self.num_qubits {
            return Err(Error::InvalidNumberOfQubits(self.num_qubits));
        }
        for &index in actual_measured_qubits {
            if index >= self.num_qubits {
                return Err(Error::InvalidQubitIndex(index, self.num_qubits));
            }
        }

        let results: Vec<MeasurementResult> = (0..n)
            .into_par_iter()
            .map(|_| self.measure(basis, actual_measured_qubits))
            .collect::<Result<Vec<MeasurementResult>, Error>>()?;
        Ok(results)
    }

    /// Performs a tensor product of two state vectors and returns the resulting state.
    /// Uses parallel computation if the resulting dimension is large enough.
    ///
    /// # Arguments
    ///
    /// * `other` - The other state vector to perform the tensor product with.
    ///
    /// # Returns
    ///
    /// * `result` - A result containing the new state object if successful, or an error if the operation fails.
    ///
    /// # Errors
    ///
    /// * Returns an error if either state vector is empty.
    /// * Returns an error if either state vector has an invalid number of qubits.
    pub fn tensor_product(&self, other: &Self) -> Result<Self, Error> {
        if self.num_qubits == 0 || other.num_qubits == 0 {
            return Err(Error::InvalidNumberOfQubits(0));
        }

        let new_num_qubits: usize = self.num_qubits + other.num_qubits;
        let new_dim: usize = 1 << new_num_qubits;
        let other_dim: usize = 1 << other.num_qubits; // Cache dimension of other state

        // Threshold for using parallel computation
        const PARALLEL_THRESHOLD: usize = 1 << 6; // Parallelise when dimension is larger than 64

        let new_state_vector: Vec<Complex<f64>> = if new_dim > PARALLEL_THRESHOLD {
            // Parallel calculation for large states
            (0..new_dim)
                .into_par_iter()
                .map(|new_index| {
                    let i: usize = new_index >> other.num_qubits; // Index for self.state_vector
                    let j: usize = new_index & (other_dim - 1); // Index for other.state_vector
                    self.state_vector[i] * other.state_vector[j]
                })
                .collect()
        } else {
            // Sequential calculation for smaller states
            let mut temp_state_vector: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); new_dim];
            for i in 0..self.state_vector.len() {
                for j in 0..other.state_vector.len() {
                    temp_state_vector[(i * other_dim) + j] =
                        self.state_vector[i] * other.state_vector[j];
                }
            }
            temp_state_vector
        };

        Self::new(new_state_vector) // For normalisation check
    }

    /// Performs a tensor product of two state vectors without checking for validity.
    #[allow(dead_code)]
    pub(crate) fn tensor_product_unchecked(&self, other: &Self) -> Self {
        let new_num_qubits: usize = self.num_qubits + other.num_qubits;
        let new_dim: usize = 1 << new_num_qubits;
        let other_dim: usize = 1 << other.num_qubits; // Cache dimension of other state

        // Threshold for using parallel computation
        const PARALLEL_THRESHOLD: usize = 1 << 6; // Parallelise when dimension is larger than 64

        let new_state_vector: Vec<Complex<f64>> = if new_dim > PARALLEL_THRESHOLD {
            // Parallel calculation for large states
            (0..new_dim)
                .into_par_iter()
                .map(|new_index| {
                    let i: usize = new_index >> other.num_qubits; // Index for self.state_vector
                    let j: usize = new_index & (other_dim - 1); // Index for other.state_vector
                    self.state_vector[i] * other.state_vector[j]
                })
                .collect()
        } else {
            // Sequential calculation for smaller states
            let mut temp_state_vector: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); new_dim];
            for i in 0..self.state_vector.len() {
                for j in 0..other.state_vector.len() {
                    temp_state_vector[(i * other_dim) + j] =
                        self.state_vector[i] * other.state_vector[j];
                }
            }
            temp_state_vector
        };

        Self {
            state_vector: new_state_vector,
            num_qubits: new_num_qubits,
        }
    }

    /// Performs an inner product of two state vectors and returns the resulting complex number.
    ///
    /// # Arguments
    ///
    /// * `other` - The other state vector to perform the inner product with.
    ///
    /// # Returns
    ///
    /// * `result` - A result containing the inner product as a complex number if successful, or an error if the operation fails.
    ///
    /// # Errors
    ///
    /// * Returns an error if either state vector is empty.
    /// * Returns an error if the state vectors have different dimensions.
    pub fn inner_product(&self, other: &Self) -> Result<Complex<f64>, Error> {
        if self.num_qubits == 0 || other.num_qubits == 0 {
            return Err(Error::InvalidNumberOfQubits(0));
        }

        if self.state_vector.len() != other.state_vector.len() {
            return Err(Error::InvalidNumberOfQubits(self.state_vector.len()));
        }

        const PARALLEL_THRESHOLD: usize = 1 << 6; // Threshold for parallelisation
        let len = self.state_vector.len();

        let inner_product: Complex<f64> = if len > PARALLEL_THRESHOLD {
            self.state_vector
                .par_iter()
                .zip(other.state_vector.par_iter())
                .map(|(a, b)| a.conj() * b)
                .sum()
        } else {
            self.state_vector
                .iter()
                .zip(other.state_vector.iter())
                .map(|(a, b)| a.conj() * b)
                .sum()
        };

        Ok(inner_product)
    }

    // ***** OPERATION FUNCTIONS *****

    /// Applies a unitary operation to the state vector.
    ///
    /// # Arguments
    ///
    /// * `unitary` - The unitary operation to apply, represented as an `Operator` trait object.
    ///
    /// * target_qubits - The indices of the qubits to apply the unitary operation to.
    ///
    /// * control_qubits - The indices of the control qubits for the unitary operation, if any.
    ///
    /// # Returns
    ///
    /// * `Result` - A result containing the new state object if successful, or an error if the operation fails.
    ///
    /// # Errors
    ///
    /// * Returns an error if the unitary operation is invalid.
    ///
    /// * Returns an error if the number of qubits provided is invalid.
    ///
    /// * Returns an error if the indices are out of bounds for the state vector.
    pub fn operate(
        &self,
        unitary: impl Operator,
        target_qubits: &[usize],
        control_qubits: &[usize],
    ) -> Result<Self, Error> {
        // Check for valid indices
        let num_target: usize = target_qubits.len();
        let num_control: usize = control_qubits.len();

        if unitary.base_qubits() != (num_target + num_control) {
            return Err(Error::InvalidNumberOfQubits(unitary.base_qubits()));
        }

        if num_target > self.num_qubits {
            return Err(Error::InvalidNumberOfQubits(self.num_qubits));
        }

        for &index in target_qubits {
            if index >= self.num_qubits {
                return Err(Error::InvalidQubitIndex(index, self.num_qubits));
            }
        }

        for &index in control_qubits {
            if index >= self.num_qubits {
                return Err(Error::InvalidQubitIndex(index, self.num_qubits));
            }
        }

        // Apply the unitary operation to the state vector and return the new state
        unitary.apply(self, target_qubits, control_qubits)
    }

    // -- SINGLE-QUBIT GATES --

    /// Applies the Hadamard gate to the specified qubit in the state vector.
    ///
    /// # Arguments
    ///
    /// * `index` - The index of the qubit to apply the Hadamard gate to.
    ///
    /// # Returns
    ///
    /// * `Result` - A result containing the new state object if successful, or an error if the operation fails.
    ///
    /// # Errors
    ///
    /// * Returns an error if the index is out of bounds for the state vector.
    pub fn h(&self, index: usize) -> Result<Self, Error> {
        Hadamard {}.apply(self, &[index], &[])
    }

    /// Applies the Hadamard gate to the specified qubits in the state vector in the given order.
    ///
    /// # Arguments
    ///
    /// * `qubits` - The indices of the qubits to apply the Hadamard gate to.
    ///
    /// # Returns
    ///
    /// * `Result` - A result containing the new state object if successful, or an error if the operation fails.
    ///
    /// * # Errors
    ///
    /// * Returns an error if the number of qubits provided is invalid.
    ///
    /// * Returns an error if the indices are out of bounds for the state vector.
    pub fn h_multi(&self, qubits: &[usize]) -> Result<Self, Error> {
        let mut new_state: State = self.clone();
        let h: Hadamard = Hadamard {};
        for &qubit in qubits {
            new_state = h.apply(&new_state, &[qubit], &[])?;
        }
        Ok(new_state)
    }

    /// Applies the controlled Hadamard gate to the specified qubits in the state vector in the given order.
    ///
    /// # Arguments
    ///
    /// * `target_qubits` - The indices of the target qubits to apply the controlled Hadamard gate to.
    ///
    /// * `control_qubits` - The indices of the control qubits for the controlled Hadamard gate.
    ///
    /// # Returns
    ///
    /// * `Result` - A result containing the new state object if successful, or an error if the operation fails.
    ///
    /// * # Errors
    ///
    /// * Returns an error if the number of qubits provided is invalid.
    ///
    /// * Returns an error if the indices are out of bounds for the state vector.
    ///
    /// * Returns an error if the control qubits and target qubits overlap.
    pub fn ch_multi(
        &self,
        target_qubits: &[usize],
        control_qubits: &[usize],
    ) -> Result<Self, Error> {
        let mut new_state: State = self.clone();
        let h: Hadamard = Hadamard {};
        for &qubit in target_qubits {
            new_state = h.apply(&new_state, &[qubit], control_qubits)?;
        }
        Ok(new_state)
    }

    /// Applies the Pauli-X (NOT) gate to the specified qubit in the state vector.
    ///
    /// # Arguments
    ///
    /// * `index` - The index of the qubit to apply the Pauli-X gate to.
    ///
    /// # Returns
    ///
    /// * `Result` - A result containing the new state object if successful, or an error if the operation fails.
    ///
    /// # Errors
    ///
    /// * Returns an error if the index is out of bounds for the state vector.
    pub fn x(&self, index: usize) -> Result<Self, Error> {
        Pauli::X.apply(self, &[index], &[])
    }

    /// Applies the Pauli-X (NOT) gate to the specified qubits in the state vector in the given order.
    ///
    /// # Arguments
    ///
    /// * `qubits` - The indices of the qubits to apply the Pauli-X gate to.
    ///
    /// # Returns
    ///
    /// * `Result` - A result containing the new state object if successful, or an error if the operation fails.
    ///
    /// # Errors
    ///
    /// * Returns an error if the number of qubits provided is invalid.
    ///
    /// * Returns an error if the indices are out of bounds for the state vector.
    pub fn x_multi(&self, qubits: &[usize]) -> Result<Self, Error> {
        let mut new_state: State = self.clone();
        let x: Pauli = Pauli::X;
        for &qubit in qubits {
            new_state = x.apply(&new_state, &[qubit], &[])?;
        }
        Ok(new_state)
    }

    /// Applies the controlled Pauli-X (NOT) gate to the specified qubits in the state vector in the given order.
    ///
    /// # Arguments
    ///
    /// * `target_qubits` - The indices of the target qubits to apply the controlled Pauli-X gate to.
    /// * `control_qubits` - The indices of the control qubits for the controlled Pauli-X gate.
    ///
    /// # Returns
    ///
    /// * `Result` - A result containing the new state object if successful, or an error if the operation fails.
    ///
    /// # Errors
    ///
    /// * Returns an error if the number of qubits provided is invalid.
    /// * Returns an error if the indices are out of bounds for the state vector.
    /// * Returns an error if the control qubits and target qubits overlap.
    pub fn cx_multi(
        &self,
        target_qubits: &[usize],
        control_qubits: &[usize],
    ) -> Result<Self, Error> {
        let mut new_state: State = self.clone();
        let x: Pauli = Pauli::X;
        for &qubit in target_qubits {
            new_state = x.apply(&new_state, &[qubit], control_qubits)?;
        }
        Ok(new_state)
    }

    /// Applies the Pauli-Y gate to the specified qubit in the state vector.
    ///
    /// # Arguments
    ///
    /// * `index` - The index of the qubit to apply the Pauli-Y gate to.
    ///
    /// # Returns
    ///
    /// * `Result` - A result containing the new state object if successful, or an error if the operation fails.
    ///
    /// # Errors
    ///
    /// * Returns an error if the index is out of bounds for the state vector.
    pub fn y(&self, index: usize) -> Result<Self, Error> {
        Pauli::Y.apply(self, &[index], &[])
    }

    /// Applies the Pauli-Y gate to the specified qubits in the state vector in the given order.
    ///
    /// # Arguments
    ///
    /// * `qubits` - The indices of the qubits to apply the Pauli-Y gate to.
    ///
    /// # Returns
    ///
    /// * `Result` - A result containing the new state object if successful, or an error if the operation fails.
    ///
    /// # Errors
    ///
    /// * Returns an error if the number of qubits provided is invalid.
    ///
    /// * Returns an error if the indices are out of bounds for the state vector.
    pub fn y_multi(&self, qubits: &[usize]) -> Result<Self, Error> {
        let mut new_state: State = self.clone();
        let y: Pauli = Pauli::Y;
        for &qubit in qubits {
            new_state = y.apply(&new_state, &[qubit], &[])?;
        }
        Ok(new_state)
    }

    /// Applies the controlled Pauli-Y gate to the specified qubits in the state vector in the given order.
    ///
    /// # Arguments
    ///
    /// * `target_qubits` - The indices of the target qubits to apply the controlled Pauli-Y gate to.
    /// * `control_qubits` - The indices of the control qubits for the controlled Pauli-Y gate.
    ///
    /// # Returns
    ///
    /// * `Result` - A result containing the new state object if successful, or an error if the operation fails.
    ///
    /// # Errors
    ///
    /// * Returns an error if the number of qubits provided is invalid.
    /// * Returns an error if the indices are out of bounds for the state vector.
    /// * Returns an error if the control qubits and target qubits overlap.
    pub fn cy_multi(
        &self,
        target_qubits: &[usize],
        control_qubits: &[usize],
    ) -> Result<Self, Error> {
        let mut new_state: State = self.clone();
        let y: Pauli = Pauli::Y;
        for &qubit in target_qubits {
            new_state = y.apply(&new_state, &[qubit], control_qubits)?;
        }
        Ok(new_state)
    }

    /// Applies the Pauli-Z gate to the specified qubit in the state vector.
    ///
    /// # Arguments
    ///
    /// * `index` - The index of the qubit to apply the Pauli-Z gate to.
    ///
    /// # Returns
    ///
    /// * `Result` - A result containing the new state object if successful, or an error if the operation fails.
    ///
    /// # Errors
    ///
    /// * Returns an error if the index is out of bounds for the state vector.
    pub fn z(&self, index: usize) -> Result<Self, Error> {
        Pauli::Z.apply(self, &[index], &[])
    }

    /// Applies the Pauli-Z gate to the specified qubits in the state vector in the given order.
    ///
    /// # Arguments
    ///
    /// * `qubits` - The indices of the qubits to apply the Pauli-Z gate to.
    ///
    /// # Returns
    ///
    /// * `Result` - A result containing the new state object if successful, or an error if the operation fails.
    ///
    /// # Errors
    ///
    /// * Returns an error if the number of qubits provided is invalid.
    ///
    /// * Returns an error if the indices are out of bounds for the state vector.
    pub fn z_multi(&self, qubits: &[usize]) -> Result<Self, Error> {
        let mut new_state: State = self.clone();
        let z: Pauli = Pauli::Z;
        for &qubit in qubits {
            new_state = z.apply(&new_state, &[qubit], &[])?;
        }
        Ok(new_state)
    }

    /// Applies the controlled Pauli-Z gate to the specified qubits in the state vector in the given order.
    ///
    /// # Arguments
    ///
    /// * `target_qubits` - The indices of the target qubits to apply the controlled Pauli-Z gate to.
    /// * `control_qubits` - The indices of the control qubits for the controlled Pauli-Z gate.
    ///
    /// # Returns
    ///
    /// * `Result` - A result containing the new state object if successful, or an error if the operation fails.
    ///
    /// # Errors
    ///
    /// * Returns an error if the number of qubits provided is invalid.
    /// * Returns an error if the indices are out of bounds for the state vector.
    /// * Returns an error if the control qubits and target qubits overlap.
    pub fn cz_multi(
        &self,
        target_qubits: &[usize],
        control_qubits: &[usize],
    ) -> Result<Self, Error> {
        let mut new_state: State = self.clone();
        let z: Pauli = Pauli::Z;
        for &qubit in target_qubits {
            new_state = z.apply(&new_state, &[qubit], control_qubits)?;
        }
        Ok(new_state)
    }

    /// Applies the Identity gate to the state vector.
    ///
    /// # Arguments
    ///
    /// * `qubit` - The index of the qubit to apply the Identity gate to.
    ///
    /// # Returns
    ///
    /// * `Result` - A result containing the new state object if successful, or an error if the operation fails.
    ///
    /// # Errors
    ///
    /// * Returns an error if the index is out of bounds for the state vector.
    pub fn i(&self, qubit: usize) -> Result<Self, Error> {
        Identity {}.apply(self, &[qubit], &[])
    }

    /// Applies the Identity gate to the state vector for multiple qubits in the given order.
    ///
    /// # Arguments
    ///
    /// * `qubits` - The indices of the qubits to apply the Identity gate to.
    ///
    /// # Returns
    ///
    /// * `Result` - A result containing the new state object if successful, or an error if the operation fails.
    ///
    /// # Errors
    ///
    /// * Returns an error if the number of qubits provided is invalid.
    ///
    /// * Returns an error if the indices are out of bounds for the state vector.
    pub fn i_multi(&self, qubits: &[usize]) -> Result<Self, Error> {
        let mut new_state: State = self.clone();
        let i: Identity = Identity {};
        for &qubit in qubits {
            new_state = i.apply(&new_state, &[qubit], &[])?;
        }
        Ok(new_state)
    }

    /// Applies the controlled Identity gate to the state vector for multiple qubits in the given order.
    ///
    /// # Arguments
    ///
    /// * `target_qubits` - The indices of the target qubits to apply the controlled Identity gate to.
    /// * `control_qubits` - The indices of the control qubits for the controlled Identity gate.
    ///
    /// # Returns
    ///
    /// * `Result` - A result containing the new state object if successful, or an error if the operation fails.
    ///
    /// # Errors
    ///
    /// * Returns an error if the number of qubits provided is invalid.
    /// * Returns an error if the indices are out of bounds for the state vector.
    /// * Returns an error if the control qubits and target qubits overlap.
    pub fn ci_multi(
        &self,
        target_qubits: &[usize],
        control_qubits: &[usize],
    ) -> Result<Self, Error> {
        let mut new_state: State = self.clone();
        let i: Identity = Identity {};
        for &qubit in target_qubits {
            new_state = i.apply(&new_state, &[qubit], control_qubits)?;
        }
        Ok(new_state)
    }

    /// Applies the Phase S gate to the specified qubit in the state vector.
    ///
    /// # Arguments
    ///
    /// * `index` - The index of the qubit to apply the Phase S gate to.
    ///
    /// # Returns
    ///
    /// * `Result` - A result containing the new state object if successful, or an error if the operation fails.
    ///
    /// # Errors
    ///
    /// * Returns an error if the index is out of bounds for the state vector.
    pub fn s(&self, index: usize) -> Result<Self, Error> {
        PhaseS {}.apply(self, &[index], &[])
    }

    /// Applies the Phase S gate to the specified qubits in the state vector in the given order.
    ///
    /// # Arguments
    ///
    /// * `qubits` - The indices of the qubits to apply the Phase S gate to.
    ///
    /// # Returns
    ///
    /// * `Result` - A result containing the new state object if successful, or an error if the operation fails.
    ///
    /// # Errors
    ///
    /// * Returns an error if the number of qubits provided is invalid.
    ///
    /// * Returns an error if the indices are out of bounds for the state vector.
    pub fn s_multi(&self, qubits: &[usize]) -> Result<Self, Error> {
        let mut new_state: State = self.clone();
        let s_gate: PhaseS = PhaseS {};
        for &qubit in qubits {
            new_state = s_gate.apply(&new_state, &[qubit], &[])?;
        }
        Ok(new_state)
    }

    /// Applies the controlled Phase S gate to the specified qubits in the state vector in the given order.
    ///
    /// # Arguments
    ///
    /// * `target_qubits` - The indices of the target qubits to apply the controlled Phase S gate to.
    /// * `control_qubits` - The indices of the control qubits for the controlled Phase S gate.
    ///
    /// # Returns
    ///
    /// * `Result` - A result containing the new state object if successful, or an error if the operation fails.
    ///
    /// # Errors
    ///
    /// * Returns an error if the number of qubits provided is invalid.
    /// * Returns an error if the indices are out of bounds for the state vector.
    /// * Returns an error if the control qubits and target qubits overlap.
    pub fn cs_multi(
        &self,
        target_qubits: &[usize],
        control_qubits: &[usize],
    ) -> Result<Self, Error> {
        let mut new_state: State = self.clone();
        let s_gate: PhaseS = PhaseS {};
        for &qubit in target_qubits {
            new_state = s_gate.apply(&new_state, &[qubit], control_qubits)?;
        }
        Ok(new_state)
    }

    /// Applies the Phase T gate to the specified qubit in the state vector.
    ///
    /// # Arguments
    ///
    /// * `index` - The index of the qubit to apply the Phase T gate to.
    ///
    /// # Returns
    ///
    /// * `Result` - A result containing the new state object if successful, or an error if the operation fails.
    ///
    /// # Errors
    ///
    /// * Returns an error if the index is out of bounds for the state vector.
    pub fn t(&self, index: usize) -> Result<Self, Error> {
        PhaseT {}.apply(self, &[index], &[])
    }

    /// Applies the Phase T gate to the specified qubits in the state vector in the given order.
    ///
    /// # Arguments
    ///
    /// * `qubits` - The indices of the qubits to apply the Phase T gate to.
    ///
    /// # Returns
    ///
    /// * `Result` - A result containing the new state object if successful, or an error if the operation fails.
    ///
    /// # Errors
    ///
    /// * Returns an error if the number of qubits provided is invalid.
    ///
    /// * Returns an error if the indices are out of bounds for the state vector.
    pub fn t_multi(&self, qubits: &[usize]) -> Result<Self, Error> {
        let mut new_state: State = self.clone();
        let t_gate: PhaseT = PhaseT {};
        for &qubit in qubits {
            new_state = t_gate.apply(&new_state, &[qubit], &[])?;
        }
        Ok(new_state)
    }

    /// Applies the controlled Phase T gate to the specified qubits in the state vector in the given order.
    ///
    /// # Arguments
    ///
    /// * `target_qubits` - The indices of the target qubits to apply the controlled Phase T gate to.
    /// * `control_qubits` - The indices of the control qubits for the controlled Phase T gate.
    ///
    /// # Returns
    ///
    /// * `Result` - A result containing the new state object if successful, or an error if the operation fails.
    ///
    /// # Errors
    ///
    /// * Returns an error if the number of qubits provided is invalid.
    /// * Returns an error if the indices are out of bounds for the state vector.
    /// * Returns an error if the control qubits and target qubits overlap.
    pub fn ct_multi(
        &self,
        target_qubits: &[usize],
        control_qubits: &[usize],
    ) -> Result<Self, Error> {
        let mut new_state: State = self.clone();
        let t_gate: PhaseT = PhaseT {};
        for &qubit in target_qubits {
            new_state = t_gate.apply(&new_state, &[qubit], control_qubits)?;
        }
        Ok(new_state)
    }

    /// Applies the Phase S dagger gate to the specified qubit in the state vector.
    ///
    /// # Arguments
    ///
    /// * `index` - The index of the qubit to apply the Phase S dagger gate to.
    ///
    /// # Returns
    ///
    /// * `Result` - A result containing the new state object if successful, or an error if the operation fails.
    ///
    /// # Errors
    ///
    /// * Returns an error if the index is out of bounds for the state vector.
    pub fn s_dag(&self, index: usize) -> Result<Self, Error> {
        PhaseSdag {}.apply(self, &[index], &[])
    }

    /// Applies the Phase S dagger gate to the specified qubits in the state vector in the given order.
    ///
    /// # Arguments
    ///
    /// * `qubits` - The indices of the qubits to apply the Phase S dagger gate to.
    ///
    /// # Returns
    ///
    /// * `Result` - A result containing the new state object if successful, or an error if the operation fails.
    ///
    /// # Errors
    ///
    /// * Returns an error if the number of qubits provided is invalid.
    ///
    /// * Returns an error if the indices are out of bounds for the state vector.
    pub fn s_dag_multi(&self, qubits: &[usize]) -> Result<Self, Error> {
        let mut new_state: State = self.clone();
        let sdag_gate: PhaseSdag = PhaseSdag {};
        for &qubit in qubits {
            new_state = sdag_gate.apply(&new_state, &[qubit], &[])?;
        }
        Ok(new_state)
    }

    /// Applies the controlled Phase S dagger gate to the specified qubits in the state vector in the given order.
    ///
    /// # Arguments
    ///
    /// * `target_qubits` - The indices of the target qubits to apply the controlled Phase S dagger gate to.
    /// * `control_qubits` - The indices of the control qubits for the controlled Phase S dagger gate.
    ///
    /// # Returns
    ///
    /// * `Result` - A result containing the new state object if successful, or an error if the operation fails.
    ///
    /// # Errors
    ///
    /// * Returns an error if the number of qubits provided is invalid.
    /// * Returns an error if the indices are out of bounds for the state vector.
    /// * Returns an error if the control qubits and target qubits overlap.
    pub fn cs_dag_multi(
        &self,
        target_qubits: &[usize],
        control_qubits: &[usize],
    ) -> Result<Self, Error> {
        let mut new_state: State = self.clone();
        let sdag_gate: PhaseSdag = PhaseSdag {};
        for &qubit in target_qubits {
            new_state = sdag_gate.apply(&new_state, &[qubit], control_qubits)?;
        }
        Ok(new_state)
    }

    /// Applies the Phase T dagger gate to the specified qubit in the state vector.
    ///
    /// # Arguments
    ///
    /// * `index` - The index of the qubit to apply the Phase T dagger gate to.
    ///
    /// # Returns
    ///
    /// * `Result` - A result containing the new state object if successful, or an error if the operation fails.
    ///
    /// # Errors
    ///
    /// * Returns an error if the index is out of bounds for the state vector.
    pub fn t_dag(&self, index: usize) -> Result<Self, Error> {
        PhaseTdag {}.apply(self, &[index], &[])
    }

    /// Applies the Phase T dagger gate to the specified qubits in the state vector in the given order.
    ///
    /// # Arguments
    ///
    /// * `qubits` - The indices of the qubits to apply the Phase T dagger gate to.
    ///
    /// # Returns
    ///
    /// * `Result` - A result containing the new state object if successful, or an error if the operation fails.
    ///
    /// # Errors
    ///
    /// * Returns an error if the number of qubits provided is invalid.
    ///
    /// * Returns an error if the indices are out of bounds for the state vector.
    pub fn t_dag_multi(&self, qubits: &[usize]) -> Result<Self, Error> {
        let mut new_state: State = self.clone();
        let tdag_gate: PhaseTdag = PhaseTdag {};
        for &qubit in qubits {
            new_state = tdag_gate.apply(&new_state, &[qubit], &[])?;
        }
        Ok(new_state)
    }

    /// Applies the controlled Phase T dagger gate to the specified qubits in the state vector in the given order.
    ///
    /// # Arguments
    ///
    /// * `target_qubits` - The indices of the target qubits to apply the controlled Phase T dagger gate to.
    /// * `control_qubits` - The indices of the control qubits for the controlled Phase T dagger gate.
    ///
    /// # Returns
    ///
    /// * `Result` - A result containing the new state object if successful, or an error if the operation fails.
    ///
    /// # Errors
    ///
    /// * Returns an error if the number of qubits provided is invalid.
    /// * Returns an error if the indices are out of bounds for the state vector.
    /// * Returns an error if the control qubits and target qubits overlap.
    pub fn ct_dag_multi(
        &self,
        target_qubits: &[usize],
        control_qubits: &[usize],
    ) -> Result<Self, Error> {
        let mut new_state: State = self.clone();
        let tdag_gate: PhaseTdag = PhaseTdag {};
        for &qubit in target_qubits {
            new_state = tdag_gate.apply(&new_state, &[qubit], control_qubits)?;
        }
        Ok(new_state)
    }

    /// Applies the Phase Shift gate with the specified angle to the given qubit.
    ///
    /// # Arguments
    ///
    /// * `index` - The index of the qubit to apply the Phase Shift gate to.
    /// * `angle` - The phase shift angle in radians.
    ///
    /// # Returns
    ///
    /// * `Result` - A result containing the new state object if successful, or an error if the operation fails.
    ///
    /// # Errors
    ///
    /// * Returns an error if the index is out of bounds for the state vector.
    pub fn p(&self, index: usize, angle: f64) -> Result<Self, Error> {
        PhaseShift::new(angle).apply(self, &[index], &[])
    }

    /// Applies the Phase Shift gate with the specified angle to the given qubits in order.
    ///
    /// # Arguments
    ///
    /// * `qubits` - The indices of the qubits to apply the Phase Shift gate to.
    ///
    /// * `angle` - The phase shift angle in radians.
    ///
    /// # Returns
    ///
    /// * `Result` - A result containing the new state object if successful, or an error if the operation fails.
    ///
    /// # Errors
    ///
    /// * Returns an error if the number of qubits provided is invalid.
    ///
    /// * Returns an error if the indices are out of bounds for the state vector.
    pub fn p_multi(&self, qubits: &[usize], angle: f64) -> Result<Self, Error> {
        let mut new_state: State = self.clone();
        let phase_shift_gate: PhaseShift = PhaseShift::new(angle);
        for &qubit in qubits {
            new_state = phase_shift_gate.apply(&new_state, &[qubit], &[])?;
        }
        Ok(new_state)
    }

    /// Applies the controlled Phase Shift gate with the specified angle to the given qubits in order.
    ///
    /// # Arguments
    ///
    /// * `target_qubits` - The indices of the target qubits to apply the controlled Phase Shift gate to.
    /// * `control_qubits` - The indices of the control qubits for the controlled Phase Shift gate.
    /// * `angle` - The phase shift angle in radians.
    ///
    /// # Returns
    ///
    /// * `Result` - A result containing the new state object if successful, or an error if the operation fails.
    ///
    /// # Errors
    ///
    /// * Returns an error if the number of qubits provided is invalid.
    /// * Returns an error if the indices are out of bounds for the state vector.
    /// * Returns an error if the control qubits and target qubits overlap.
    pub fn cp_multi(
        &self,
        target_qubits: &[usize],
        control_qubits: &[usize],
        angle: f64,
    ) -> Result<Self, Error> {
        let mut new_state: State = self.clone();
        let phase_shift_gate: PhaseShift = PhaseShift::new(angle);
        for &qubit in target_qubits {
            new_state = phase_shift_gate.apply(&new_state, &[qubit], control_qubits)?;
        }
        Ok(new_state)
    }

    /// Applies the RotateX gate with the specified angle to the given qubit.
    ///
    /// # Arguments
    ///
    /// * `index` - The index of the qubit to apply the RotateX gate to.
    /// * `angle` - The rotation angle in radians.
    ///
    /// # Returns
    ///
    /// * `Result` - A result containing the new state object if successful, or an error if the operation fails.
    ///
    /// # Errors
    ///
    /// * Returns an error if the index is out of bounds for the state vector.
    pub fn rx(&self, index: usize, angle: f64) -> Result<Self, Error> {
        RotateX::new(angle).apply(self, &[index], &[])
    }

    /// Applies the RotateX gate with the specified angle to the given qubits in order.
    ///
    /// # Arguments
    ///
    /// * `qubits` - The indices of the qubits to apply the RotateX gate to.
    ///
    /// * `angle` - The rotation angle in radians.
    ///
    /// # Returns
    ///
    /// * `Result` - A result containing the new state object if successful, or an error if the operation fails.
    ///
    /// # Errors
    ///
    /// * Returns an error if the number of qubits provided is invalid.
    ///
    /// * Returns an error if the indices are out of bounds for the state vector.
    pub fn rx_multi(&self, qubits: &[usize], angle: f64) -> Result<Self, Error> {
        let mut new_state: State = self.clone();
        let rotate_x_gate: RotateX = RotateX::new(angle);
        for &qubit in qubits {
            new_state = rotate_x_gate.apply(&new_state, &[qubit], &[])?;
        }
        Ok(new_state)
    }

    /// Applies the controlled RotateX gate with the specified angle to the given qubits in order.
    ///
    /// # Arguments
    ///
    /// * `target_qubits` - The indices of the target qubits to apply the controlled RotateX gate to.
    /// * `control_qubits` - The indices of the control qubits for the controlled RotateX gate.
    /// * `angle` - The rotation angle in radians.
    ///
    /// # Returns
    ///
    /// * `Result` - A result containing the new state object if successful, or an error if the operation fails.
    ///
    /// # Errors
    ///
    /// * Returns an error if the number of qubits provided is invalid.
    /// * Returns an error if the indices are out of bounds for the state vector.
    /// * Returns an error if the control qubits and target qubits overlap.
    pub fn crx_multi(
        &self,
        target_qubits: &[usize],
        control_qubits: &[usize],
        angle: f64,
    ) -> Result<Self, Error> {
        let mut new_state: State = self.clone();
        let rotate_x_gate: RotateX = RotateX::new(angle);
        for &qubit in target_qubits {
            new_state = rotate_x_gate.apply(&new_state, &[qubit], control_qubits)?;
        }
        Ok(new_state)
    }

    /// Applies the RotateY gate with the specified angle to the given qubit.
    ///
    /// # Arguments
    ///
    /// * `index` - The index of the qubit to apply the RotateY gate to.
    /// * `angle` - The rotation angle in radians.
    ///
    /// # Returns
    ///
    /// * `Result` - A result containing the new state object if successful, or an error if the operation fails.
    ///
    /// # Errors
    ///
    /// * Returns an error if the index is out of bounds for the state vector.
    pub fn ry(&self, index: usize, angle: f64) -> Result<Self, Error> {
        RotateY::new(angle).apply(self, &[index], &[])
    }

    /// Applies the RotateY gate with the specified angle to the given qubits in order.
    ///
    /// # Arguments
    ///
    /// * `qubits` - The indices of the qubits to apply the RotateY gate to.
    ///
    /// * `angle` - The rotation angle in radians.
    ///
    /// # Returns
    ///
    /// * `Result` - A result containing the new state object if successful, or an error if the operation fails.
    ///
    /// # Errors
    ///
    /// * Returns an error if the number of qubits provided is invalid.
    ///
    /// * Returns an error if the indices are out of bounds for the state vector.
    pub fn ry_multi(&self, qubits: &[usize], angle: f64) -> Result<Self, Error> {
        let mut new_state: State = self.clone();
        let rotate_y_gate: RotateY = RotateY::new(angle);
        for &qubit in qubits {
            new_state = rotate_y_gate.apply(&new_state, &[qubit], &[])?;
        }
        Ok(new_state)
    }

    /// Applies the controlled RotateY gate with the specified angle to the given qubits in order.
    ///
    /// # Arguments
    ///
    /// * `target_qubits` - The indices of the target qubits to apply the controlled RotateY gate to.
    /// * `control_qubits` - The indices of the control qubits for the controlled RotateY gate.
    /// * `angle` - The rotation angle in radians.
    ///
    /// # Returns
    ///
    /// * `Result` - A result containing the new state object if successful, or an error if the operation fails.
    ///
    /// # Errors
    ///
    /// * Returns an error if the number of qubits provided is invalid.
    /// * Returns an error if the indices are out of bounds for the state vector.
    /// * Returns an error if the control qubits and target qubits overlap.
    pub fn cry_multi(
        &self,
        target_qubits: &[usize],
        control_qubits: &[usize],
        angle: f64,
    ) -> Result<Self, Error> {
        let mut new_state: State = self.clone();
        let rotate_y_gate: RotateY = RotateY::new(angle);
        for &qubit in target_qubits {
            new_state = rotate_y_gate.apply(&new_state, &[qubit], control_qubits)?;
        }
        Ok(new_state)
    }

    /// Applies the RotateZ gate with the specified angle to the given qubit.
    ///
    /// # Arguments
    ///
    /// * `index` - The index of the qubit to apply the RotateZ gate to.
    /// * `angle` - The rotation angle in radians.
    ///
    /// # Returns
    ///
    /// * `Result` - A result containing the new state object if successful, or an error if the operation fails.
    ///
    /// # Errors
    ///
    /// * Returns an error if the index is out of bounds for the state vector.
    pub fn rz(&self, index: usize, angle: f64) -> Result<Self, Error> {
        RotateZ::new(angle).apply(self, &[index], &[])
    }

    /// Applies the RotateZ gate with the specified angle to the given qubits in order.
    ///
    /// # Arguments
    ///
    /// * `qubits` - The indices of the qubits to apply the RotateZ gate to.
    ///
    /// * `angle` - The rotation angle in radians.
    ///
    /// # Returns
    ///
    /// * `Result` - A result containing the new state object if successful, or an error if the operation fails.
    ///
    /// # Errors
    ///
    /// * Returns an error if the number of qubits provided is invalid.
    ///
    /// * Returns an error if the indices are out of bounds for the state vector.
    pub fn rz_multi(&self, qubits: &[usize], angle: f64) -> Result<Self, Error> {
        let mut new_state: State = self.clone();
        let rotate_z_gate: RotateZ = RotateZ::new(angle);
        for &qubit in qubits {
            new_state = rotate_z_gate.apply(&new_state, &[qubit], &[])?;
        }
        Ok(new_state)
    }

    /// Applies the controlled RotateZ gate with the specified angle to the given qubits in order.
    ///
    /// # Arguments
    ///
    /// * `target_qubits` - The indices of the target qubits to apply the controlled RotateZ gate to.
    /// * `control_qubits` - The indices of the control qubits for the controlled RotateZ gate.
    /// * `angle` - The rotation angle in radians.
    ///
    /// # Returns
    ///
    /// * `Result` - A result containing the new state object if successful, or an error if the operation fails.
    ///
    /// # Errors
    ///
    /// * Returns an error if the number of qubits provided is invalid.
    /// * Returns an error if the indices are out of bounds for the state vector.
    /// * Returns an error if the control qubits and target qubits overlap.
    pub fn crz_multi(
        &self,
        target_qubits: &[usize],
        control_qubits: &[usize],
        angle: f64,
    ) -> Result<Self, Error> {
        let mut new_state: State = self.clone();
        let rotate_z_gate: RotateZ = RotateZ::new(angle);
        for &qubit in target_qubits {
            new_state = rotate_z_gate.apply(&new_state, &[qubit], control_qubits)?;
        }
        Ok(new_state)
    }

    /// Applies the unitary gate to the specified qubit in the state vector.
    ///
    /// # Arguments
    ///
    /// * `index` - The index of the qubit to apply the unitary gate to.
    ///
    /// * `unitary` - The unitary matrix to apply.
    ///
    /// # Returns
    ///
    /// * `Result` - A result containing the new state object if successful, or an error if the operation fails.
    ///
    /// # Errors
    ///
    /// * Returns an error if the index is out of bounds for the state vector.
    ///
    /// * Returns an error if the unitary matrix is not unitary.
    pub fn unitary(&self, index: usize, unitary: [[Complex<f64>; 2]; 2]) -> Result<Self, Error> {
        let mut new_state: State = self.clone();
        let unitary_gate: Unitary2 = Unitary2::new(unitary)?;
        new_state = unitary_gate.apply(&new_state, &[index], &[])?;
        Ok(new_state)
    }

    /// Applies the unitary gate to the specified qubits in the state vector in the given order.
    ///
    /// # Arguments
    ///
    /// * `qubits` - The indices of the qubits to apply the unitary gate to.
    ///
    /// * `unitary` - The unitary matrix to apply.
    ///
    /// # Returns
    ///
    /// * `Result` - A result containing the new state object if successful, or an error if the operation fails.
    ///
    /// # Errors
    ///
    /// * Returns an error if the number of qubits provided is invalid.
    ///
    /// * Returns an error if the indices are out of bounds for the state vector.
    ///
    /// * Returns an error if the unitary matrix is not unitary.
    pub fn unitary_multi(
        &self,
        qubits: &[usize],
        unitary: [[Complex<f64>; 2]; 2],
    ) -> Result<Self, Error> {
        let mut new_state: State = self.clone();
        let unitary_gate: Unitary2 = Unitary2::new(unitary)?;
        for &qubit in qubits {
            new_state = unitary_gate.apply(&new_state, &[qubit], &[])?;
        }
        Ok(new_state)
    }

    /// Applies the controlled unitary gate to the specified qubits in the state vector in the given order.
    ///
    /// # Arguments
    ///
    /// * `target_qubits` - The indices of the target qubits to apply the controlled unitary gate to.
    ///
    /// * `control_qubits` - The indices of the control qubits for the controlled unitary gate.
    ///
    /// * `unitary` - The unitary matrix to apply.
    ///
    /// # Returns
    ///
    /// * `Result` - A result containing the new state object if successful, or an error if the operation fails.
    ///
    /// # Errors
    ///
    /// * Returns an error if the number of qubits provided is invalid.
    ///
    /// * Returns an error if the indices are out of bounds for the state vector.
    ///
    /// * Returns an error if the control qubits and target qubits overlap.
    ///
    /// * Returns an error if the unitary matrix is not unitary.
    pub fn cunitary_multi(
        &self,
        target_qubits: &[usize],
        control_qubits: &[usize],
        unitary: [[Complex<f64>; 2]; 2],
    ) -> Result<Self, Error> {
        let mut new_state: State = self.clone();
        let unitary_gate: Unitary2 = Unitary2::new(unitary)?;
        for &qubit in target_qubits {
            new_state = unitary_gate.apply(&new_state, &[qubit], control_qubits)?;
        }
        Ok(new_state)
    }

    /// Applies the Unitary (constructed from rotation angle and phase shift) to the specified qubit in the state vector.
    ///
    /// # Arguments
    ///
    /// * `index` - The index of the qubit to apply the Unitary gate to.
    ///
    /// * `angle` - The rotation angle in radians.
    ///
    /// * `phase` - The phase shift angle in radians.
    ///
    /// # Returns
    ///
    /// * `Result` - A result containing the new state object if successful, or an error if the operation fails.
    ///
    /// # Errors
    ///
    /// * Returns an error if the index is out of bounds for the state vector.
    pub fn ry_phase(&self, index: usize, angle: f64, phase: f64) -> Result<Self, Error> {
        Unitary2::from_ry_phase(angle, phase).apply(self, &[index], &[])
    }

    /// Applies the Unitary (constructed from rotation angle and phase shift) to the specified qubits in the state vector in the given order.
    ///
    /// # Arguments
    ///
    /// * `qubits` - The indices of the qubits to apply the Unitary gate to.
    ///
    /// * `angle` - The rotation angle in radians.
    ///
    /// * `phase` - The phase shift angle in radians.
    ///
    /// # Returns
    ///
    /// * `Result` - A result containing the new state object if successful, or an error if the operation fails.
    ///
    /// # Errors
    ///
    /// * Returns an error if the number of qubits provided is invalid.
    ///
    /// * Returns an error if the indices are out of bounds for the state vector.
    pub fn ry_phase_multi(&self, qubits: &[usize], angle: f64, phase: f64) -> Result<Self, Error> {
        let mut new_state: State = self.clone();
        let unitary_gate: Unitary2 = Unitary2::from_ry_phase(angle, phase);
        for &qubit in qubits {
            new_state = unitary_gate.apply(&new_state, &[qubit], &[])?;
        }
        Ok(new_state)
    }

    /// Applies the controlled Unitary (constructed from rotation angle and phase shift) to the specified qubits in the state vector in the given order.
    ///
    /// # Arguments
    ///
    /// * `target_qubits` - The indices of the target qubits to apply the controlled Unitary gate to.
    ///
    /// * `control_qubits` - The indices of the control qubits for the controlled Unitary gate.
    ///
    /// * `angle` - The rotation angle in radians.
    ///
    /// * * `phase` - The phase shift angle in radians.
    ///
    /// # Returns
    ///
    /// * `Result` - A result containing the new state object if successful, or an error if the operation fails.
    ///
    /// # Errors
    ///
    /// * Returns an error if the number of qubits provided is invalid.
    ///
    /// * Returns an error if the indices are out of bounds for the state vector.
    ///
    /// * Returns an error if the control qubits and target qubits overlap.
    pub fn cry_phase_gates(
        &self,
        target_qubits: &[usize],
        control_qubits: &[usize],
        angle: f64,
        phase: f64,
    ) -> Result<Self, Error> {
        let mut new_state: State = self.clone();
        let unitary_gate: Unitary2 = Unitary2::from_ry_phase(angle, phase);
        for &qubit in target_qubits {
            new_state = unitary_gate.apply(&new_state, &[qubit], control_qubits)?;
        }
        Ok(new_state)
    }

    /// Applies the Unitary (constructed from rotation angle and phase shift) to the specified qubit in the state vector.
    /// This is the adjoint of the ry_phase operation.
    /// 
    /// # Arguments
    ///
    /// * `qubit` - The index of the qubit to apply the adjoint operation to.
    /// 
    /// * `angle` - The rotation angle in radians.
    /// 
    /// * `phase` - The phase shift angle in radians.
    /// 
    /// # Returns
    ///
    /// * `Result` - A result containing the new state object if successful, or an error if the operation fails.
    /// 
    /// # Errors
    /// 
    /// * Returns an error if the target qubit is out of bounds for the state vector
    pub fn ry_phase_dag(
        &self,
        qubit: usize,
        angle: f64,
        phase: f64,
    ) -> Result<Self, Error> {
        Unitary2::from_ry_phase_dagger(angle, phase).apply(self, &[qubit], &[])
    }

    /// Applies the Unitary (constructed from rotation angle and phase shift) to the specified qubits in the state vector in the given order.
    /// This is the adjoint of the ry_phase operation.
    /// 
    /// # Arguments
    /// 
    /// * `qubits` - The indices of the qubits to apply the adjoint ry_phase operation to.
    /// 
    /// * `angle` - The rotation angle in radians.
    /// 
    /// * `phase` - The phase shift angle in radians.
    /// 
    /// # Returns
    ///
    /// * `Result` - A result containing the new state object if successful, or an error if the operation fails.
    /// 
    /// # Errors
    ///
    /// * Returns an error if the number of qubits provided is invalid.
    /// 
    /// * Returns an error if the indices are out of bounds for the state vector.
    pub fn ry_phase_dag_multi(
        &self,
        qubits: &[usize],
        angle: f64,
        phase: f64,
    ) -> Result<Self, Error> {
        let mut new_state: State = self.clone();
        let unitary_gate: Unitary2 = Unitary2::from_ry_phase_dagger(angle, phase);
        for &qubit in qubits {
            new_state = unitary_gate.apply(&new_state, &[qubit], &[])?;
        }
        Ok(new_state)
    }

    /// Applies the controlled Unitary (constructed from rotation angle and phase shift) to the specified qubits in the state vector in the given order.
    /// This is the controlled adjoint of the ry_phase operation.
    /// 
    /// # Arguments
    ///
    /// * `target_qubits` - The index of the target qubit.
    /// 
    /// * `control_qubits` - The indices of the control qubits.
    ///
    /// * `angle` - The rotation angle in radians.
    /// 
    /// * `phase` - The phase shift angle in radians.
    /// 
    /// # Returns
    ///
    /// * `Result` - A result containing the new state object if successful, or an error if the operation fails.
    ///
    /// # Errors
    ///
    /// * Returns an error if the number of qubits provided is invalid.
    ///
    /// * Returns an error if the indices are out of bounds for the state vector.
    ///
    /// * Returns an error if the control qubits and target qubits overlap.
    pub fn cry_phase_dag_gates(
        &self,
        target_qubits: &[usize],
        control_qubits: &[usize],
        angle: f64,
        phase: f64,
    ) -> Result<Self, Error> {
        let mut new_state: State = self.clone();
        let unitary_gate: Unitary2 = Unitary2::from_ry_phase_dagger(angle, phase);
        for &qubit in target_qubits {
            new_state = unitary_gate.apply(&new_state, &[qubit], control_qubits)?;
        }
        Ok(new_state)
    }

    // -- MULTI-QUBIT GATES --

    /// Applies the CNOT (Controlled-NOT) gate to the state vector.
    ///
    /// # Arguments
    ///
    /// * `control` - The index of the control qubit.
    /// * `target` - The index of the target qubit.
    ///
    /// # Returns
    ///
    /// * `Result` - A result containing the new state object if successful, or an error if the operation fails.
    ///
    /// # Errors
    ///
    /// * Returns an error if any index is out of bounds for the state vector.
    pub fn cnot(&self, control: usize, target: usize) -> Result<Self, Error> {
        CNOT {}.apply(self, &[target], &[control])
    }

    /// Applies the SWAP gate to the state vector.
    ///
    /// # Arguments
    ///
    /// * `qubit1` - The index of the first qubit to swap.
    /// * `qubit2` - The index of the second qubit to swap.
    ///
    /// # Returns
    ///
    /// * `Result` - A result containing the new state object if successful, or an error if the operation fails.
    ///
    /// # Errors
    ///
    /// * Returns an error if any index is out of bounds for the state vector.
    pub fn swap(&self, qubit1: usize, qubit2: usize) -> Result<Self, Error> {
        SWAP {}.apply(self, &[qubit1, qubit2], &[])
    }

    /// Applies the controlled SWAP gate to the state vector.
    ///
    /// # Arguments
    ///
    /// * `target1` - The index of the first target qubit to swap.
    /// * `target2` - The index of the second target qubit to swap.
    /// * `controls` - The indices of the control qubits.
    ///
    /// # Returns
    ///
    /// * `Result` - A result containing the new state object if successful, or an error if the operation fails.
    ///
    /// # Errors
    ///
    /// * Returns an error if any index is out of bounds for the state vector.
    /// * Returns an error if target or control qubits overlap.
    pub fn cswap(&self, target1: usize, target2: usize, controls: &[usize]) -> Result<Self, Error> {
        SWAP {}.apply(self, &[target1, target2], controls)
    }

    /// Applies the Matchgate to the state vector
    ///
    /// # Arguments
    ///
    /// * `target` - The index of the first target qubit. The second target qubit is automatically determined to be the next qubit.
    /// * `theta` - The rotation angle in radians
    /// * `phi1` - The first phase angle in radians
    /// * `phi2` - The second phase angle in radians
    ///
    /// # Returns
    ///
    /// * `Result` - A result containing the new state object if successful, or an error if the operation fails.
    ///
    /// # Errors
    ///
    /// * Returns an error if any index is out of bounds for the state vector.
    /// * Returns an error if target or control qubits overlap.
    pub fn matchgate(
        &self,
        target: usize,
        theta: f64,
        phi1: f64,
        phi2: f64,
    ) -> Result<State, Error> {
        Matchgate { theta, phi1, phi2 }.apply(self, &[target], &[])
    }

    /// Applies the controlled Matchgate to the state vector
    ///
    /// # Arguments
    ///
    /// * `target` - The index of the first target qubit. The second target qubit is automatically determined to be the next qubit.
    /// * `theta` - The rotation angle in radians
    /// * `phi1` - The first phase angle in radians
    /// * `phi2` - The second phase angle in radians
    /// * `controls` - The indices of the control qubits
    ///
    /// # Returns
    ///
    /// * `Result` - A result containing the new state object if successful, or an error if the operation fails.
    ///
    /// # Errors
    ///
    /// * Returns an error if any index is out of bounds for the state vector.
    /// * Returns an error if target or control qubits overlap.
    pub fn cmatchgate(
        &self,
        target: usize,
        theta: f64,
        phi1: f64,
        phi2: f64,
        controls: &[usize],
    ) -> Result<State, Error> {
        Matchgate { theta, phi1, phi2 }.apply(self, &[target], controls)
    }

    /// Applies the Toffoli (Controlled-Controlled-NOT) gate to the state vector.
    ///
    /// # Arguments
    ///
    /// * `control1` - The index of the first control qubit.
    /// * `control2` - The index of the second control qubit.
    /// * `target` - The index of the target qubit.
    ///
    /// # Returns
    ///
    /// * `Result` - A result containing the new state object if successful, or an error if the operation fails.
    ///
    /// # Errors
    ///
    /// * Returns an error if any index is out of bounds for the state vector.
    pub fn toffoli(&self, control1: usize, control2: usize, target: usize) -> Result<Self, Error> {
        Toffoli {}.apply(self, &[target], &[control1, control2])
    }
}

impl PartialEq for State {
    fn eq(&self, other: &Self) -> bool {
        // Check if the number of qubits is the same
        if self.num_qubits != other.num_qubits {
            return false;
        }

        // Check if the state vectors have the same length (should be redundant with num_qubits check)
        if self.state_vector.len() != other.state_vector.len() {
            return false;
        }

        // Check if each element in the state vectors is approximately equal within epsilon
        for (a, b) in self.state_vector.iter().zip(other.state_vector.iter()) {
            let real_diff = (a.re - b.re).abs();
            let imag_diff = (a.im - b.im).abs();

            if real_diff > f32::EPSILON.into() || imag_diff > f32::EPSILON.into() {
                return false;
            }
        }

        true
    }
}

/// A trait to enable chainable operations on Result<State, Error>
pub trait ChainableState {
    // -- SINGLE QUBIT GATES --

    /// Applies the Hadamard gate to the specified qubit in the state vector.
    fn h(self, index: usize) -> Result<State, Error>;

    /// Applies the Hadamard gate to the specified qubits in the state vector in the given order.
    fn h_multi(self, qubits: &[usize]) -> Result<State, Error>;

    /// Applies the controlled Hadamard gate to the specified qubits in the state vector in the given order.
    fn ch_multi(self, target_qubits: &[usize], control_qubits: &[usize]) -> Result<State, Error>;

    /// Applies the Pauli-X (NOT) gate to the specified qubit in the state vector.
    fn x(self, index: usize) -> Result<State, Error>;

    /// Applies the Pauli-X (NOT) gate to the specified qubits in the state vector in the given order.
    fn x_multi(self, qubits: &[usize]) -> Result<State, Error>;

    /// Applies the controlled Pauli-X (NOT) gate to the specified qubits in the state vector in the given order.
    fn cx_multi(self, target_qubits: &[usize], control_qubits: &[usize]) -> Result<State, Error>;

    /// Applies the Pauli-Y gate to the specified qubit in the state vector.
    fn y(self, index: usize) -> Result<State, Error>;

    /// Applies the Pauli-Y gate to the specified qubits in the state vector in the given order.
    fn y_multi(self, qubits: &[usize]) -> Result<State, Error>;

    /// Applies the controlled Pauli-Y gate to the specified qubits in the state vector in the given order.
    fn cy_multi(self, target_qubits: &[usize], control_qubits: &[usize]) -> Result<State, Error>;

    /// Applies the Pauli-Z gate to the specified qubit in the state vector.
    fn z(self, index: usize) -> Result<State, Error>;

    /// Applies the Pauli-Z gate to the specified qubits in the state vector in the given order.
    fn z_multi(self, qubits: &[usize]) -> Result<State, Error>;

    /// Applies the controlled Pauli-Z gate to the specified qubits in the state vector in the given order.
    fn cz_multi(self, target_qubits: &[usize], control_qubits: &[usize]) -> Result<State, Error>;

    /// Applies the Identity gate to the state vector.
    fn i(self, qubit: usize) -> Result<State, Error>;

    /// Applies the Identity gate to the state vector for multiple qubits in the given order.
    fn i_multi(self, qubits: &[usize]) -> Result<State, Error>;

    /// Applies the controlled Identity gate to the state vector for multiple qubits in the given order.
    fn ci_multi(self, target_qubits: &[usize], control_qubits: &[usize]) -> Result<State, Error>;

    /// Applies the Phase S gate to the specified qubit in the state vector.
    fn s(self, index: usize) -> Result<State, Error>;
    /// Applies the Phase S gate to the specified qubits in the state vector in the given order.
    fn s_multi(self, qubits: &[usize]) -> Result<State, Error>;

    /// Applies the controlled Phase S gate to the specified qubits in the state vector in the given order.
    fn cs_multi(self, target_qubits: &[usize], control_qubits: &[usize]) -> Result<State, Error>;

    /// Applies the Phase T gate to the specified qubit in the state vector.
    fn t(self, index: usize) -> Result<State, Error>;
    /// Applies the Phase T gate to the specified qubits in the state vector in the given order.
    fn t_multi(self, qubits: &[usize]) -> Result<State, Error>;

    /// Applies the controlled Phase T gate to the specified qubits in the state vector in the given order.
    fn ct_multi(self, target_qubits: &[usize], control_qubits: &[usize]) -> Result<State, Error>;

    /// Applies the Phase S dagger gate to the specified qubit in the state vector.
    fn s_dag(self, index: usize) -> Result<State, Error>;
    /// Applies the Phase S dagger gate to the specified qubits in the state vector in the given order.
    fn s_dag_multi(self, qubits: &[usize]) -> Result<State, Error>;

    /// Applies the controlled Phase S dagger gate to the specified qubits in the state vector in the given order.
    fn cs_dag_multi(
        self,
        target_qubits: &[usize],
        control_qubits: &[usize],
    ) -> Result<State, Error>;

    /// Applies the Phase T dagger gate to the specified qubit in the state vector.
    fn t_dag(self, index: usize) -> Result<State, Error>;
    /// Applies the Phase T dagger gate to the specified qubits in the state vector in the given order.
    fn t_dag_multi(self, qubits: &[usize]) -> Result<State, Error>;

    /// Applies the controlled Phase T dagger gate to the specified qubits in the state vector in the given order.
    fn ct_dag_multi(
        self,
        target_qubits: &[usize],
        control_qubits: &[usize],
    ) -> Result<State, Error>;

    /// Applies the Phase Shift gate with the specified angle to the given qubit.
    fn p(self, index: usize, angle: f64) -> Result<State, Error>;
    /// Applies the Phase Shift gate with the specified angle to the given qubits in order.
    fn p_multi(self, qubits: &[usize], angle: f64) -> Result<State, Error>;

    /// Applies the controlled Phase Shift gate with the specified angle to the given qubits in order.
    fn cp_multi(
        self,
        target_qubits: &[usize],
        control_qubits: &[usize],
        angle: f64,
    ) -> Result<State, Error>;

    /// Applies the RotateX gate with the specified angle to the given qubit.
    fn rx(self, index: usize, angle: f64) -> Result<State, Error>;
    /// Applies the RotateX gate with the specified angle to the given qubits in order.
    fn rx_multi(self, qubits: &[usize], angle: f64) -> Result<State, Error>;

    /// Applies the controlled RotateX gate with the specified angle to the given qubits in order.
    fn crx_multi(
        self,
        target_qubits: &[usize],
        control_qubits: &[usize],
        angle: f64,
    ) -> Result<State, Error>;

    /// Applies the RotateY gate with the specified angle to the given qubit.
    fn ry(self, index: usize, angle: f64) -> Result<State, Error>;
    /// Applies the RotateY gate with the specified angle to the given qubits in order.
    fn ry_multi(self, qubits: &[usize], angle: f64) -> Result<State, Error>;

    /// Applies the controlled RotateY gate with the specified angle to the given qubits in order.
    fn cry_multi(
        self,
        target_qubits: &[usize],
        control_qubits: &[usize],
        angle: f64,
    ) -> Result<State, Error>;

    /// Applies the RotateZ gate with the specified angle to the given qubit.
    fn rz(self, index: usize, angle: f64) -> Result<State, Error>;
    /// Applies the RotateZ gate with the specified angle to the given qubits in order.
    fn rz_multi(self, qubits: &[usize], angle: f64) -> Result<State, Error>;

    /// Applies the controlled RotateZ gate with the specified angle to the given qubits in order.
    fn crz_multi(
        self,
        target_qubits: &[usize],
        control_qubits: &[usize],
        angle: f64,
    ) -> Result<State, Error>;

    /// Applies the unitary gate to the specified qubit in the state vector.
    fn unitary(self, index: usize, unitary: [[Complex<f64>; 2]; 2]) -> Result<State, Error>;

    /// Applies the unitary gate to the specified qubits in the state vector in the given order.
    fn unitary_multi(
        self,
        qubits: &[usize],
        unitary: [[Complex<f64>; 2]; 2],
    ) -> Result<State, Error>;

    /// Applies the controlled unitary gate to the specified qubits in the state vector in the given order.
    fn cunitary_multi(
        self,
        target_qubits: &[usize],
        control_qubits: &[usize],
        unitary: [[Complex<f64>; 2]; 2],
    ) -> Result<State, Error>;

    /// Applies the Unitary (constructed from rotation angle and phase shift) to the specified qubit in the state vector.
    fn ry_phase(self, index: usize, angle: f64, phase: f64) -> Result<State, Error>;
    /// Applies the Unitary (constructed from rotation angle and phase shift) to the specified qubits in the state vector in the given order.
    fn ry_phase_multi(self, qubits: &[usize], angle: f64, phase: f64) -> Result<State, Error>;
    /// Applies the controlled Unitary (constructed from rotation angle and phase shift) to the specified qubits in the state vector in the given order.
    fn cry_phase_gates(
        self,
        target_qubits: &[usize],
        control_qubits: &[usize],
        angle: f64,
        phase: f64,
    ) -> Result<State, Error>;
    /// Applies the adjoint of the Unitary (constructed from rotation angle and phase shift) to the specified qubit in the state vector.
    fn ry_phase_dag(self, index: usize, angle: f64, phase: f64) -> Result<State, Error>;
    /// Applies the adjoint of the Unitary (constructed from rotation angle and phase shift) to the specified qubits in the state vector in the given order.
    fn ry_phase_dag_multi(self, qubits: &[usize], angle: f64, phase: f64) -> Result<State, Error>;
    /// Applies the controlled adjoint of the Unitary (constructed from rotation angle and phase shift) to the specified qubits in the state vector in the given order.
    fn cry_phase_dag_gates(
        self,
        target_qubits: &[usize],
        control_qubits: &[usize],
        angle: f64,
        phase: f64,
    ) -> Result<State, Error>;

    // -- MULTI-QUBIT GATES --

    /// Applies the CNOT (Controlled-NOT) gate to the state vector.
    fn cnot(self, control: usize, target: usize) -> Result<State, Error>;

    /// Applies the SWAP gate to the state vector.
    fn swap(self, qubit1: usize, qubit2: usize) -> Result<State, Error>;

    /// Applies the controlled SWAP gate to the state vector.
    fn cswap(self, target1: usize, target2: usize, controls: &[usize]) -> Result<State, Error>;

    /// Applies the Toffoli (Controlled-Controlled-NOT) gate to the state vector.
    fn toffoli(self, control1: usize, control2: usize, target: usize) -> Result<State, Error>;

    /// Applies the Matchgate to the state vector.
    fn matchgate(self, target: usize, theta: f64, phi1: f64, phi2: f64) -> Result<State, Error>;

    /// Applies the controlled Matchgate to the state vector.
    fn cmatchgate(
        self,
        target: usize,
        theta: f64,
        phi1: f64,
        phi2: f64,
        controls: &[usize],
    ) -> Result<State, Error>;

    /// Applies a unitary operation to the state vector.
    fn operate(
        self,
        unitary: impl Operator,
        target_qubits: &[usize],
        control_qubits: &[usize],
    ) -> Result<State, Error>;

    // -- MEASUREMENT --

    /// Measures the state vector in the specified basis and returns the measurement result.
    fn measure(
        self,
        basis: MeasurementBasis,
        measured_qubits: &[usize],
    ) -> Result<MeasurementResult, Error>;

    /// Measures the state vector `n` times in the specified basis and returns the measurement results.
    fn measure_n(
        self,
        basis: MeasurementBasis,
        measured_qubits: &[usize],
        n: usize,
    ) -> Result<Vec<MeasurementResult>, Error>;
}

macro_rules! impl_chainable_state {
    ($($method:ident($($arg:ident: $arg_type:ty),*) -> $return_type:ty);* $(;)?) => {
        impl ChainableState for Result<State, Error> {
            $(
                fn $method(self, $($arg: $arg_type),*) -> $return_type {
                    self.and_then(|state| state.$method($($arg),*))
                }
            )*
        }
    };
}

impl_chainable_state! {
    // -- SINGLE QUBIT GATES --
    h(index: usize) -> Result<State, Error>;
    h_multi(qubits: &[usize]) -> Result<State, Error>;
    ch_multi(target_qubits: &[usize], control_qubits: &[usize]) -> Result<State, Error>;
    x(index: usize) -> Result<State, Error>;
    x_multi(qubits: &[usize]) -> Result<State, Error>;
    cx_multi(target_qubits: &[usize], control_qubits: &[usize]) -> Result<State, Error>;
    y(index: usize) -> Result<State, Error>;
    y_multi(qubits: &[usize]) -> Result<State, Error>;
    cy_multi(target_qubits: &[usize], control_qubits: &[usize]) -> Result<State, Error>;
    z(index: usize) -> Result<State, Error>;
    z_multi(qubits: &[usize]) -> Result<State, Error>;
    cz_multi(target_qubits: &[usize], control_qubits: &[usize]) -> Result<State, Error>;
    i(qubit: usize) -> Result<State, Error>;
    i_multi(qubits: &[usize]) -> Result<State, Error>;
    ci_multi(target_qubits: &[usize], control_qubits: &[usize]) -> Result<State, Error>;
    s(index: usize) -> Result<State, Error>;
    s_multi(qubits: &[usize]) -> Result<State, Error>;
    cs_multi(target_qubits: &[usize], control_qubits: &[usize]) -> Result<State, Error>;
    t(index: usize) -> Result<State, Error>;
    t_multi(qubits: &[usize]) -> Result<State, Error>;
    ct_multi(target_qubits: &[usize], control_qubits: &[usize]) -> Result<State, Error>;
    s_dag(index: usize) -> Result<State, Error>;
    s_dag_multi(qubits: &[usize]) -> Result<State, Error>;
    cs_dag_multi(target_qubits: &[usize], control_qubits: &[usize]) -> Result<State, Error>;
    t_dag(index: usize) -> Result<State, Error>;
    t_dag_multi(qubits: &[usize]) -> Result<State, Error>;
    ct_dag_multi(target_qubits: &[usize], control_qubits: &[usize]) -> Result<State, Error>;
    p(index: usize, angle: f64) -> Result<State, Error>;
    p_multi(qubits: &[usize], angle: f64) -> Result<State, Error>;
    cp_multi(target_qubits: &[usize], control_qubits: &[usize], angle: f64) -> Result<State, Error>;
    rx(index: usize, angle: f64) -> Result<State, Error>;
    rx_multi(qubits: &[usize], angle: f64) -> Result<State, Error>;
    crx_multi(target_qubits: &[usize], control_qubits: &[usize], angle: f64) -> Result<State, Error>;
    ry(index: usize, angle: f64) -> Result<State, Error>;
    ry_multi(qubits: &[usize], angle: f64) -> Result<State, Error>;
    cry_multi(target_qubits: &[usize], control_qubits: &[usize], angle: f64) -> Result<State, Error>;
    rz(index: usize, angle: f64) -> Result<State, Error>;
    rz_multi(qubits: &[usize], angle: f64) -> Result<State, Error>;
    crz_multi(target_qubits: &[usize], control_qubits: &[usize], angle: f64) -> Result<State, Error>;
    unitary(index: usize, unitary: [[Complex<f64>; 2]; 2]) -> Result<State, Error>;
    unitary_multi(qubits: &[usize], unitary: [[Complex<f64>; 2]; 2]) -> Result<State, Error>;
    cunitary_multi(target_qubits: &[usize], control_qubits: &[usize], unitary: [[Complex<f64>; 2]; 2]) -> Result<State, Error>;
    ry_phase(index: usize, angle: f64, phase: f64) -> Result<State, Error>;
    ry_phase_multi(qubits: &[usize], angle: f64, phase: f64) -> Result<State, Error>;
    cry_phase_gates(target_qubits: &[usize], control_qubits: &[usize], angle: f64, phase: f64) -> Result<State, Error>;
    ry_phase_dag(index: usize, angle: f64, phase: f64) -> Result<State, Error>;
    ry_phase_dag_multi(qubits: &[usize], angle: f64, phase: f64) -> Result<State, Error>;
    cry_phase_dag_gates(target_qubits: &[usize], control_qubits: &[usize], angle: f64, phase: f64) -> Result<State, Error>;

    // -- MULTI-QUBIT GATES --
    cnot(control: usize, target: usize) -> Result<State, Error>;
    swap(qubit1: usize, qubit2: usize) -> Result<State, Error>;
    cswap(target1: usize, target2: usize, controls: &[usize]) -> Result<State, Error>;
    toffoli(control1: usize, control2: usize, target: usize) -> Result<State, Error>;
    matchgate(target: usize, theta: f64, phi1: f64, phi2: f64) -> Result<State, Error>;
    cmatchgate(target: usize, theta: f64, phi1: f64, phi2: f64, controls: &[usize]) -> Result<State, Error>;
    operate(unitary: impl Operator, target_qubits: &[usize], control_qubits: &[usize]) -> Result<State, Error>;
    measure(basis: MeasurementBasis, measured_qubits: &[usize]) -> Result<MeasurementResult, Error>;
    measure_n(basis: MeasurementBasis, measured_qubits: &[usize], n: usize) -> Result<Vec<MeasurementResult>, Error>;
}

// Implement multiplication by Complex<f64>
impl Mul<Complex<f64>> for State {
    type Output = Self;

    /// Multiplies each amplitude in the state vector by a complex scalar.
    /// Note: This operation typically results in an unnormalised state.
    fn mul(self, rhs: Complex<f64>) -> Self::Output {
        let new_state_vector: Vec<Complex<f64>> = self
            .state_vector
            .into_par_iter() // Use parallel iterator for potential performance gain
            .map(|amplitude| amplitude * rhs)
            .collect();

        // Create a new State directly, bypassing the normalization check in `new`
        State {
            state_vector: new_state_vector,
            num_qubits: self.num_qubits,
        }
    }
}

// Implement multiplication by f64
impl Mul<f64> for State {
    type Output = Self;

    /// Multiplies each amplitude in the state vector by a real scalar.
    /// Note: This operation typically results in an unnormalised state.
    fn mul(self, rhs: f64) -> Self::Output {
        let complex_rhs = Complex::new(rhs, 0.0); // Convert f64 to Complex<f64>
        let new_state_vector: Vec<Complex<f64>> = self
            .state_vector
            .into_par_iter() // Use parallel iterator
            .map(|amplitude| amplitude * complex_rhs)
            .collect();

        // Create a new State directly
        State {
            state_vector: new_state_vector,
            num_qubits: self.num_qubits,
        }
    }
}

// Implement multiplication State = f64 * State
impl Mul<State> for f64 {
    type Output = State;

    /// Multiplies each amplitude in the state vector by a real scalar from the left.
    /// Note: This operation typically results in an unnormalised state.
    fn mul(self, rhs: State) -> Self::Output {
        let complex_lhs = Complex::new(self, 0.0); // Convert f64 to Complex<f64>
        let new_state_vector: Vec<Complex<f64>> = rhs
            .state_vector
            .into_par_iter() // Use parallel iterator
            .map(|amplitude| complex_lhs * amplitude)
            .collect();

        // Create a new State directly
        State {
            state_vector: new_state_vector,
            num_qubits: rhs.num_qubits,
        }
    }
}

// Implement multiplication State = Complex<f64> * State
impl Mul<State> for Complex<f64> {
    type Output = State;

    /// Multiplies each amplitude in the state vector by a complex scalar from the left.
    /// Note: This operation typically results in an unnormalised state.
    fn mul(self, rhs: State) -> Self::Output {
        let new_state_vector: Vec<Complex<f64>> = rhs
            .state_vector
            .into_par_iter() // Use parallel iterator
            .map(|amplitude| self * amplitude)
            .collect();

        // Create a new State directly
        State {
            state_vector: new_state_vector,
            num_qubits: rhs.num_qubits,
        }
    }
}

// Implement addition for State + State
impl Add<State> for State {
    type Output = Self;

    /// Adds two state vectors element-wise.
    /// Panics if the states do not have the same number of qubits.
    /// Note: This operation typically results in an unnormalised state.
    fn add(self, rhs: State) -> Self::Output {
        if self.num_qubits != rhs.num_qubits {
            panic!(
                "Cannot add states with different numbers of qubits: {} != {}",
                self.num_qubits, rhs.num_qubits
            );
        }

        let new_state_vector: Vec<Complex<f64>> = self
            .state_vector
            .into_par_iter()
            .zip(rhs.state_vector.into_par_iter())
            .map(|(a, b)| a + b)
            .collect();

        // Create a new State directly
        State {
            state_vector: new_state_vector,
            num_qubits: self.num_qubits,
        }
    }
}

// Implement sum for State
impl std::iter::Sum for State {
    fn sum<I>(mut iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        // Take the first state as the initial accumulator.
        let mut accumulator = match iter.next() {
            Some(first_state) => first_state,
            None => {
                panic!(
                    "Cannot sum an empty iterator of State objects: num_qubits for the sum is undefined."
                );
            }
        };

        // Fold the rest of the states into the accumulator.
        // The `Add` impl for `State` handles element-wise addition and
        // panics if `num_qubits` do not match.
        for state in iter {
            if accumulator.num_qubits != state.num_qubits {
                panic!(
                    "Cannot sum states with different numbers of qubits: {} != {}",
                    accumulator.num_qubits, state.num_qubits
                );
            }
            accumulator = accumulator + state; // Uses the implemented Add for State
        }
        accumulator
    }
}

// Implement subtraction for State - State
impl Sub<State> for State {
    type Output = Self;

    /// Subtracts the right-hand state vector from the left-hand state vector element-wise.
    /// Panics if the states do not have the same number of qubits.
    /// Note: This operation typically results in an unnormalised state.
    fn sub(self, rhs: State) -> Self::Output {
        if self.num_qubits != rhs.num_qubits {
            panic!(
                "Cannot subtract states with different numbers of qubits: {} != {}",
                self.num_qubits, rhs.num_qubits
            );
        }

        let new_state_vector: Vec<Complex<f64>> = self
            .state_vector
            .into_par_iter()
            .zip(rhs.state_vector.into_par_iter())
            .map(|(a, b)| a - b)
            .collect();

        // Create a new State directly
        State {
            state_vector: new_state_vector,
            num_qubits: self.num_qubits,
        }
    }
}

impl std::fmt::Debug for State {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut state_str = String::new();
        state_str.push_str(format!("State with {} qubits:\n[", self.num_qubits).as_str());
        for amplitude in &self.state_vector {
            let amplitude_string: String = if amplitude.im == 0.0 {
                format!("{:.2}", amplitude.re)
            } else if amplitude.re == 0.0 {
                format!("{:.2}i", amplitude.im)
            } else {
                format!("{:.2} + {:.2}i", amplitude.re, amplitude.im)
            };
            // Add the amplitude to the string representations
            state_str.push_str(format!("{}, ", amplitude_string).as_str());
        }
        state_str.pop(); // Remove the last comma
        state_str.pop(); // Remove the last space
        state_str.push(']');
        write!(f, "{}", state_str)
    }
}
