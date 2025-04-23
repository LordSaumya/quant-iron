use crate::components::measurement::{MeasurementBasis, MeasurementResult};
use crate::errors::Error;
use nalgebra::{DMatrix, DVector};
use num_complex::Complex;
use rand::Rng;
use rayon::prelude::*;

#[derive(Debug, Clone, PartialEq)]
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
        // Check if the state vector is empty
        if state_vector.is_empty() {
            return Err(Error::InvalidNumberOfQubits(0));
        }
        // Check if the square norm (probability) of the state vector is 1
        let norm: f64 = state_vector.iter().map(|x| x.norm_sqr()).sum();
        if (norm - 1.0).abs() > f64::EPSILON {
            return Err(Error::StateVectorNotNormalised);
        }

        let num_qubits: usize = (state_vector.len() as f64).log(2.0).round() as usize;

        // Check if the number of qubits is valid (i.e., the length of the state vector is a power of 2)
        if (1 << num_qubits) != state_vector.len() {
            return Err(Error::InvalidNumberOfQubits(num_qubits as usize));
        }

        Ok(Self {
            state_vector,
            num_qubits,
        })
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
            return Err(Error::InvalidNumberOfQubits(num_qubits as usize));
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
            return Err(Error::InvalidNumberOfQubits(num_qubits as usize));
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
            return Err(Error::InvalidNumberOfQubits(num_qubits as usize));
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
            return Err(Error::InvalidNumberOfQubits(num_qubits as usize));
        }
        let dim: usize = 1 << num_qubits;
        let amplitude: Complex<f64> = Complex::new(1.0 / (dim as f64).sqrt(), 0.0);

        let mut state_vector: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); dim];

        for i in 0..dim {
            // Set +amplitude for even number of 1s, -amplitude for odd number of 1s
            let num_ones = i.count_ones() as usize;
            state_vector[i] = if num_ones % 2 == 0 {
                amplitude
            } else {
                -amplitude
            };
        }

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
            return Err(Error::InvalidNumberOfQubits(self.num_qubits as usize));
        }
        for &index in actual_measured_qubits {
            if index >= self.num_qubits {
                return Err(Error::InvalidQubitIndex(index, self.num_qubits));
            }
        }

        match basis {
            MeasurementBasis::Computational => {
                let num_outcomes: usize = 1 << num_measured;
                let num_unmeasured: usize = self.num_qubits - num_measured;

                let measured_mask: usize = actual_measured_qubits
                    .iter()
                    .fold(0, |mask, &q| mask | (1 << q));
                let unmeasured_qubit_indices: Vec<usize> = (0..self.num_qubits)
                    .filter(|&q| ((measured_mask >> q) & 1) == 0) // Collect indices where the bit is NOT set in measured_mask
                    .collect();

                // Calculate probabilities for each outcome (outcome as a single integer 0..num_outcomes-1)
                let probabilities: Vec<f64> = (0..num_outcomes)
                    .into_par_iter()
                    .map(|outcome_val| {
                        let mut amplitude_sum_sqr: f64 = 0.0;

                        let mut outcome_base_index = 0;
                        for (bit_index, &qubit_index) in actual_measured_qubits.iter().enumerate() {
                            if ((outcome_val >> bit_index) & 1) != 0 {
                                outcome_base_index |= 1 << qubit_index;
                            }
                        }

                        for unmeasured_comb_val in 0..(1 << num_unmeasured) {
                            let mut unmeasured_part_index = 0;
                            for (unmeasured_bit_index, &global_qubit_index) in
                                unmeasured_qubit_indices.iter().enumerate()
                            {
                                if ((unmeasured_comb_val >> unmeasured_bit_index) & 1) != 0 {
                                    unmeasured_part_index |= 1 << global_qubit_index;
                                }
                            }
                            // Combine the measured bits (in outcome_base_index) with the unmeasured bits (in unmeasured_part_index)
                            let basis_state_index: usize = outcome_base_index | unmeasured_part_index;

                            // Add the probability contribution of this matching basis state
                            amplitude_sum_sqr += self.state_vector[basis_state_index].norm_sqr();
                        }
                        amplitude_sum_sqr
                    })
                    .collect();

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

                // Collapse the state vector into a new vector
                let mut collapsed_state_data: Vec<Complex<f64>> =
                    vec![Complex::new(0.0, 0.0); self.state_vector.len()];
                let mut normalisation_sq: f64 = 0.0;

                // Calculate sampled_outcome_base_index (bits from sampled_outcome_int at measured_qubits positions)
                let mut sampled_outcome_base_index = 0;
                for (bit_index, &qubit_index) in actual_measured_qubits.iter().enumerate() {
                    if ((sampled_outcome_int >> bit_index) & 1) != 0 {
                        sampled_outcome_base_index |= 1 << qubit_index;
                    }
                }

                for unmeasured_comb_val in 0..(1 << num_unmeasured) {
                    let mut unmeasured_part_index = 0;
                    for (unmeasured_bit_index, &global_qubit_index) in
                        unmeasured_qubit_indices.iter().enumerate()
                    {
                        if ((unmeasured_comb_val >> unmeasured_bit_index) & 1) != 0 {
                            unmeasured_part_index |= 1 << global_qubit_index;
                        }
                    }
                    let basis_state_index: usize =
                        sampled_outcome_base_index | unmeasured_part_index;

                    // Copy the amplitude from the original state to the collapsed state
                    collapsed_state_data[basis_state_index] = self.state_vector[basis_state_index];
                    // Update normalisation based on the amplitude from the ORIGINAL state
                    normalisation_sq += self.state_vector[basis_state_index].norm_sqr();
                }

                // Renormalise the new collapsed state vector
                if normalisation_sq > f64::EPSILON {
                    let norm_factor: f64 = normalisation_sq.sqrt();
                    for amplitude in collapsed_state_data.iter_mut() {
                        *amplitude /= norm_factor;
                    }
                }

                // Convert the sampled integer outcome to a Vec<u8>
                let mut outcome_binary_vec: Vec<u8> = vec![0; num_measured];
                for i in 0..num_measured {
                    outcome_binary_vec[i] = ((sampled_outcome_int >> i) & 1) as u8;
                }

                // Create the measurement result
                Ok(MeasurementResult {
                    basis,
                    indices: actual_measured_qubits.to_vec(),
                    outcomes: outcome_binary_vec,
                    new_state: State::new(collapsed_state_data)?,
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
            return Err(Error::InvalidNumberOfQubits(self.num_qubits as usize));
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
}