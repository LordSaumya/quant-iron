use crate::{
    components::{operator::Operator, operator::Pauli, state::State},
    errors::Error,
};
use num_complex::Complex;
use std::collections::HashMap;

/// Represents a Pauli string, which is a product of Pauli operators (X, Y, Z) acting on qubits.
/// Used to represent a term in a Hamiltonian or a quantum operator.
#[derive(Debug, Clone)]
pub struct PauliString {
    /// A mapping from qubit indices to Pauli operators.
    ops: HashMap<usize, Pauli>,
    /// The coefficient of the Pauli string, which is a complex number.
    coefficient: Complex<f64>,
}

impl PauliString {
    /// Creates a new Pauli string with the given coefficient and an empty set of operators.
    ///
    /// # Arguments
    /// * `coefficient` - The coefficient of the Pauli string, represented as a complex number.
    ///
    /// # Returns
    /// A new `PauliString` instance with the specified coefficient and no operators.
    pub fn new(coefficient: Complex<f64>) -> Self {
        Self {
            ops: HashMap::new(),
            coefficient,
        }
    }

    /// Creates a new Pauli string with the given coefficient and a set of operators.
    ///
    /// # Arguments
    /// * `coefficient` - The coefficient of the Pauli string, represented as a complex number.
    /// * `ops` - A mapping from qubit indices to Pauli operators.
    ///
    /// # Returns
    /// A new `PauliString` instance with the specified coefficient and operators.
    pub fn with_ops(coefficient: Complex<f64>, ops: HashMap<usize, Pauli>) -> Self {
        Self { ops, coefficient }
    }

    /// Adds a Pauli operator to the Pauli string at the specified qubit index.
    ///
    /// # Arguments
    /// * `qubit` - The index of the qubit to which the operator is applied.
    /// * `op` - The Pauli operator to be added (X, Y, or Z).
    pub fn add_op(&mut self, qubit: usize, op: Pauli) {
        self.ops.insert(qubit, op);
    }

    /// Returns the coefficient of the Pauli string.
    ///
    /// # Returns
    ///
    /// * `Complex<f64>` - The coefficient of the Pauli string, represented as a complex number.
    pub fn coefficient(&self) -> Complex<f64> {
        self.coefficient
    }

    /// Returns a reference to the operators in the Pauli string.
    ///
    /// # Returns
    ///
    /// * `&HashMap<usize, Pauli>` - A reference to the mapping of qubit indices to Pauli operators.
    pub fn ops(&self) -> &HashMap<usize, Pauli> {
        &self.ops
    }

    /// Applies the Pauli string to a given state.
    ///
    /// # Arguments
    /// * `state` - The state to which the Pauli string is applied.
    ///
    /// # Returns
    /// * `Result<State, Error>` - The resulting state after applying the Pauli string, or an error if the operation fails.
    ///
    /// # Errors
    ///
    /// * Returns an error if the operations in the Pauli string refer to qubits outside the range of the state.
    pub fn apply(&self, state: &State) -> Result<State, Error> {
        // If the Pauli string is empty, return the state multiplied by the coefficient
        if self.ops.is_empty() {
            return Ok(state.clone() * self.coefficient);
        }

        // Check if the operations refer to qubits outside the range of the state
        for qubit in self.ops.keys() {
            if *qubit >= state.num_qubits() {
                return Err(Error::InvalidQubitIndex(*qubit, state.num_qubits()));
            }
        }

        // Apply the Pauli string to the state
        let mut new_state: State = state.clone();
        for (qubit, op) in &self.ops {
            new_state = op.apply(&mut new_state, &[*qubit], &[])?;
        }
        Ok(new_state * self.coefficient)
    }

    /// Applies the exponential of the Pauli string to a given state.
    ///
    /// # Arguments
    ///
    /// * `state` - The state to which the exponential of the Pauli string is applied.
    ///
    /// # Returns
    ///
    /// * `Result<State, Error>` - The resulting state after applying the exponential of the Pauli string, or an error if the operation fails.
    ///
    /// # Errors
    ///
    /// * Returns an error if the operations in the Pauli string refer to qubits outside the range of the state.
    pub fn apply_exp(&self, state: &State) -> Result<State, Error> {
        // If the Pauli string is empty, return the state multiplied by the exponential of the coefficient
        let alpha: Complex<f64> = self.coefficient;

        if self.ops.is_empty() {
            // P_ops is Identity. exp(alpha * I) |state> = exp(alpha) * |state>
            return Ok(state.clone() * alpha.exp());
        }

        // Check if the operations refer to qubits outside the range of the state
        for qubit_idx in self.ops.keys() {
            if *qubit_idx >= state.num_qubits() {
                return Err(Error::InvalidQubitIndex(*qubit_idx, state.num_qubits()));
            }
        }

        // 1. Calculate P_ops |state>
        let mut p_ops_psi_state: State = state.clone();
        for (qubit_idx, pauli_op) in &self.ops {
            p_ops_psi_state = pauli_op.apply(&mut p_ops_psi_state, &[*qubit_idx], &[])?;
        }
        // 2. Calculate scalar coefficients for exp(alpha * P_ops) = cosh(alpha)*I + sinh(alpha)*P_ops
        let cosh_alpha: Complex<f64> = alpha.cosh();
        let sinh_alpha: Complex<f64> = alpha.sinh();

        // 3. Calculate cosh(alpha) * |state>
        let term_identity_part: State = state.clone() * cosh_alpha;

        // 4. Calculate sinh(alpha) * (P_ops |state>)
        let term_operator_part: State = p_ops_psi_state * sinh_alpha;

        // 5. Add the two terms: cosh(alpha) * |state> + sinh(alpha) * (P_ops |state>)
        Ok(term_identity_part + term_operator_part)
    }

    /// Applies the exponential of the Pauli string to a given state with a specified factor.
    ///
    /// # Arguments
    /// * `state` - The state to which the exponential of the Pauli string is applied.
    /// * `factor` - A complex factor to be multiplied with the coefficient of the Pauli string.
    ///
    /// # Returns
    /// * `Result<State, Error>` - The resulting state after applying the exponential of the Pauli string with the factor, or an error if the operation fails.
    ///
    /// # Errors
    ///
    /// * Returns an error if the operations in the Pauli string refer to qubits outside the range of the state.
    pub fn apply_exp_factor(&self, state: &State, factor: Complex<f64>) -> Result<State, Error> {
        // Calculate the effective coefficient for the exponentiation
        let alpha: Complex<f64> = self.coefficient * factor;

        if self.ops.is_empty() {
            // P_ops is Identity. exp(alpha * I) |state> = exp(alpha) * |state>
            return Ok(state.clone() * alpha.exp());
        }

        // Check if the operations refer to qubits outside the range of the state
        for qubit_idx in self.ops.keys() { // Changed 'qubit' to 'qubit_idx' for clarity
            if *qubit_idx >= state.num_qubits() {
                return Err(Error::InvalidQubitIndex(*qubit_idx, state.num_qubits()));
            }
        }

        // 1. Calculate P_ops |state> (operator part only, without self.coefficient)
        let mut p_ops_psi_state: State = state.clone();
        for (qubit_idx, pauli_op) in &self.ops {
            p_ops_psi_state = pauli_op.apply(&mut p_ops_psi_state, &[*qubit_idx], &[])?;
        }
        // p_ops_psi_state now holds P_ops |state>

        // 2. Calculate scalar coefficients for exp(alpha * P_ops) = cosh(alpha)*I + sinh(alpha)*P_ops
        let cosh_alpha: Complex<f64> = alpha.cosh();
        let sinh_alpha: Complex<f64> = alpha.sinh();

        // 3. Calculate cosh(alpha) * |state>
        let term_identity_part: State = state.clone() * cosh_alpha;

        // 4. Calculate sinh(alpha) * (P_ops |state>)
        let term_operator_part: State = p_ops_psi_state * sinh_alpha;

        // 5. Add the two terms: cosh(alpha) * |state> + sinh(alpha) * (P_ops |state>)
        Ok(term_identity_part + term_operator_part)
    }

    /// Returns the Hermitian conjugate of the Pauli string.
    ///
    /// # Returns
    ///
    /// * `Self` - A new `PauliString` instance representing the Hermitian conjugate of the original Pauli string.
    pub fn hermitian_conjugate(&self) -> Self {
        PauliString {
            ops: self.ops.clone(),
            coefficient: self.coefficient.conj(),
        }
    }
}

impl std::fmt::Display for PauliString {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let coeff_str: String = if self.coefficient.re == 0.0 && self.coefficient.im == 0.0 {
            "0".to_string()
        } else if self.coefficient.im == 0.0 {
            format!("{}", self.coefficient.re)
        } else if self.coefficient.re == 0.0 {
            format!("{}i", self.coefficient.im)
        } else {
            format!("({} + {}i)", self.coefficient.re, self.coefficient.im)
        };

        let mut result: String = coeff_str + " * ";

        let mut sorted_ops: Vec<(&usize, &Pauli)> = self.ops.iter().collect();
        sorted_ops.sort_by(|&(qubit_a, op_a), &(qubit_b, op_b)| {
            if qubit_a == qubit_b {
                // If qubits are the same, sort by operator type (X -> Y -> Z)
                let op_a_str: String = format!("{}", op_a);
                let op_b_str: String = format!("{}", op_b);
                match (op_a_str.as_str(), op_b_str.as_str()) {
                    ("X", "Y") | ("X", "Z") | ("Y", "Z") => std::cmp::Ordering::Less,
                    ("Y", "X") | ("Z", "X") | ("Z", "Y") => std::cmp::Ordering::Greater,
                    _ => std::cmp::Ordering::Equal,
                }
            } else {
                // Otherwise, sort by qubit index
                qubit_a.cmp(qubit_b)
            }
        });
        for (qubit, op) in sorted_ops {
            result.push_str(&format!("{}[{}] ", op, qubit));
        }

        write!(f, "{}", result.trim())
    }
}
