use crate::{
    components::{operator::Operator, operator::Pauli, state::State, gate::Gate},
    errors::Error,
};
use num_complex::Complex;
use rayon::prelude::*;
use std::collections::HashMap;
use std::ops::{Add, Mul};

/// Represents a Pauli string, which is a product of Pauli operators (X, Y, Z) acting on qubits.
/// Used to represent a term in a Hamiltonian or a quantum operator.
#[derive(Debug, Clone, PartialEq)]
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

    /// Returns the length of the Pauli string, defined as the number of operators it contains.
    /// 
    /// # Returns
    /// 
    /// * `usize` - The number of operators in the Pauli string.
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    /// Creates a new Pauli string with the given coefficient and a set of operators.
    ///
    /// # Arguments
    /// * `coefficient` - The coefficient of the Pauli string, represented as a complex number.
    /// * `ops` - A mapping from qubit indices to Pauli operators.
    ///
    /// # Returns
    /// A new `PauliString` instance with the specified coefficient and operators.
    ///
    /// Note that the Hashmap ensures uniqueness of operators for each qubit.
    pub fn with_ops(coefficient: Complex<f64>, ops: HashMap<usize, Pauli>) -> Self {
        Self { ops, coefficient }
    }

    /// Adds a Pauli operator to the Pauli string at the specified qubit index.
    ///
    /// # Arguments
    /// * `qubit` - The index of the qubit to which the operator is applied.
    /// * `op` - The Pauli operator to be added (X, Y, or Z).
    /// 
    /// # Panics
    /// This function will panic if an operator for the same qubit index is added more than once.
    pub fn add_op(&mut self, qubit: usize, op: Pauli) {
        if self.ops.insert(qubit, op).is_some() {
            panic!("Duplicate Pauli string operator for qubit: {}", qubit);
        }
    }

    /// Adds a Pauli operator to the Pauli string at the specified qubit index and returns the new `PauliString` instance.
    ///
    /// # Arguments
    ///
    /// * `qubit` - The index of the qubit to which the operator is applied.
    /// * `op` - The Pauli operator to be added (X, Y, or Z).
    ///
    /// # Returns
    ///
    /// * `Self` - A new `PauliString` instance with the added operator.
    /// 
    /// # Panics
    /// This function will panic if an operator for the same qubit index is added more than once.
    pub fn with_op(mut self, qubit: usize, op: Pauli) -> Self {
        self.add_op(qubit, op);
        self
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

    /// Returns the list of targets of the Pauli string
    /// 
    /// # Returns
    /// * `Vec<usize>` - A vector of qubit indices that the Pauli string acts upon.
    pub fn get_targets(&self) -> Vec<usize> {
        let mut keys = self.ops.keys().cloned().collect::<Vec<usize>>();
        keys.sort();
        keys
    }

    /// Converts the Pauli string to a vector of operator gates.
    ///
    /// # Returns
    /// * `Vec<Gate>` - A vector of Gate structs representing the individual Pauli operators.
    pub fn to_gates(&self) -> Vec<Gate> {
        self.ops.iter().map(|(qubit, op)| {
            Gate::Operator(Box::new(*op), vec![*qubit], vec![])
        }).collect()
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

        // Apply the Pauli string to the state
        let mut new_state: State = state.clone();
        for (qubit, op) in &self.ops {
            new_state = op.apply(&new_state, &[*qubit], &[])?; // Assumes op.apply can take &mut State and modify it or returns a new one
        }
        Ok(new_state * self.coefficient)
    }

    /// Applies the Pauli string to a given state and normalises the new state.
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
    /// * Returns an error if the resulting state cannot be normalised (eg., has zero norm).
    pub fn apply_normalised(&self, state: &State) -> Result<State, Error> {
        let new_state: State = self.apply_operators(state)?;
        new_state.normalise()
    }

    /// Helper function to apply only the operator part of the Pauli string (P_ops) to a state.
    /// This does not include the PauliString's own coefficient.
    fn apply_operators(&self, state: &State) -> Result<State, Error> {
        if self.ops.is_empty() {
            // If there are no operators, P_ops is effectively Identity.
            // P_ops |state> = |state>.
            return Ok(state.clone());
        }

        let mut current_state = state.clone();
        for (qubit_idx, pauli_op) in &self.ops {
            current_state = pauli_op.apply(&current_state, &[*qubit_idx], &[])?;
        }
        Ok(current_state)
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

        // 1. Calculate P_ops |state> using the helper
        let p_ops_psi_state = self.apply_operators(state)?;

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

        // 1. Calculate P_ops |state> using the helper
        let p_ops_psi_state = self.apply_operators(state)?;
        // p_ops_psi_state now holds P_ops |state> (where P_ops is the product of Pauli operators without self.coefficient)

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

impl Add for PauliString {
    type Output = SumOp;

    fn add(self, other: Self) -> Self::Output {
        // Create a new SumOp with the two Pauli strings
        let terms: Vec<PauliString> = vec![self, other];
        // Return a new SumOp instance with the terms
        SumOp::new(terms)
    }
}

impl Mul<Complex<f64>> for PauliString {
    type Output = Self;

    fn mul(self, rhs: Complex<f64>) -> Self::Output {
        // Create a new Pauli string with the product of the coefficient and the given complex number
        PauliString {
            ops: self.ops.clone(),
            coefficient: self.coefficient * rhs,
        }
    }
}

impl Mul<f64> for PauliString {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self::Output {
        // Create a new Pauli string with the product of the coefficient and the given real number
        PauliString {
            ops: self.ops.clone(),
            coefficient: self.coefficient * Complex::new(rhs, 0.0),
        }
    }
}

impl Mul<PauliString> for f64 {
    type Output = PauliString;

    fn mul(self, rhs: PauliString) -> Self::Output {
        rhs * self
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

/// A vector of Pauli strings that are summed together.
/// Useful for representing Hamiltonians or observables.
///
/// # Fields
///
/// * `terms` - A vector of `PauliString` instances that are summed together.
#[derive(Debug, Clone, PartialEq)]
pub struct SumOp {
    /// A vector of Pauli strings that are summed together.
    pub terms: Vec<PauliString>,
}

impl SumOp {
    /// Creates a new `SumPauliString` instance with the given terms and number of qubits.
    ///
    /// # Arguments
    ///
    /// * `terms` - A vector of `PauliString` instances that are summed together.
    ///
    /// # Returns
    ///
    /// * A new `SumPauliString` instance with the specified terms and number of qubits.
    pub fn new(terms: Vec<PauliString>) -> Self {
        Self { terms }
    }

    /// Returns the number of terms in the sum.
    ///
    /// # Returns
    ///
    /// * `usize` - The number of terms in the sum.
    pub fn num_terms(&self) -> usize {
        self.terms.len()
    }

    /// Adds a new term to the sum.
    /// 
    /// # Arguments
    /// 
    /// * `term` - The `PauliString` term to be added to the sum.
    pub fn add_term(&mut self, term: PauliString) {
        self.terms.push(term);
    }

    /// Adds a new term to the sum and returns a new `SumOp` instance.
    /// 
    /// # Arguments
    /// 
    /// * `term` - The `PauliString` term to be added to the sum.
    /// 
    /// # Returns
    /// * `Self` - A new `SumOp` instance with the added term.
    pub fn with_term(mut self, term: PauliString) -> Self {
        self.add_term(term);
        self
    }

    /// Applies the sum of Pauli strings to a given state.
    ///
    /// # Arguments
    ///
    /// * `state` - The state to which the sum of Pauli strings is applied.
    ///
    /// # Returns
    ///
    /// * `Result<State, Error>` - The resulting state after applying the sum of Pauli strings, or an error if the operation fails.
    ///
    /// # Errors
    ///
    /// * Returns an error if the operations in the Pauli strings refer to qubits outside the range of the state.
    pub fn apply(&self, state: &State) -> Result<State, Error> {
        if self.terms.is_empty() {
            // An empty sumop is equivalent to the zero operator
            return Ok(state.clone() * 0.0);
        }

        Ok(self
            .terms
            .par_iter()
            .map(|term| term.apply(state))
            .collect::<Result<Vec<State>, Error>>()?
            .into_iter()
            .sum())
    }

    /// Calculates the expectation value <psi|H|psi> = Sum_i <psi|P_i|psi>.
    ///
    /// The expectation value is generally real for Hermitian operators and normalised states.
    /// However, this function returns a `Complex<f64>` as intermediate PauliStrings
    /// might have complex coefficients or the operator/state might not be strictly physical.
    ///
    /// # Arguments
    /// * `state` - The state |psi> for which to calculate the expectation value.
    ///             For a physically meaningful expectation value, this state should be normalised.
    ///
    /// # Returns
    /// * `Result<Complex<f64>, Error>` - The expectation value, or an error if the operation fails.
    ///
    /// # Errors
    /// * Returns an error if any underlying `PauliString::apply` fails (eg., invalid qubit index).
    /// * Returns an error if `state.inner_product` fails (eg., mismatched number of qubits,
    ///   though `PauliString::apply` should also catch qubit count issues).
    pub fn expectation_value(&self, state: &State) -> Result<Complex<f64>, Error> {
        if self.terms.is_empty() {
            // The expectation value of a zero operator is zero.
            return Ok(Complex::new(0.0, 0.0));
        }

        let expectation_values_per_term: Vec<Complex<f64>> = self
            .terms
            .par_iter()
            .map(|term| {
                // For each term P_i in the sum H = Sum_i P_i:
                // 1. Calculate |phi_i> = P_i |psi>
                let phi_i_state = term.apply(state)?;

                // 2. Calculate <psi | phi_i> = <psi | P_i | psi>
                state.inner_product(&phi_i_state)
            })
            .collect::<Result<Vec<Complex<f64>>, Error>>()?; // Collect into Result<Vec<_>, E>, propagating errors

        // Sum the individual expectation values <psi|P_i|psi>
        // Complex<f64> from num_complex implements std::iter::Sum.
        Ok(expectation_values_per_term.into_iter().sum())
    }
}

impl std::fmt::Display for SumOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut result: String = String::new();
        for (i, term) in self.terms.iter().enumerate() {
            if i > 0 {
                result.push_str(" + ");
            }
            result.push_str(&format!("{}", term));
        }
        write!(f, "{}", result)
    }
}

impl Mul<Complex<f64>> for SumOp {
    type Output = Self;

    fn mul(self, rhs: Complex<f64>) -> Self::Output {
        // Create a new SumOp with the coefficient multiplied by the given complex number
        let terms: Vec<PauliString> = self
            .terms
            .into_iter()
            .map(|term| term * rhs)
            .collect();
        // Return a new SumOp instance with the modified terms
        SumOp::new(terms)
    }
}

impl Mul<f64> for SumOp {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self::Output {
        // Create a new SumOp with the coefficient multiplied by the given real number
        let terms: Vec<PauliString> = self
            .terms
            .into_iter()
            .map(|term| term * rhs)
            .collect();
        // Return a new SumOp instance with the modified terms
        SumOp::new(terms)
    }
}

impl Add for SumOp {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        // Create a new SumOp with the two sets of terms
        let mut terms: Vec<PauliString> = self.terms;
        terms.extend(other.terms);
        // Return a new SumOp instance with the combined terms
        SumOp::new(terms)
    }
}

impl Add<PauliString> for SumOp {
    type Output = Self;

    fn add(self, other: PauliString) -> Self::Output {
        // Create a new SumOp with the existing terms and the new Pauli string
        let mut terms: Vec<PauliString> = self.terms;
        terms.push(other);
        // Return a new SumOp instance with the combined terms
        SumOp::new(terms)
    }
}
