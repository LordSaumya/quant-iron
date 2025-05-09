use crate::{
    circuit::Circuit,
    components::gate::Gate,
    errors::Error,
    circuit::CircuitBuilder,
};

/// A subroutine for a quantum circuit.
///
/// # Fields
///
/// * `gates` - A vector of gates in the subroutine.
///
/// * `num_qubits` - The number of qubits in the subroutine.
pub struct Subroutine {
    /// The gates in the subroutine.
    pub gates: Vec<Gate>,
    /// The number of qubits in the subroutine.
    pub num_qubits: usize,
}

impl Subroutine {
    /// Creates a new subroutine with the specified number of qubits.
    ///
    /// # Arguments
    ///
    /// * `num_qubits` - The number of qubits in the subroutine.
    pub fn new(num_qubits: usize) -> Self {
        Subroutine {
            gates: Vec::new(),
            num_qubits,
        }
    }

    /// Creates a new subroutine with the specified gates and number of qubits.
    ///
    /// # Arguments
    ///
    /// * `gates` - A vector of gates in the subroutine.
    /// * `num_qubits` - The number of qubits in the subroutine.
    pub fn with_gates(gates: Vec<Gate>, num_qubits: usize) -> Subroutine {
        Subroutine { gates, num_qubits }
    }

    /// Gets the gates in the subroutine.
    ///
    /// # Returns
    ///
    /// * `&Vec<Gate>` - A reference to the vector of gates in the subroutine.
    pub fn get_gates(&self) -> &Vec<Gate> {
        &self.gates
    }

    /// Adds a gate to the subroutine.
    ///
    /// # Arguments
    ///
    /// * `gate` - The gate to be added to the subroutine.
    pub fn add_gate(&mut self, gate: Gate) {
        self.gates.push(gate);
    }

    /// Adds multiple gates to the subroutine.
    ///
    /// # Arguments
    ///
    /// * `gates` - A vector of gates to be added to the subroutine.
    pub fn add_gates(&mut self, gates: Vec<Gate>) {
        self.gates.extend(gates);
    }

    /// Gets the number of qubits in the subroutine.
    ///
    /// # Returns
    ///
    /// * `usize` - The number of qubits in the subroutine.
    pub fn get_num_qubits(&self) -> usize {
        self.num_qubits
    }

    // -- COMMON SUBROUTINES --

    /// Creates a quantum fourier transform subroutine for the specified qubits.
    ///
    /// # Arguments
    ///
    /// * `qubits` - A vector of indices of the qubits to be transformed.
    ///
    /// * `num_qubits` - The number of qubits in the subroutine.
    pub fn qft(qubits: Vec<usize>, num_qubits: usize) -> Subroutine {
        let mut builder: CircuitBuilder = CircuitBuilder::new(num_qubits);
        let n: usize = qubits.len();
        for i in 0..n {
            // Apply Hadamard gate
            builder.h_gate(qubits[i]);
            // Apply controlled phase rotations
            // (j-i) iterates from 1 up to n-1-i.
            // Let k_loop_val = j-i.
            let mut power_of_2_denominator = 2.0; // Initial value for k_loop_val = 1 (i.e., 2^1)
            for k_loop_val in 1..(n - i) { // k_loop_val goes from 1 to (n-i-1)
                let original_j_qubit_index = i + k_loop_val; // This is the original 'j'
                let angle = std::f64::consts::PI / power_of_2_denominator;
                builder.cp_gates(vec![qubits[i]], vec![qubits[original_j_qubit_index]], angle);
                power_of_2_denominator *= 2.0; // Update for the next k_loop_val
            }
        }
        // Swap qubits at the end
        for i in 0..(n / 2) {
            builder.swap_gate(qubits[i], qubits[n - 1 - i]);
        }
        builder.build_subroutine()
    }

    /// Creates a quantum inverse fourier transform subroutine for the specified qubits.
    /// 
    /// # Arguments
    /// 
    /// * `qubits` - A vector of indices of the qubits to be transformed.
    /// 
    /// * `num_qubits` - The number of qubits in the subroutine.
    /// 
    /// # Returns
    /// 
    /// * `Subroutine` - A new instance of the Subroutine struct.
    pub fn iqft(qubits: Vec<usize>, num_qubits: usize) -> Subroutine {
        let mut builder: CircuitBuilder = CircuitBuilder::new(num_qubits);
        let n: usize = qubits.len();

        // Apply swaps first
        for i in 0..(n / 2) {
            builder.swap_gate(qubits[i], qubits[n - 1 - i]);
        }

        // Apply inverse controlled rotations and Hadamards
        for i in (0..n).rev() {
            // Apply controlled phase rotations (inverse)
            // (j-i) iterates from (n-1-i) down to 1.
            // Let k_loop_val = j-i.
            if n > i + 1 { // Check if there are any rotations to apply for this 'i'
                let k_initial = (n - 1) - i; // Max value of (j-i)
                let mut power_of_2_denominator = 2_f64.powi(k_initial as i32);

                // Iterate for k_loop_val from k_initial down to 1
                for iteration_count in 0..k_initial {
                    let k_loop_val = k_initial - iteration_count; // Current k_loop_val (j-i)
                    let original_j_qubit_index = i + k_loop_val; // This is the original 'j'

                    let angle = -std::f64::consts::PI / power_of_2_denominator;
                    builder.cp_gates(vec![qubits[i]], vec![qubits[original_j_qubit_index]], angle);

                    if k_loop_val > 1 { // If not the last iteration (where k_loop_val would be 1)
                        power_of_2_denominator /= 2.0; // Update for the next (smaller) k_loop_val
                    }
                }
            }
            // Apply Hadamard gate
            builder.h_gate(qubits[i]);
        }
        builder.build_subroutine()
    }
}

// Allow conversion from Subroutine to Circuit
impl TryFrom<Subroutine> for Circuit {
    fn try_from(value: Subroutine) -> Result<Self, Self::Error> {
        let mut circuit = Circuit::new(value.num_qubits);
        for gate in value.gates {
            circuit.add_gate(gate)?;
        }
        Ok(circuit)
    }

    type Error = Error;
}
