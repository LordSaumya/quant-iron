use crate::{
    components::{
        operator::Pauli,
        pauli_string::{PauliString, SumOp},
    },
    errors::Error,
};

use num_complex::Complex;
use rayon::prelude::*;

/// Generates the Hamiltonian for a 1D Ising model with periodic boundary conditions
///
/// # Arguments
///
/// * `h` - Vector representing the field strength for each spin. Must be of length `N`.
/// * `j` - Vector representing the coupling strength between spins. Must be of length `N`. Each element `j[i]` represents the coupling strength between spin `i` and spin `i+1`.
/// * `mu` - The magnetic moment
///
/// # Parameters
///
/// * `N` - The number of spins in the system
///
/// # Returns
///
/// * `Result<SumOp, Error>` - A result containing the Hamiltonian as a `SumOp` or an error
pub fn ising_1d<const N: usize>(h: [f64; N], j: [f64; N], mu: f64) -> Result<SumOp, Error> {
    if N < 2 {
        return Err(Error::InvalidNumberOfInputs(N, 2));
    }

    // Check if all coefficients are zero
    let all_zero = h.iter().all(|&val| val == 0.0) && j.iter().all(|&val| val == 0.0);
    if all_zero {
        return Ok(SumOp::new(Vec::new()));
    }

    // Use chunk-based parallelism for better load balancing
    let chunk_size = std::cmp::max(1, N / rayon::current_num_threads());
    
    // Parallelise the computation using Rayon with chunks
    let terms: Vec<PauliString> = (0..N)
        .into_par_iter()
        .chunks(chunk_size)
        .flat_map(|chunk| {
            let mut local_terms = Vec::new();

            for i in chunk {
                // - \sum J_{i} Z_i Z_{i+1}
                let j_val = j[i];
                if j_val != 0.0 {
                    let coeff_1: Complex<f64> = Complex::new(j_val, 0.0) * -1.0;
                    let mut pauli_1: PauliString = PauliString::new(coeff_1);
                    pauli_1.add_op(i, Pauli::Z);
                    pauli_1.add_op((i + 1) % N, Pauli::Z);
                    local_terms.push(pauli_1);
                }

                // - \sum_i m h_i Z_i
                let h_val = h[i];
                if h_val != 0.0 {
                    let coeff_2: Complex<f64> = -1.0 * mu * Complex::new(h_val, 0.0);
                    let mut pauli_2: PauliString = PauliString::new(coeff_2);
                    pauli_2.add_op(i, Pauli::Z);
                    local_terms.push(pauli_2);
                }
            }

            local_terms
        })
        .collect();

    Ok(SumOp::new(terms))
}

/// Generates the Hamiltonian for a 1D Ising model with periodic boundary conditions and uniform coupling and field strengths for all spins
///
/// # Arguments
///
/// * `n` - The number of spins in the system
/// * `h` - The field strength for each spin
/// * `j` - The coupling strength between spins
/// * `mu` - The magnetic moment
///
/// # Parameters
///
/// # Returns
///
/// * `Result<SumOp, Error>` - A result containing the Hamiltonian as a `SumOp` or an error
pub fn ising_1d_uniform(n: usize, h: f64, j: f64, mu: f64) -> Result<SumOp, Error> {
    if n < 2 {
        return Err(Error::InvalidNumberOfInputs(n, 2));
    }

    // Early exit if both coefficients are zero
    if h == 0.0 && j == 0.0 {
        return Ok(SumOp::new(Vec::new()));
    }

    // Pre-compute coefficients
    let coeff_1: Complex<f64> = Complex::new(j as f64, 0.0) * -1.0;
    let coeff_2: Complex<f64> = -1.0 * mu * Complex::new(h, 0.0);
    let include_j: bool = j != 0.0;
    let include_h: bool = h != 0.0;

    // Use chunk-based parallelism for better load balancing
    let chunk_size: usize = std::cmp::max(1, n / rayon::current_num_threads());
    
    // Parallelise using chunks
    let terms: Vec<PauliString> = (0..n)
        .into_par_iter()
        .chunks(chunk_size)
        .flat_map(|chunk| {
            let mut local_terms = Vec::new();

            for i in chunk {
                // - \sum J_{i} Z_i Z_{i+1}
                if include_j {
                    let mut pauli_1: PauliString = PauliString::new(coeff_1);
                    pauli_1.add_op(i, Pauli::Z);
                    pauli_1.add_op((i + 1) % n, Pauli::Z);
                    local_terms.push(pauli_1);
                }

                // - \sum_i m h_i Z_i
                if include_h {
                    let mut pauli_2: PauliString = PauliString::new(coeff_2);
                    pauli_2.add_op(i, Pauli::Z);
                    local_terms.push(pauli_2);
                }
            }

            local_terms
        })
        .collect();

    Ok(SumOp::new(terms))
}

/// Generates the Hamiltonian for a 2D Ising model with periodic boundary conditions.
///
/// # Arguments
///
/// * `h_param` - Matrix representing the field strength for each spin. `h_param[row][col]` is the field at site (row, col).
///               Dimensions: `N x M`.
/// * `j_param` - Matrix representing the coupling strengths for each site.
///               For site `(r,c)`:
///               `j_param[r][c][0]` is the vertical coupling J_vertical with site `((r+1)%N, c)`.
///               `j_param[r][c][1]` is the horizontal coupling J_horizontal with site `(r, (c+1)%M)`.
///               Dimensions: `N x M x 2`.
/// * `mu` - The magnetic moment.
///
/// # Parameters
///
/// * `N` - The number of rows of spins in the system.
/// * `M` - The number of columns of spins in the system.
///
/// # Returns
///
/// * `Result<SumOp, Error>` - A result containing the Hamiltonian as a `SumOp` or an error.
pub fn ising_2d<const N: usize, const M: usize>(
    h_param: [[f64; M]; N],
    j_param: [[[f64; 2]; M]; N],
    mu: f64,
) -> Result<SumOp, Error> {
    if N < 2 {
        return Err(Error::InvalidNumberOfInputs(N, 2));
    } else if M < 2 {
        return Err(Error::InvalidNumberOfInputs(M, 2));
    }

    // Check if all coefficients are zero for early exit
    let mut all_zero = true;
    'outer: for r in 0..N {
        for c in 0..M {
            if h_param[r][c] != 0.0 || j_param[r][c][0] != 0.0 || j_param[r][c][1] != 0.0 {
                all_zero = false;
                break 'outer;
            }
        }
    }
    
    if all_zero {
        return Ok(SumOp::new(Vec::new()));
    }

    // Use chunk-based parallelism for better load balancing
    let chunk_size: usize = std::cmp::max(1, (N * M) / rayon::current_num_threads());
    
    // Flatten the 2D grid into 1D for more efficient chunking
    let terms: Vec<PauliString> = (0..N*M)
        .into_par_iter()
        .chunks(chunk_size)
        .flat_map(|chunk| {
            let mut local_terms = Vec::new();
            
            for idx in chunk {
                let r: usize = idx / M;
                let c: usize = idx % M;
                let current_site_1d_idx: usize = idx;

                // Field term: - mu * h_param[r][c] * Z_{r,c}
                let field_coeff_val: f64 = h_param[r][c];
                if field_coeff_val != 0.0 {
                    let field_coeff: Complex<f64> = -1.0 * mu * Complex::new(field_coeff_val, 0.0);
                    let mut field_pauli_string: PauliString = PauliString::new(field_coeff);
                    field_pauli_string.add_op(current_site_1d_idx, Pauli::Z);
                    local_terms.push(field_pauli_string);
                }

                // Vertical coupling term
                let vertical_coupling_val: f64 = j_param[r][c][0];
                if vertical_coupling_val != 0.0 {
                    let vertical_coeff: Complex<f64> = Complex::new(vertical_coupling_val, 0.0) * -1.0;
                    let mut vertical_pauli_string: PauliString = PauliString::new(vertical_coeff);
                    vertical_pauli_string.add_op(current_site_1d_idx, Pauli::Z);
                    let next_row_site_1d_idx: usize = ((r + 1) % N) * M + c;
                    vertical_pauli_string.add_op(next_row_site_1d_idx, Pauli::Z);
                    local_terms.push(vertical_pauli_string);
                }

                // Horizontal coupling term
                let horizontal_coupling_val: f64 = j_param[r][c][1];
                if horizontal_coupling_val != 0.0 {
                    let horizontal_coeff: Complex<f64> = Complex::new(horizontal_coupling_val, 0.0) * -1.0;
                    let mut horizontal_pauli_string: PauliString = PauliString::new(horizontal_coeff);
                    horizontal_pauli_string.add_op(current_site_1d_idx, Pauli::Z);
                    let next_col_site_1d_idx: usize = r * M + ((c + 1) % M);
                    horizontal_pauli_string.add_op(next_col_site_1d_idx, Pauli::Z);
                    local_terms.push(horizontal_pauli_string);
                }
            }
            
            local_terms
        })
        .collect();

    Ok(SumOp::new(terms))
}

/// Generates the Hamiltonian for a 2D Ising model with periodic boundary conditions and uniform coupling and field strengths for all spins.
///
/// # Arguments
///
/// * `n` - The number of rows of spins in the system.
/// * `m` - The number of columns of spins in the system.
/// * `h` - The field strength for each spin.
/// * `j` - The coupling strength between spins.
/// * `mu` - The magnetic moment.
///
/// # Parameters
///
/// * `N` - The number of rows of spins in the system.
/// * `M` - The number of columns of spins in the system.
///
/// # Returns
///
/// * `Result<SumOp, Error>` - A result containing the Hamiltonian as a `SumOp` or an error.
pub fn ising_2d_uniform(n: usize, m: usize, h: f64, j: f64, mu: f64) -> Result<SumOp, Error> {
    if n < 2 {
        return Err(Error::InvalidNumberOfInputs(n, 2));
    } else if m < 2 {
        return Err(Error::InvalidNumberOfInputs(m, 2));
    }

    // Skip all calculations if both h and j are zero
    if h == 0.0 && j == 0.0 {
        return Ok(SumOp::new(Vec::new()));
    }

    // Pre-compute coefficients
    let field_coeff: Complex<f64> = -1.0 * mu * Complex::new(h, 0.0);
    let coupling_coeff: Complex<f64> = Complex::new(j, 0.0) * -1.0;

    // Skip coupling calculations entirely if j is zero
    let include_coupling: bool = j != 0.0;
    let include_field: bool = h != 0.0;
    
    // Use chunk-based parallelism for better load balancing
    let chunk_size: usize = std::cmp::max(1, (n * m) / rayon::current_num_threads());
    
    // Flatten the 2D grid into 1D for more efficient chunking
    let terms: Vec<PauliString> = (0..n*m)
        .into_par_iter()
        .chunks(chunk_size)
        .flat_map(|chunk| {
            let mut local_terms = Vec::with_capacity(chunk.len() * (if include_coupling { 3 } else { 1 }));
            
            for idx in chunk {
                let r: usize = idx / m;
                let c: usize = idx % m;
                let current_site_1d_idx: usize = idx;
                
                // Field term: - mu * h * Z_{r,c}
                if include_field {
                    let mut field_pauli_string: PauliString = PauliString::new(field_coeff);
                    field_pauli_string.add_op(current_site_1d_idx, Pauli::Z);
                    local_terms.push(field_pauli_string);
                }
                
                // Only calculate coupling terms if j is not zero
                if include_coupling {
                    // Vertical coupling term
                    let mut vertical_pauli_string: PauliString = PauliString::new(coupling_coeff);
                    vertical_pauli_string.add_op(current_site_1d_idx, Pauli::Z);
                    let next_row_site_1d_idx: usize = ((r + 1) % n) * m + c;
                    vertical_pauli_string.add_op(next_row_site_1d_idx, Pauli::Z);
                    local_terms.push(vertical_pauli_string);

                    // Horizontal coupling term
                    let mut horizontal_pauli_string: PauliString = PauliString::new(coupling_coeff);
                    horizontal_pauli_string.add_op(current_site_1d_idx, Pauli::Z);
                    let next_col_site_1d_idx: usize = r * m + ((c + 1) % m);
                    horizontal_pauli_string.add_op(next_col_site_1d_idx, Pauli::Z);
                    local_terms.push(horizontal_pauli_string);
                }
            }
            
            local_terms
        })
        .collect();

    Ok(SumOp::new(terms))
}
