use crate::{
    components::{
        operator::Pauli,
        pauli_string::{PauliString, SumOp},
    },
    errors::Error,
};

use num_complex::Complex;
use rayon::prelude::*;

/// Generates the Hamiltonian for a 1D anisotropic Heisenberg model with periodic boundary conditions
/// and constant coupling and field strengths.
/// H = -1/2 Σ (J_x X_i X_{i+1} + J_y Y_i Y_{i+1} + J_z Z_i Z_{i+1}) - 1/2 Σ μ h Z_i
///
/// # Arguments
///
/// * `n` - The number of spins in the system.
/// * `jx` - The coupling strength J_x for all X_i X_{i+1} terms.
/// * `jy` - The coupling strength J_y for all Y_i Y_{i+1} terms.
/// * `jz` - The coupling strength J_z for all Z_i Z_{i+1} terms.
/// * `h` - The uniform field strength h for all spins.
/// * `mu` - The magnetic moment.
///
/// # Returns
///
/// * `Result<SumOp, Error>` - A result containing the Hamiltonian as a `SumOp` or an error.
pub fn heisenberg_1d(
    n: usize,
    jx: f64,
    jy: f64,
    jz: f64,
    h: f64,
    mu: f64,
) -> Result<SumOp, Error> {
    if n < 2 {
        return Err(Error::InvalidNumberOfInputs(n, 2));
    }

    // Check if all coefficients are zero for early exit
    if jx == 0.0 && jy == 0.0 && jz == 0.0 && h == 0.0 {
        return Ok(SumOp::new(Vec::new()));
    }

    // Pre-compute coefficients as they are uniform across all sites
    let coeff_x: Complex<f64> = Complex::new(-0.5 * jx, 0.0);
    let coeff_y: Complex<f64> = Complex::new(-0.5 * jy, 0.0);
    let coeff_z_coupling: Complex<f64> = Complex::new(-0.5 * jz, 0.0);
    let field_coeff: Complex<f64> = -mu * Complex::new(-0.5 * h, 0.0);

    let include_jx = jx != 0.0;
    let include_jy = jy != 0.0;
    let include_jz = jz != 0.0;
    let include_h = h != 0.0;

    let chunk_size = std::cmp::max(1, n / rayon::current_num_threads());

    let terms: Vec<PauliString> = (0..n)
        .into_par_iter()
        .chunks(chunk_size)
        .flat_map(|chunk| {
            let mut local_terms_capacity = chunk.len() * 4;
            if n == 1 {
                local_terms_capacity = chunk.len();
            }
            let mut local_terms = Vec::with_capacity(local_terms_capacity);

            for i in chunk {
                if n > 1 {
                    let neighbor_idx = (i + 1) % n;

                    if include_jx {
                        let mut pauli_x: PauliString = PauliString::new(coeff_x);
                        pauli_x.add_op(i, Pauli::X);
                        pauli_x.add_op(neighbor_idx, Pauli::X);
                        local_terms.push(pauli_x);
                    }
                    if include_jy {
                        let mut pauli_y: PauliString = PauliString::new(coeff_y);
                        pauli_y.add_op(i, Pauli::Y);
                        pauli_y.add_op(neighbor_idx, Pauli::Y);
                        local_terms.push(pauli_y);
                    }
                    if include_jz {
                        let mut pauli_z_c: PauliString = PauliString::new(coeff_z_coupling);
                        pauli_z_c.add_op(i, Pauli::Z);
                        pauli_z_c.add_op(neighbor_idx, Pauli::Z);
                        local_terms.push(pauli_z_c);
                    }
                }
                if include_h {
                    let mut pauli_z_field: PauliString = PauliString::new(field_coeff);
                    pauli_z_field.add_op(i, Pauli::Z);
                    local_terms.push(pauli_z_field);
                }
            }
            local_terms
        })
        .collect();

    Ok(SumOp::new(terms))
}

/// Generates the Hamiltonian for a 2D anisotropic Heisenberg model with periodic boundary conditions
/// and constant coupling and field strengths.
/// H = -1/2 Σ_{<i,j>} (J_x X_i X_j + J_y Y_i Y_j + J_z Z_i Z_j) - 1/2 Σ_k μ h Z_k
/// where <i,j> denotes nearest neighbors (vertical and horizontal).
///
/// # Arguments
///
/// * `n_rows` - The number of rows of spins in the system.
/// * `m_cols` - The number of columns of spins in the system.
/// * `jx` - The coupling strength J_x for all X_i X_j terms.
/// * `jy` - The coupling strength J_y for all Y_i Y_j terms.
/// * `jz` - The coupling strength J_z for all Z_i Z_j terms.
/// * `h_field` - The uniform field strength h for all spins.
/// * `mu` - The magnetic moment.
///
/// # Returns
///
/// * `Result<SumOp, Error>` - A result containing the Hamiltonian as a `SumOp` or an error.
pub fn heisenberg_2d(
    n_rows: usize,
    m_cols: usize,
    jx: f64,
    jy: f64,
    jz: f64,
    h_field: f64,
    mu: f64,
) -> Result<SumOp, Error> {
    if n_rows < 2 {
        return Err(Error::InvalidNumberOfInputs(n_rows, 2));
    } else if m_cols < 2 {
        return Err(Error::InvalidNumberOfInputs(m_cols, 2));
    }

    // Check if all coefficients are zero for early exit
    if jx == 0.0 && jy == 0.0 && jz == 0.0 && h_field == 0.0 {
        return Ok(SumOp::new(Vec::new()));
    }

    let coeff_x: Complex<f64> = Complex::new(-0.5 * jx, 0.0);
    let coeff_y: Complex<f64> = Complex::new(-0.5 * jy, 0.0);
    let coeff_z_coupling: Complex<f64> = Complex::new(-0.5 * jz, 0.0);
    let field_coeff: Complex<f64> = mu * Complex::new(-0.5 * h_field, 0.0);

    let include_jx = jx != 0.0;
    let include_jy = jy != 0.0;
    let include_jz = jz != 0.0;
    let include_h = h_field != 0.0;

    let total_sites = n_rows * m_cols;
    let chunk_size = std::cmp::max(1, total_sites / rayon::current_num_threads());

    let terms: Vec<PauliString> = (0..total_sites)
        .into_par_iter()
        .chunks(chunk_size)
        .flat_map(|chunk| {
            let mut local_terms = Vec::with_capacity(chunk.len() * 7);

            for site_idx_1d in chunk {
                let r = site_idx_1d / m_cols;
                let c = site_idx_1d % m_cols;

                if include_h {
                    let mut pauli_z_field: PauliString = PauliString::new(field_coeff);
                    pauli_z_field.add_op(site_idx_1d, Pauli::Z);
                    local_terms.push(pauli_z_field);
                }

                if n_rows > 1 { // Check n_rows for vertical coupling
                    let neighbor_r_idx_1d = ((r + 1) % n_rows) * m_cols + c;
                    if include_jx {
                        let mut px: PauliString = PauliString::new(coeff_x);
                        px.add_op(site_idx_1d, Pauli::X);
                        px.add_op(neighbor_r_idx_1d, Pauli::X);
                        local_terms.push(px);
                    }
                    if include_jy {
                        let mut py: PauliString = PauliString::new(coeff_y);
                        py.add_op(site_idx_1d, Pauli::Y);
                        py.add_op(neighbor_r_idx_1d, Pauli::Y);
                        local_terms.push(py);
                    }
                    if include_jz {
                        let mut pz: PauliString = PauliString::new(coeff_z_coupling);
                        pz.add_op(site_idx_1d, Pauli::Z);
                        pz.add_op(neighbor_r_idx_1d, Pauli::Z);
                        local_terms.push(pz);
                    }
                }

                if m_cols > 1 { // Check m_cols for horizontal coupling
                    let neighbor_c_idx_1d = r * m_cols + ((c + 1) % m_cols);
                    if include_jx {
                        let mut px: PauliString = PauliString::new(coeff_x);
                        px.add_op(site_idx_1d, Pauli::X);
                        px.add_op(neighbor_c_idx_1d, Pauli::X);
                        local_terms.push(px);
                    }
                    if include_jy {
                        let mut py: PauliString = PauliString::new(coeff_y);
                        py.add_op(site_idx_1d, Pauli::Y);
                        py.add_op(neighbor_c_idx_1d, Pauli::Y);
                        local_terms.push(py);
                    }
                    if include_jz {
                        let mut pz: PauliString = PauliString::new(coeff_z_coupling);
                        pz.add_op(site_idx_1d, Pauli::Z);
                        pz.add_op(neighbor_c_idx_1d, Pauli::Z);
                        local_terms.push(pz);
                    }
                }
            }
            local_terms
        })
        .collect();

    Ok(SumOp::new(terms))
}
