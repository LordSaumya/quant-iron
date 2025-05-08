use crate::{
    components::{
        operator::Pauli,
        pauli_string::{PauliString, SumOp},
    },
    errors::Error,
};

use num_complex::Complex;

/// Generates the Hamiltonian for a 1D Ising model with periodic boundary conditions
///
/// # Arguments
///
/// * `h` - Vector representing the field strength for each spin. Must be of length `N`.
/// * `j` - Vector representing the coupling strength between spins. Must be of length `N`. Each element `j[i]` represents the coupling strength between spin `i` and spin `i+1`.
/// * `m` - The magnetic moment
///
/// # Parameters
///
/// * `N` - The number of spins in the system
///
/// # Returns
///
/// * `Result<SumOp, Error>` - A result containing the Hamiltonian as a `SumOp` or an error
pub fn ising_1d<const N: usize>(h: [f64; N], j: [f64; N], m: f64) -> Result<SumOp, Error> {
    if N < 2 {
        return Err(Error::InvalidNumberOfInputs(N, 2));
    }

    let mut terms: Vec<PauliString> = Vec::new();
    for i in 0..N {
        // - \sum J_{i} Z_i Z_{i+1}
        let coeff_1: Complex<f64> = Complex::new(j[i] as f64, 0.0) * -1.0;
        let mut pauli_1: PauliString = PauliString::new(coeff_1);
        pauli_1.add_op(i, Pauli::Z);
        pauli_1.add_op((i + 1) % N, Pauli::Z);
        terms.push(pauli_1);

        // - \sum_i m h_i Z_i
        let coeff_2: Complex<f64> = -1.0 * m * Complex::new(h[i], 0.0);
        let mut pauli_2: PauliString = PauliString::new(coeff_2);
        pauli_2.add_op(i, Pauli::Z);
        terms.push(pauli_2);
    }

    Ok(SumOp::new(terms))
}

/// Generates the Hamiltonian for a 1D Ising model with periodic boundary conditions and uniform coupling and field strengths for all spins
///
/// # Arguments
///
/// * `h` - The field strength for each spin
/// * `j` - The coupling strength between spins
/// * `m` - The magnetic moment
///
/// # Parameters
///
/// * `N` - The number of spins in the system
///
/// # Returns
///
/// * `Result<SumOp, Error>` - A result containing the Hamiltonian as a `SumOp` or an error
pub fn ising_1d_uniform<const N: usize>(h: f64, j: f64, m: f64) -> Result<SumOp, Error> {
    if N < 2 {
        return Err(Error::InvalidNumberOfInputs(N, 2));
    }

    let mut terms: Vec<PauliString> = Vec::new();
    for i in 0..N {
        // - \sum J_{i} Z_i Z_{i+1}
        let coeff_1: Complex<f64> = Complex::new(j as f64, 0.0) * -1.0;
        if coeff_1 != Complex::new(0.0, 0.0) {
            let mut pauli_1: PauliString = PauliString::new(coeff_1);
            pauli_1.add_op(i, Pauli::Z);
            pauli_1.add_op((i + 1) % N, Pauli::Z);
            terms.push(pauli_1);
        }

        // - \sum_i m h_i Z_i
        let coeff_2: Complex<f64> = -1.0 * m * Complex::new(h, 0.0);
        if coeff_2 != Complex::new(0.0, 0.0) {
            let mut pauli_2: PauliString = PauliString::new(coeff_2);
            pauli_2.add_op(i, Pauli::Z);
            terms.push(pauli_2);
        }
    }

    Ok(SumOp::new(terms))
}

/// Generates the Hamiltonian for a 2D Ising model with periodic boundary conditions.
///
/// # Arguments
///
/// * `h_param` - Matrix representing the field strength for each spin. `h_param[row][col]` is the field at site (row, col).
///               Dimensions: `N x M`.
/// * `j_param` - Matrix representing the coupling strengths.
///               `j_param[0][row][col]` is the vertical coupling J_vertical for site (row, col) with ((row+1)%N_ROWS, col).
///               `j_param[1][row][col]` is the horizontal coupling J_horizontal for site (row, col) with (row, (col+1)%M_COLS).
///               Dimensions: `2 x N x M`.
/// * `m_param` - The magnetic moment (scales field terms).
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
    j_param: [[[f64; M]; N]; 2],
    m_param: f64,
) -> Result<SumOp, Error> {
    if N < 2 {
        return Err(Error::InvalidNumberOfInputs(N, 2));
    } else if M < 2 {
        return Err(Error::InvalidNumberOfInputs(M, 2));
    }

    let mut terms: Vec<PauliString> = Vec::new();

    for r in 0..N {
        // Iterate over rows
        for c in 0..M {
            // Iterate over columns
            let current_site_1d_idx: usize = r * M + c;

            // Field term: - m_param * h_param[r][c] * Z_{r,c}
            let field_coeff_val: f64 = h_param[r][c];
            let field_coeff: Complex<f64> = -1.0 * m_param * Complex::new(field_coeff_val, 0.0);
            let mut field_pauli_string: PauliString = PauliString::new(field_coeff);
            field_pauli_string.add_op(current_site_1d_idx, Pauli::Z);
            terms.push(field_pauli_string);

            // Vertical coupling term: - J_vertical[r][c] * Z_{r,c} * Z_{(r+1)%N_ROWS, c}
            // J_vertical[r][c] is j_param[0][r][c]
            let vertical_coupling_val: f64 = j_param[0][r][c];
            if vertical_coupling_val != 0.0 {
                let vertical_coeff: Complex<f64> = Complex::new(vertical_coupling_val, 0.0) * -1.0;
                let mut vertical_pauli_string: PauliString = PauliString::new(vertical_coeff);
                vertical_pauli_string.add_op(current_site_1d_idx, Pauli::Z);
                let next_row_site_1d_idx: usize = ((r + 1) % N) * M + c;
                vertical_pauli_string.add_op(next_row_site_1d_idx, Pauli::Z);
                terms.push(vertical_pauli_string);
            }

            // Horizontal coupling term: - J_horizontal[r][c] * Z_{r,c} * Z_{r, (c+1)%M_COLS}
            // J_horizontal[r][c] is j_param[1][r][c]
            let horizontal_coupling_val: f64 = j_param[1][r][c];
            if horizontal_coupling_val != 0.0 {
                let horizontal_coeff: Complex<f64> = Complex::new(horizontal_coupling_val, 0.0) * -1.0;
                let mut horizontal_pauli_string: PauliString = PauliString::new(horizontal_coeff);
                horizontal_pauli_string.add_op(current_site_1d_idx, Pauli::Z);
                let next_col_site_1d_idx: usize = r * M + ((c + 1) % M);
                horizontal_pauli_string.add_op(next_col_site_1d_idx, Pauli::Z);
                terms.push(horizontal_pauli_string);
            }
        }
    }

    Ok(SumOp::new(terms))
}

/// Generates the Hamiltonian for a 2D Ising model with periodic boundary conditions and uniform coupling and field strengths for all spins.
///
/// # Arguments
///
/// * `h` - The field strength for each spin.
/// * `j` - The coupling strength between spins.
/// * `m` - The magnetic moment.
///
/// # Parameters
///
/// * `N` - The number of rows of spins in the system.
/// * `M` - The number of columns of spins in the system.
///
/// # Returns
///
/// * `Result<SumOp, Error>` - A result containing the Hamiltonian as a `SumOp` or an error.
pub fn ising_2d_uniform<const N: usize, const M: usize>(
    h: f64,
    j: f64,
    m: f64,
) -> Result<SumOp, Error> {
    if N < 2 {
        return Err(Error::InvalidNumberOfInputs(N, 2));
    } else if M < 2 {
        return Err(Error::InvalidNumberOfInputs(M, 2));
    }

    let mut terms: Vec<PauliString> = Vec::new();

    for r in 0..N {
        // Iterate over rows
        for c in 0..M {
            // Iterate over columns
            let current_site_1d_idx: usize = r * M + c;

            // Field term: - m * h * Z_{r,c}
            let field_coeff_val: f64 = h;
            let field_coeff: Complex<f64> = -1.0 * m * Complex::new(field_coeff_val, 0.0);
            let mut field_pauli_string: PauliString = PauliString::new(field_coeff);
            field_pauli_string.add_op(current_site_1d_idx, Pauli::Z);
            terms.push(field_pauli_string);

            // Vertical coupling term: - j * Z_{r,c} * Z_{(r+1)%N_ROWS, c}
            let vertical_coupling_val: f64 = j;
            if vertical_coupling_val != 0.0 {
                let vertical_coeff: Complex<f64> = Complex::new(vertical_coupling_val, 0.0) * -1.0;
                let mut vertical_pauli_string: PauliString = PauliString::new(vertical_coeff);
                vertical_pauli_string.add_op(current_site_1d_idx, Pauli::Z);
                let next_row_site_1d_idx: usize = ((r + 1) % N) * M + c;
                vertical_pauli_string.add_op(next_row_site_1d_idx, Pauli::Z);
                terms.push(vertical_pauli_string);
            }

            // Horizontal coupling term: - j * Z_{r,c} * Z_{r, (c+1)%M_COLS}
            let horizontal_coupling_val: f64 = j;
            if horizontal_coupling_val != 0.0 {
                let horizontal_coeff: Complex<f64> = Complex::new(horizontal_coupling_val, 0.0) * -1.0;
                let mut horizontal_pauli_string: PauliString = PauliString::new(horizontal_coeff);
                horizontal_pauli_string.add_op(current_site_1d_idx, Pauli::Z);
                let next_col_site_1d_idx: usize = r * M + ((c + 1) % M);
                horizontal_pauli_string.add_op(next_col_site_1d_idx, Pauli::Z);
                terms.push(horizontal_pauli_string);
            }
        }
    }

    Ok(SumOp::new(terms))
}
