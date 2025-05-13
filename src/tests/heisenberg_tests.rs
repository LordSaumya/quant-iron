use crate::{
    components::{
        operator::Pauli,
        pauli_string::{PauliString, SumOp},
        state::State,
    },
    errors::Error,
    models::heisenberg::{heisenberg_1d, heisenberg_2d},
};

use num_complex::Complex;
use rayon::vec;

#[test]
fn test_heisenberg_1d_success() {
    let n = 4;
    let jx = 1.0;
    let jy = 2.0;
    let jz = 3.0;
    let h = 4.0;
    let mu = 5.0;

    let result = heisenberg_1d(n, jx, jy, jz, h, mu);
    assert!(result.is_ok());

    let result = result.unwrap();
    assert_eq!(result.terms.len(), 16); // 4 sites * (3 coupling terms + 1 field term)

    let expected_terms = vec![
        // Field terms
        -0.5 * h * mu * PauliString::new(Complex::new(1.0, 0.0)).with_op(0, Pauli::Z),
        -0.5 * h * mu * PauliString::new(Complex::new(1.0, 0.0)).with_op(1, Pauli::Z),
        -0.5 * h * mu * PauliString::new(Complex::new(1.0, 0.0)).with_op(2, Pauli::Z),
        -0.5 * h * mu * PauliString::new(Complex::new(1.0, 0.0)).with_op(3, Pauli::Z),

        // Coupling terms
        -0.5 * jx
            * PauliString::new(Complex::new(1.0, 0.0))
                .with_op(0, Pauli::X)
                .with_op(1, Pauli::X),
        -0.5 * jx
            * PauliString::new(Complex::new(1.0, 0.0))
                .with_op(1, Pauli::X)
                .with_op(2, Pauli::X),
        -0.5 * jx
            * PauliString::new(Complex::new(1.0, 0.0))
                .with_op(2, Pauli::X)
                .with_op(3, Pauli::X),
        -0.5 * jx
            * PauliString::new(Complex::new(1.0, 0.0))
                .with_op(3, Pauli::X)
                .with_op(0, Pauli::X),
        -0.5 * jy
            * PauliString::new(Complex::new(1.0, 0.0))
                .with_op(0, Pauli::Y)
                .with_op(1, Pauli::Y),
        -0.5 * jy
            * PauliString::new(Complex::new(1.0, 0.0))
                .with_op(1, Pauli::Y)
                .with_op(2, Pauli::Y),
        -0.5 * jy
            * PauliString::new(Complex::new(1.0, 0.0))
                .with_op(2, Pauli::Y)
                .with_op(3, Pauli::Y),
        -0.5 * jy
            * PauliString::new(Complex::new(1.0, 0.0))
                .with_op(3, Pauli::Y)
                .with_op(0, Pauli::Y),
        -0.5 * jz
            * PauliString::new(Complex::new(1.0, 0.0))
                .with_op(0, Pauli::Z)
                .with_op(1, Pauli::Z),
        -0.5 * jz
            * PauliString::new(Complex::new(1.0, 0.0))
                .with_op(1, Pauli::Z)
                .with_op(2, Pauli::Z),
        -0.5 * jz
            * PauliString::new(Complex::new(1.0, 0.0))
                .with_op(2, Pauli::Z)
                .with_op(3, Pauli::Z),
        -0.5 * jz
            * PauliString::new(Complex::new(1.0, 0.0))
                .with_op(3, Pauli::Z)
                .with_op(0, Pauli::Z),
    ];

    let state = State::new_basis_n(4, 6).unwrap();
    let result_state = result.apply(&state).unwrap();
    let expected_state = SumOp::new(expected_terms).apply(&state).unwrap();

    assert_eq!(result_state, expected_state);
}

#[test]
fn test_heisenberg_1d_error() {
    let n = 1; // Invalid number of sites
    let jx = 1.0;
    let jy = 2.0;
    let jz = 3.0;
    let h = 4.0;
    let mu = 5.0;

    let result = heisenberg_1d(n, jx, jy, jz, h, mu);
    assert_eq!(result.unwrap_err(), Error::InvalidNumberOfInputs(1, 2));
}

#[test]
fn test_heisenberg_2d_success() {
    let n = 3;
    let m = 3;
    let jx = 1.0;
    let jy = 2.0;
    let jz = 3.0;
    let h = 4.0;
    let mu = 5.0;

    let result: Result<SumOp, Error> = heisenberg_2d(n, m, jx, jy, jz, h, mu);
    assert!(result.is_ok());

    let result = result.unwrap();
    assert_eq!(result.terms.len(), 63); // 3 x 3 sites * (2 x 3 coupling terms + 1 field term)

    let expected_terms = vec![
        // Field terms
        -0.5 * h * mu * PauliString::new(Complex::new(1.0, 0.0)).with_op(0, Pauli::Z),
        -0.5 * h * mu * PauliString::new(Complex::new(1.0, 0.0)).with_op(1, Pauli::Z),
        -0.5 * h * mu * PauliString::new(Complex::new(1.0, 0.0)).with_op(2, Pauli::Z),
        -0.5 * h * mu * PauliString::new(Complex::new(1.0, 0.0)).with_op(3, Pauli::Z),
        -0.5 * h * mu * PauliString::new(Complex::new(1.0, 0.0)).with_op(4, Pauli::Z),
        -0.5 * h * mu * PauliString::new(Complex::new(1.0, 0.0)).with_op(5, Pauli::Z),
        -0.5 * h * mu * PauliString::new(Complex::new(1.0, 0.0)).with_op(6, Pauli::Z),
        -0.5 * h * mu * PauliString::new(Complex::new(1.0, 0.0)).with_op(7, Pauli::Z),
        -0.5 * h * mu * PauliString::new(Complex::new(1.0, 0.0)).with_op(8, Pauli::Z),

        // Coupling terms
        
        // Jx Horizontal
        -0.5 * jx * PauliString::new(Complex::new(1.0, 0.0)).with_op(0, Pauli::X).with_op(1, Pauli::X),
        -0.5 * jx * PauliString::new(Complex::new(1.0, 0.0)).with_op(1, Pauli::X).with_op(2, Pauli::X),
        -0.5 * jx * PauliString::new(Complex::new(1.0, 0.0)).with_op(2, Pauli::X).with_op(0, Pauli::X),
        -0.5 * jx * PauliString::new(Complex::new(1.0, 0.0)).with_op(3, Pauli::X).with_op(4, Pauli::X),
        -0.5 * jx * PauliString::new(Complex::new(1.0, 0.0)).with_op(4, Pauli::X).with_op(5, Pauli::X),
        -0.5 * jx * PauliString::new(Complex::new(1.0, 0.0)).with_op(5, Pauli::X).with_op(3, Pauli::X),
        -0.5 * jx * PauliString::new(Complex::new(1.0, 0.0)).with_op(6, Pauli::X).with_op(7, Pauli::X),
        -0.5 * jx * PauliString::new(Complex::new(1.0, 0.0)).with_op(7, Pauli::X).with_op(8, Pauli::X),
        -0.5 * jx * PauliString::new(Complex::new(1.0, 0.0)).with_op(8, Pauli::X).with_op(6, Pauli::X),

        // Jx Vertical
        -0.5 * jx * PauliString::new(Complex::new(1.0, 0.0)).with_op(0, Pauli::X).with_op(3, Pauli::X),
        -0.5 * jx * PauliString::new(Complex::new(1.0, 0.0)).with_op(3, Pauli::X).with_op(6, Pauli::X),
        -0.5 * jx * PauliString::new(Complex::new(1.0, 0.0)).with_op(6, Pauli::X).with_op(0, Pauli::X),
        -0.5 * jx * PauliString::new(Complex::new(1.0, 0.0)).with_op(1, Pauli::X).with_op(4, Pauli::X),
        -0.5 * jx * PauliString::new(Complex::new(1.0, 0.0)).with_op(4, Pauli::X).with_op(7, Pauli::X),
        -0.5 * jx * PauliString::new(Complex::new(1.0, 0.0)).with_op(7, Pauli::X).with_op(1, Pauli::X),
        -0.5 * jx * PauliString::new(Complex::new(1.0, 0.0)).with_op(2, Pauli::X).with_op(5, Pauli::X),
        -0.5 * jx * PauliString::new(Complex::new(1.0, 0.0)).with_op(5, Pauli::X).with_op(8, Pauli::X),
        -0.5 * jx * PauliString::new(Complex::new(1.0, 0.0)).with_op(8, Pauli::X).with_op(2, Pauli::X),

        // Jy Horizontal
        -0.5 * jy * PauliString::new(Complex::new(1.0, 0.0)).with_op(0, Pauli::Y).with_op(1, Pauli::Y),
        -0.5 * jy * PauliString::new(Complex::new(1.0, 0.0)).with_op(1, Pauli::Y).with_op(2, Pauli::Y),
        -0.5 * jy * PauliString::new(Complex::new(1.0, 0.0)).with_op(2, Pauli::Y).with_op(0, Pauli::Y),
        -0.5 * jy * PauliString::new(Complex::new(1.0, 0.0)).with_op(3, Pauli::Y).with_op(4, Pauli::Y),
        -0.5 * jy * PauliString::new(Complex::new(1.0, 0.0)).with_op(4, Pauli::Y).with_op(5, Pauli::Y),
        -0.5 * jy * PauliString::new(Complex::new(1.0, 0.0)).with_op(5, Pauli::Y).with_op(3, Pauli::Y),
        -0.5 * jy * PauliString::new(Complex::new(1.0, 0.0)).with_op(6, Pauli::Y).with_op(7, Pauli::Y),
        -0.5 * jy * PauliString::new(Complex::new(1.0, 0.0)).with_op(7, Pauli::Y).with_op(8, Pauli::Y),
        -0.5 * jy * PauliString::new(Complex::new(1.0, 0.0)).with_op(8, Pauli::Y).with_op(6, Pauli::Y),

        // Jy Vertical
        -0.5 * jy * PauliString::new(Complex::new(1.0, 0.0)).with_op(0, Pauli::Y).with_op(3, Pauli::Y),
        -0.5 * jy * PauliString::new(Complex::new(1.0, 0.0)).with_op(3, Pauli::Y).with_op(6, Pauli::Y),
        -0.5 * jy * PauliString::new(Complex::new(1.0, 0.0)).with_op(6, Pauli::Y).with_op(0, Pauli::Y),
        -0.5 * jy * PauliString::new(Complex::new(1.0, 0.0)).with_op(1, Pauli::Y).with_op(4, Pauli::Y),
        -0.5 * jy * PauliString::new(Complex::new(1.0, 0.0)).with_op(4, Pauli::Y).with_op(7, Pauli::Y),
        -0.5 * jy * PauliString::new(Complex::new(1.0, 0.0)).with_op(7, Pauli::Y).with_op(1, Pauli::Y),
        -0.5 * jy * PauliString::new(Complex::new(1.0, 0.0)).with_op(2, Pauli::Y).with_op(5, Pauli::Y),
        -0.5 * jy * PauliString::new(Complex::new(1.0, 0.0)).with_op(5, Pauli::Y).with_op(8, Pauli::Y),
        -0.5 * jy * PauliString::new(Complex::new(1.0, 0.0)).with_op(8, Pauli::Y).with_op(2, Pauli::Y),

        // Jz Horizontal
        -0.5 * jz * PauliString::new(Complex::new(1.0, 0.0)).with_op(0, Pauli::Z).with_op(1, Pauli::Z),
        -0.5 * jz * PauliString::new(Complex::new(1.0, 0.0)).with_op(1, Pauli::Z).with_op(2, Pauli::Z),
        -0.5 * jz * PauliString::new(Complex::new(1.0, 0.0)).with_op(2, Pauli::Z).with_op(0, Pauli::Z),
        -0.5 * jz * PauliString::new(Complex::new(1.0, 0.0)).with_op(3, Pauli::Z).with_op(4, Pauli::Z),
        -0.5 * jz * PauliString::new(Complex::new(1.0, 0.0)).with_op(4, Pauli::Z).with_op(5, Pauli::Z),
        -0.5 * jz * PauliString::new(Complex::new(1.0, 0.0)).with_op(5, Pauli::Z).with_op(3, Pauli::Z),
        -0.5 * jz * PauliString::new(Complex::new(1.0, 0.0)).with_op(6, Pauli::Z).with_op(7, Pauli::Z),
        -0.5 * jz * PauliString::new(Complex::new(1.0, 0.0)).with_op(7, Pauli::Z).with_op(8, Pauli::Z),
        -0.5 * jz * PauliString::new(Complex::new(1.0, 0.0)).with_op(8, Pauli::Z).with_op(6, Pauli::Z),

        // Jz Vertical
        -0.5 * jz * PauliString::new(Complex::new(1.0, 0.0)).with_op(0, Pauli::Z).with_op(3, Pauli::Z),
        -0.5 * jz * PauliString::new(Complex::new(1.0, 0.0)).with_op(3, Pauli::Z).with_op(6, Pauli::Z),
        -0.5 * jz * PauliString::new(Complex::new(1.0, 0.0)).with_op(6, Pauli::Z).with_op(0, Pauli::Z),
        -0.5 * jz * PauliString::new(Complex::new(1.0, 0.0)).with_op(1, Pauli::Z).with_op(4, Pauli::Z),
        -0.5 * jz * PauliString::new(Complex::new(1.0, 0.0)).with_op(4, Pauli::Z).with_op(7, Pauli::Z),
        -0.5 * jz * PauliString::new(Complex::new(1.0, 0.0)).with_op(7, Pauli::Z).with_op(1, Pauli::Z),
        -0.5 * jz * PauliString::new(Complex::new(1.0, 0.0)).with_op(2, Pauli::Z).with_op(5, Pauli::Z),
        -0.5 * jz * PauliString::new(Complex::new(1.0, 0.0)).with_op(5, Pauli::Z).with_op(8, Pauli::Z),
        -0.5 * jz * PauliString::new(Complex::new(1.0, 0.0)).with_op(8, Pauli::Z).with_op(2, Pauli::Z),
    ];

    let state = State::new_basis_n(9, 255).unwrap();
    let result_state = result.apply(&state).unwrap();
    let expected_state = SumOp::new(expected_terms).apply(&state).unwrap();

    assert_eq!(result_state, expected_state);
}

#[test]
fn test_heisenberg_2d_error() {
    let n = 1; // Invalid number of sites
    let m = 2;
    let jx = 1.0;
    let jy = 2.0;
    let jz = 3.0;
    let h = 4.0;
    let mu = 5.0;

    let result: Result<SumOp, Error> = heisenberg_2d(n, m, jx, jy, jz, h, mu);
    assert_eq!(result.unwrap_err(), Error::InvalidNumberOfInputs(1, 2));

    let new_n = 2;
    let new_m = 1; // Invalid number of sites
    let result2: Result<SumOp, Error> = heisenberg_2d(new_n, new_m, jx, jy, jz, h, mu);
    assert_eq!(result2.unwrap_err(), Error::InvalidNumberOfInputs(1, 2));
}