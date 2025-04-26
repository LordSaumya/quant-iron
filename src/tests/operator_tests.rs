use crate::{
    components::{
        ChainableState,
        operator::{
            CNOT, Hadamard, Identity, Operator, Pauli, PhaseS, PhaseSdag, PhaseShift, PhaseT,
            PhaseTdag, RotateX, RotateY, RotateZ, SWAP, Toffoli,
        },
        state::State,
    },
    errors::Error,
};
use num_complex::Complex;
use std::f64::consts::PI;

// -- SINGLE-QUBIT OPERATORS --
#[test]
fn test_operator_hadamard_success() {
    // h(|0>) = |+>
    // h(|1>) = |->
    // h(|+>) = |0>
    // h(|->) = |1>
    let zero_state: State = State::new_zero(1).unwrap();
    let one_state: State = State::new_basis_n(1, 1).unwrap();
    let plus_state: State = State::new_plus(1).unwrap();
    let minus_state: State = State::new_minus(1).unwrap();

    assert_eq!(zero_state.h(0).unwrap(), plus_state);
    assert_eq!(one_state.h(0).unwrap(), minus_state);
    assert_eq!(plus_state.h(0).unwrap(), zero_state);
    assert_eq!(minus_state.h(0).unwrap(), one_state);

    // h(|00>) = |++>
    let two_qubit_state: State = State::new_zero(2).unwrap();
    let new_state: State = two_qubit_state.h_multi(&[0, 1]).unwrap();
    let expected_state: State = State::new_plus(2).unwrap();

    assert_eq!(new_state, expected_state);

    // Base qubits = 1
    assert_eq!(Hadamard {}.base_qubits(), 1);
}

#[test]
fn test_operator_pauli_x_success() {
    // x(|0>) = |1>
    // x(|1>) = |0>
    // x(|+>) = |+> (|+> is an eigenstate of X with eigenvalue = +1)
    // x(|->) = -|-> (|-> is an eigenstate of X with eigenvalue = -1)
    let zero_state: State = State::new_zero(1).unwrap();
    let one_state: State = State::new_basis_n(1, 1).unwrap();
    let plus_state: State = State::new_plus(1).unwrap();
    let minus_state: State = State::new_minus(1).unwrap();

    assert_eq!(zero_state.x(0).unwrap(), one_state);
    assert_eq!(one_state.x(0).unwrap(), zero_state);
    assert_eq!(plus_state.x(0).unwrap(), plus_state);
    assert_eq!(minus_state.x(0).unwrap(), minus_state * -1.0);

    // x(|00>) = |11>
    let two_qubit_state: State = State::new_zero(2).unwrap();
    let new_state: State = two_qubit_state.x_multi(&[0, 1]).unwrap();
    let expected_state: State = State::new_basis_n(2, 3).unwrap(); // |11> in 2 qubits
    assert_eq!(new_state, expected_state);

    // x(x(|0>)) = |0> (Pauli-X matrix is self-inverse)
    let new_state: State = zero_state.x(0).x(0).unwrap();
    assert_eq!(new_state, zero_state);

    // Base qubits = 1
    assert_eq!(Pauli::X.base_qubits(), 1);
}

#[test]
fn test_operator_pauli_y_success() {
    //  y(|0>) = i|1>
    //  y(|1>) = -i|0>
    //  y(|+>) = -i|->
    //  y(|->) = i|+>
    let zero_state: State = State::new_zero(1).unwrap();
    let one_state: State = State::new_basis_n(1, 1).unwrap();
    let plus_state: State = State::new_plus(1).unwrap();
    let minus_state: State = State::new_minus(1).unwrap();

    let i: Complex<f64> = Complex::new(0.0, 1.0);
    let neg_i: Complex<f64> = Complex::new(0.0, -1.0);
    let neg_one: Complex<f64> = Complex::new(-1.0, 0.0);

    assert_eq!(zero_state.y(0).unwrap(), i * one_state.clone());
    assert_eq!(one_state.y(0).unwrap(), neg_i * zero_state.clone());
    assert_eq!(plus_state.y(0).unwrap(), neg_i * minus_state.clone());
    assert_eq!(minus_state.y(0).unwrap(), i * plus_state.clone());

    // y(|00>) = -|11>
    let two_qubit_state: State = State::new_zero(2).unwrap();
    let new_state: State = two_qubit_state.y_multi(&[0, 1]).unwrap();
    let expected_state: State = State::new_basis_n(2, 3).unwrap() * neg_one;
    assert_eq!(new_state, expected_state);

    // y(y(|0>)) = |0> (Pauli-Y matrix is self-inverse)
    let new_state: State = zero_state.y(0).y(0).unwrap();
    assert_eq!(new_state, zero_state);

    // Base qubits = 1
    assert_eq!(Pauli::Y.base_qubits(), 1);
}

#[test]
fn test_operator_pauli_success() {
    // z(|0>) = |0> (|0> is an eigenstate of Z with eigenvalue = +1)
    // z(|1>) = -|1> (|1> is an eigenstate of Z with eigenvalue = -1)
    // z(|+>) = |->
    // z(|->) = |+>
    let zero_state: State = State::new_zero(1).unwrap();
    let one_state: State = State::new_basis_n(1, 1).unwrap();
    let plus_state: State = State::new_plus(1).unwrap();
    let minus_state: State = State::new_minus(1).unwrap();

    assert_eq!(zero_state.z(0).unwrap(), zero_state.clone());
    assert_eq!(one_state.z(0).unwrap(), one_state.clone() * -1.0);
    assert_eq!(plus_state.z(0).unwrap(), minus_state.clone());
    assert_eq!(minus_state.z(0).unwrap(), plus_state.clone());

    // z(|00>) = |00>
    let two_qubit_state: State = State::new_zero(2).unwrap();
    let new_state: State = two_qubit_state.z_multi(&[0, 1]).unwrap();
    let expected_state: State = State::new_zero(2).unwrap();
    assert_eq!(new_state, expected_state);

    // z(z(|+>)) = |+> (Pauli-Z matrix is self-inverse)
    let new_state: State = plus_state.z(0).z(0).unwrap();
    assert_eq!(new_state, plus_state);

    // Base qubits = 1
    assert_eq!(Pauli::Z.base_qubits(), 1);
}

#[test]
fn test_operator_identity_success() {
    // i(|0>) = |0>
    // i(|1>) = |1>
    // i(|+>) = |+>
    // i(|->) = |->
    let zero_state: State = State::new_zero(1).unwrap();
    let one_state: State = State::new_basis_n(1, 1).unwrap();
    let plus_state: State = State::new_plus(1).unwrap();
    let minus_state: State = State::new_minus(1).unwrap();

    assert_eq!(zero_state.i(0).unwrap(), zero_state.clone());
    assert_eq!(one_state.i(0).unwrap(), one_state.clone());
    assert_eq!(plus_state.i(0).unwrap(), plus_state.clone());
    assert_eq!(minus_state.i(0).unwrap(), minus_state.clone());

    // i(|00>) = |00>
    let two_qubit_state: State = State::new_zero(2).unwrap();
    let new_state: State = two_qubit_state.i_multi(&[0, 1]).unwrap();
    let expected_state: State = State::new_zero(2).unwrap();
    assert_eq!(new_state, expected_state);

    // Base qubits = 1
    assert_eq!(Identity {}.base_qubits(), 1);
}

#[test]
fn test_operator_phase_s_success() {
    // s(|0>) = |0>
    // s(|1>) = i|1>
    // s(|+>) = 1/sqrt(2)(|0> + i|1>)
    // s(|->) = 1/sqrt(2)(|0> - i|1>)
    let zero_state: State = State::new_zero(1).unwrap();
    let one_state: State = State::new_basis_n(1, 1).unwrap();
    let plus_state: State = State::new_plus(1).unwrap();
    let minus_state: State = State::new_minus(1).unwrap();

    let i: Complex<f64> = Complex::new(0.0, 1.0);
    let invrt2: Complex<f64> = Complex::new(1.0 / 2.0_f64.sqrt(), 0.0);
    let expected_plus: State = invrt2 * (zero_state.clone() + i * one_state.clone());
    let expected_minus: State = invrt2 * (zero_state.clone() - i * one_state.clone());

    assert_eq!(zero_state.s(0).unwrap(), zero_state.clone());
    assert_eq!(one_state.s(0).unwrap(), i * one_state.clone());
    assert_eq!(plus_state.s(0).unwrap(), expected_plus.clone());
    assert_eq!(minus_state.s(0).unwrap(), expected_minus.clone());

    // s(|00>) = |00>
    let two_qubit_state: State = State::new_zero(2).unwrap();
    let new_state: State = two_qubit_state.s_multi(&[0, 1]).unwrap();
    let expected_state: State = State::new_zero(2).unwrap();
    assert_eq!(new_state, expected_state);

    // Base qubits = 1
    assert_eq!(PhaseS {}.base_qubits(), 1);
}

#[test]
fn test_operator_phase_t_success() {
    // t(|0>) = |0>
    // t(|1>) = e^(i*PI/4)|1>
    // t(|+>) = 1/sqrt(2)(|0> + e^(i*PI/4)|1>)
    // t(|->) = 1/sqrt(2)(|0> - e^(i*PI/4)|1>)
    let zero_state: State = State::new_zero(1).unwrap();
    let one_state: State = State::new_basis_n(1, 1).unwrap();
    let plus_state: State = State::new_plus(1).unwrap();
    let minus_state: State = State::new_minus(1).unwrap();

    let eipi4: Complex<f64> = Complex::new(0.0, PI / 4.0).exp();
    let invrt2: Complex<f64> = Complex::new(1.0 / 2.0_f64.sqrt(), 0.0);
    let expected_plus: State = invrt2 * (zero_state.clone() + eipi4 * one_state.clone());
    let expected_minus: State = invrt2 * (zero_state.clone() - eipi4 * one_state.clone());

    assert_eq!(zero_state.t(0).unwrap(), zero_state.clone());
    assert_eq!(one_state.t(0).unwrap(), eipi4 * one_state.clone());
    assert_eq!(plus_state.t(0).unwrap(), expected_plus.clone());
    assert_eq!(minus_state.t(0).unwrap(), expected_minus.clone());

    // t(|00>) = |00>
    let two_qubit_state: State = State::new_zero(2).unwrap();
    let new_state: State = two_qubit_state.t_multi(&[0, 1]).unwrap();
    let expected_state: State = State::new_zero(2).unwrap();
    assert_eq!(new_state, expected_state);

    // Base qubits = 1
    assert_eq!(PhaseT {}.base_qubits(), 1);
}

#[test]
fn test_operator_s_dag_success() {
    // s_dag(|0>) = |0>
    // s_dag(|1>) = -i|1>
    // s_dag(|+>) = 1/sqrt(2)(|0> - i|1>)
    // s_dag(|->) = 1/sqrt(2)(|0> + i|1>)
    let zero_state: State = State::new_zero(1).unwrap();
    let one_state: State = State::new_basis_n(1, 1).unwrap();
    let plus_state: State = State::new_plus(1).unwrap();
    let minus_state: State = State::new_minus(1).unwrap();

    let i: Complex<f64> = Complex::new(0.0, 1.0);
    let neg_i: Complex<f64> = Complex::new(0.0, -1.0);
    let invrt2: Complex<f64> = Complex::new(1.0 / 2.0_f64.sqrt(), 0.0);

    let expected_one: State = neg_i * one_state.clone();
    let expected_plus: State = invrt2 * (zero_state.clone() - i * one_state.clone());
    let expected_minus: State = invrt2 * (zero_state.clone() + i * one_state.clone());

    assert_eq!(zero_state.s_dag(0).unwrap(), zero_state.clone());
    assert_eq!(one_state.s_dag(0).unwrap(), expected_one.clone());
    assert_eq!(plus_state.s_dag(0).unwrap(), expected_plus.clone());
    assert_eq!(minus_state.s_dag(0).unwrap(), expected_minus.clone());

    // s_dag(|00>) = |00>
    let two_qubit_state: State = State::new_zero(2).unwrap();
    let new_state: State = two_qubit_state.s_dag_multi(&[0, 1]).unwrap();
    let expected_state: State = State::new_zero(2).unwrap();
    assert_eq!(new_state, expected_state);

    // s_dag(s(|+>)) = |+> (S_dag is inverse of S)
    let new_state: State = plus_state.s(0).s_dag(0).unwrap();
    assert_eq!(new_state, plus_state);

    // Base qubits = 1
    assert_eq!(PhaseSdag {}.base_qubits(), 1);
}

#[test]
fn test_operator_t_dag_success() {
    // t_dag(|0>) = |0>
    // t_dag(|1>) = e^(-i*PI/4)|1>
    // t_dag(|+>) = 1/sqrt(2)(|0> + e^(-i*PI/4)|1>)
    // t_dag(|->) = 1/sqrt(2)(|0> - e^(-i*PI/4)|1>)
    let zero_state: State = State::new_zero(1).unwrap();
    let one_state: State = State::new_basis_n(1, 1).unwrap();
    let plus_state: State = State::new_plus(1).unwrap();
    let minus_state: State = State::new_minus(1).unwrap();

    let enegipi4: Complex<f64> = Complex::new(0.0, -PI / 4.0).exp();
    let invrt2: Complex<f64> = Complex::new(1.0 / 2.0_f64.sqrt(), 0.0);

    let expected_one: State = enegipi4 * one_state.clone();
    let expected_plus: State = invrt2 * (zero_state.clone() + enegipi4 * one_state.clone());
    let expected_minus: State = invrt2 * (zero_state.clone() - enegipi4 * one_state.clone());

    assert_eq!(zero_state.t_dag(0).unwrap(), zero_state.clone());
    assert_eq!(one_state.t_dag(0).unwrap(), expected_one.clone());
    assert_eq!(plus_state.t_dag(0).unwrap(), expected_plus.clone());
    assert_eq!(minus_state.t_dag(0).unwrap(), expected_minus.clone());

    // t_dag(|00>) = |00>
    let two_qubit_state: State = State::new_zero(2).unwrap();
    let new_state: State = two_qubit_state.t_dag_multi(&[0, 1]).unwrap();
    let expected_state: State = State::new_zero(2).unwrap();
    assert_eq!(new_state, expected_state);

    // t_dag(t(|+>)) = |+> (T_dag is inverse of T)
    let new_state: State = plus_state.t(0).t_dag(0).unwrap();
    assert_eq!(new_state, plus_state);

    // Base qubits = 1
    assert_eq!(PhaseTdag {}.base_qubits(), 1);
}

#[test]
fn test_operator_phase_shift_success() {
    // p(|0>) = |0>
    // p(|1>) = e^(i*theta)|1>
    // p(|+>) = 1/sqrt(2)(|0> + e^(i*theta)|1>)
    // p(|->) = 1/sqrt(2)(|0> - e^(i*theta)|1>)

    let theta: f64 = PI / 2.5; // Example angle

    let zero_state: State = State::new_zero(1).unwrap();
    let one_state: State = State::new_basis_n(1, 1).unwrap();
    let plus_state: State = State::new_plus(1).unwrap();
    let minus_state: State = State::new_minus(1).unwrap();

    let eitheta: Complex<f64> = Complex::new(0.0, theta).exp();
    let invrt2: Complex<f64> = Complex::new(1.0 / 2.0_f64.sqrt(), 0.0);

    let expected_one: State = eitheta * one_state.clone();
    let expected_plus: State = invrt2 * (zero_state.clone() + eitheta * one_state.clone());
    let expected_minus: State = invrt2 * (zero_state.clone() - eitheta * one_state.clone());

    assert_eq!(zero_state.p(0, theta).unwrap(), zero_state.clone());
    assert_eq!(one_state.p(0, theta).unwrap(), expected_one.clone());
    assert_eq!(plus_state.p(0, theta).unwrap(), expected_plus.clone());
    assert_eq!(minus_state.p(0, theta).unwrap(), expected_minus.clone());

    // p(|+>, Pi/2) = s(|+>), p(|+>, -Pi/2) = s_dag(|+>)
    let new_state: State = plus_state.p(0, -PI / 2.0).unwrap();
    assert_eq!(new_state, plus_state.s_dag(0).unwrap());
    let new_state: State = plus_state.p(0, PI / 2.0).unwrap();
    assert_eq!(new_state, plus_state.s(0).unwrap());

    // p(|+>, pi/4) = t(|+>), p(|+>, -pi/4) = t_dag(|+>)
    let new_state: State = plus_state.p(0, -PI / 4.0).unwrap();
    assert_eq!(new_state, plus_state.t_dag(0).unwrap());
    let new_state: State = plus_state.p(0, PI / 4.0).unwrap();
    assert_eq!(new_state, plus_state.t(0).unwrap());

    // p(|00>) = |00>
    let two_qubit_state: State = State::new_zero(2).unwrap();
    let new_state: State = two_qubit_state.p_multi(&[0, 1], theta).unwrap();
    let expected_state: State = State::new_zero(2).unwrap();
    assert_eq!(new_state, expected_state);

    // Base qubits = 1
    assert_eq!(PhaseShift::new(theta).base_qubits(), 1);
}

#[test]
fn test_operator_rotate_x_success() {
    // rx(|0>) = cos(theta/2)|0> - i*sin(theta/2)|1>
    // rx(|1>) = -isin(theta/2)|0> + cos(theta/2)|1>
    // rx(|+>) = e^(-i*theta/2)|+>
    // rx(|->) = e^(i*theta/2)|->
    let theta: f64 = PI / 2.5; // Example angle

    let zero_state: State = State::new_zero(1).unwrap();
    let one_state: State = State::new_basis_n(1, 1).unwrap();
    let plus_state: State = State::new_plus(1).unwrap();
    let minus_state: State = State::new_minus(1).unwrap();

    let cos_half_theta: f64 = (theta / 2.0).cos();
    let sin_half_theta: f64 = (theta / 2.0).sin();
    let eimhalf_theta: Complex<f64> = Complex::new(0.0, -theta / 2.0).exp();
    let eimhalf_theta_conj: Complex<f64> = Complex::new(0.0, theta / 2.0).exp();
    let i: Complex<f64> = Complex::new(0.0, 1.0);

    let expected_zero: State =
        cos_half_theta * zero_state.clone() + i * -sin_half_theta * one_state.clone();
    let expected_one: State =
        -i * sin_half_theta * zero_state.clone() + cos_half_theta * one_state.clone();
    let expected_plus: State = eimhalf_theta * plus_state.clone();
    let expected_minus: State = eimhalf_theta_conj * minus_state.clone();

    assert_eq!(zero_state.rx(0, theta).unwrap(), expected_zero.clone());
    assert_eq!(one_state.rx(0, theta).unwrap(), expected_one.clone());
    assert_eq!(plus_state.rx(0, theta).unwrap(), expected_plus.clone());
    assert_eq!(minus_state.rx(0, theta).unwrap(), expected_minus.clone());

    // rx(|00>) = cos^2(theta/2)|00> - i*cos(theta/2)*sin(theta/2)|01> - i*cos(theta/2)*sin(theta/2)|10> - sin^2(theta/2)|11>
    let two_qubit_state: State = State::new_zero(2).unwrap();
    let new_state: State = two_qubit_state.rx_multi(&[0, 1], theta).unwrap();
    let expected_state: State = State::new_zero(2).unwrap() * cos_half_theta * cos_half_theta
        + State::new_basis_n(2, 1).unwrap() * -i * cos_half_theta * sin_half_theta
        + State::new_basis_n(2, 2).unwrap() * -i * cos_half_theta * sin_half_theta
        + State::new_basis_n(2, 3).unwrap() * -sin_half_theta * sin_half_theta;
    assert_eq!(new_state, expected_state);

    // Base qubits = 1
    assert_eq!(RotateX::new(theta).base_qubits(), 1);
}

#[test]
fn test_operator_rotate_y_success() {
    // ry(|0>) = cos(theta/2)|0> + sin(theta/2)|1>
    // ry(|1>) = -sin(theta/2)|0> + cos(theta/2)|1>
    // ry(|+>) = 1/rt2((cos(theta/2) + sin(theta/2))|0> + (cos(theta/2) - sin(theta/2))|1>)
    // ry(|->) = 1/rt2((cos(theta/2) - sin(theta/2))|0> + (cos(theta/2) + sin(theta/2))|1>)

    let theta: f64 = PI / 2.5; // Example angle

    let zero_state: State = State::new_zero(1).unwrap();
    let one_state: State = State::new_basis_n(1, 1).unwrap();
    let plus_state: State = State::new_plus(1).unwrap();
    let minus_state: State = State::new_minus(1).unwrap();

    let cos_half_theta: f64 = (theta / 2.0).cos();
    let sin_half_theta: f64 = (theta / 2.0).sin();
    let invrt2: Complex<f64> = Complex::new(1.0 / 2.0_f64.sqrt(), 0.0);

    let expected_zero: State =
        cos_half_theta * zero_state.clone() + sin_half_theta * one_state.clone();
    let expected_one: State =
        -sin_half_theta * zero_state.clone() + cos_half_theta * one_state.clone();
    let expected_plus: State = invrt2
        * ((cos_half_theta - sin_half_theta) * zero_state.clone()
            + (cos_half_theta + sin_half_theta) * one_state.clone());
    let expected_minus: State = invrt2
        * ((cos_half_theta + sin_half_theta) * zero_state.clone()
            - (cos_half_theta - sin_half_theta) * one_state.clone());

    assert_eq!(zero_state.ry(0, theta).unwrap(), expected_zero.clone());
    assert_eq!(one_state.ry(0, theta).unwrap(), expected_one.clone());
    assert_eq!(plus_state.ry(0, theta).unwrap(), expected_plus.clone());
    assert_eq!(minus_state.ry(0, theta).unwrap(), expected_minus.clone());

    // ry(|00>) = cos^2(theta/2)|00> + sin(theta/2)*cos(theta/2)|01> + sin(theta/2)*cos(theta/2)|10> + sin^2(theta/2)|11>
    let two_qubit_state: State = State::new_zero(2).unwrap();
    let new_state: State = two_qubit_state.ry_multi(&[0, 1], theta).unwrap();
    let expected_state: State = State::new_zero(2).unwrap() * cos_half_theta * cos_half_theta
        + State::new_basis_n(2, 1).unwrap() * sin_half_theta * cos_half_theta
        + State::new_basis_n(2, 2).unwrap() * sin_half_theta * cos_half_theta
        + State::new_basis_n(2, 3).unwrap() * sin_half_theta * sin_half_theta;
    assert_eq!(new_state, expected_state);

    // Base qubits = 1
    assert_eq!(RotateY::new(theta).base_qubits(), 1);
}

#[test]
fn test_operator_rotate_z_success() {
    // rz(|0>) = e^(-i*theta/2)|0>
    // rz(|1>) = e^(i*theta/2)|1>
    // rz(|+>) = 1/rt2 * (e^(-i*theta/2) |0> + e^(i*theta/2) |1>)
    // rz(|->) = 1/rt2 * (e^(-i*theta/2) |0> - e^(i*theta/2) |1>)

    let theta: f64 = PI / 2.5; // Example angle

    let zero_state: State = State::new_zero(1).unwrap();
    let one_state: State = State::new_basis_n(1, 1).unwrap();
    let plus_state: State = State::new_plus(1).unwrap();
    let minus_state: State = State::new_minus(1).unwrap();

    let eimhalf_theta: Complex<f64> = Complex::new(0.0, -theta / 2.0).exp();
    let eimhalf_theta_conj: Complex<f64> = Complex::new(0.0, theta / 2.0).exp();
    let invrt2: Complex<f64> = Complex::new(1.0 / 2.0_f64.sqrt(), 0.0);

    let expected_zero: State = eimhalf_theta * zero_state.clone();
    let expected_one: State = eimhalf_theta_conj * one_state.clone();
    let expected_plus: State =
        invrt2 * (eimhalf_theta * zero_state.clone() + eimhalf_theta_conj * one_state.clone());
    let expected_minus: State =
        invrt2 * (eimhalf_theta * zero_state.clone() - eimhalf_theta_conj * one_state.clone());

    assert_eq!(zero_state.rz(0, theta).unwrap(), expected_zero.clone());
    assert_eq!(one_state.rz(0, theta).unwrap(), expected_one.clone());
    assert_eq!(plus_state.rz(0, theta).unwrap(), expected_plus.clone());
    assert_eq!(minus_state.rz(0, theta).unwrap(), expected_minus.clone());

    // rz(|00>) = e^(-i*theta)|00>
    let two_qubit_state: State = State::new_zero(2).unwrap();
    let new_state: State = two_qubit_state.rz_multi(&[0, 1], theta).unwrap();
    let expected_state: State = State::new_zero(2).unwrap() * eimhalf_theta * eimhalf_theta;
    assert_eq!(new_state, expected_state);

    // Base qubits = 1
    assert_eq!(RotateZ::new(theta).base_qubits(), 1);
}

// -- MULTI-QUBIT OPERATORS --

#[test]
fn test_operator_cnot_success() {
    // cnot(control = 0, target = 1)
    // |00> -> |00>
    let state: State = State::new_zero(2).unwrap();
    let new_state: State = state.cnot(0, 1).unwrap();
    let expected_state: State = State::new_zero(2).unwrap();
    assert_eq!(new_state, expected_state);

    // |01> -> |01>
    let state: State = State::new_basis_n(2, 2).unwrap();
    let new_state: State = state.cnot(0, 1).unwrap();
    let expected_state: State = State::new_basis_n(2, 2).unwrap();
    assert_eq!(new_state, expected_state);

    // |10> -> |11>
    let state: State = State::new_basis_n(2, 1).unwrap();
    let new_state: State = state.cnot(0, 1).unwrap();
    let expected_state: State = State::new_basis_n(2, 3).unwrap();
    assert_eq!(new_state, expected_state);

    // |11> -> |10>
    let state: State = State::new_basis_n(2, 3).unwrap();
    let new_state: State = state.cnot(0, 1).unwrap();
    let expected_state: State = State::new_basis_n(2, 1).unwrap();
    assert_eq!(new_state, expected_state);

    // |+0> -> 1/rt2 * (|00> + |11>)
    let state: State =
        1.0 / 2.0_f64.sqrt() * (State::new_zero(2).unwrap() + State::new_basis_n(2, 1).unwrap());
    let new_state: State = state.cnot(0, 1).unwrap();
    let expected_state: State = Complex::new(1.0 / 2.0_f64.sqrt(), 0.0)
        * (State::new_basis_n(2, 0).unwrap() + State::new_basis_n(2, 3).unwrap());
    assert_eq!(new_state, expected_state);

    // Base qubits = 2
    assert_eq!(CNOT.base_qubits(), 2);
}

#[test]
// Test SWAP gate
fn test_operator_swap_success() {
    // swap(q1 = 0, q2 = 1)
    // |00> -> |00>
    let state: State = State::new_zero(2).unwrap();
    let new_state: State = state.swap(0, 1).unwrap();
    let expected_state: State = State::new_zero(2).unwrap();
    assert_eq!(new_state, expected_state);

    // |01> -> |10>
    let state: State = State::new_basis_n(2, 2).unwrap();
    let new_state: State = state.swap(0, 1).unwrap();
    let expected_state: State = State::new_basis_n(2, 1).unwrap();
    assert_eq!(new_state, expected_state);

    // |10> -> |01>
    let state: State = State::new_basis_n(2, 1).unwrap();
    let new_state: State = state.swap(0, 1).unwrap();
    let expected_state: State = State::new_basis_n(2, 2).unwrap();
    assert_eq!(new_state, expected_state);

    // |11> -> |11>
    let state: State = State::new_basis_n(2, 3).unwrap();
    let new_state: State = state.swap(0, 1).unwrap();
    let expected_state: State = State::new_basis_n(2, 3).unwrap();
    assert_eq!(new_state, expected_state);

    // |+-> -> |-+>
    let state: State = 1.0 / 2.0
        * (State::new_zero(2).unwrap() - State::new_basis_n(2, 1).unwrap()
            + State::new_basis_n(2, 2).unwrap()
            - State::new_basis_n(2, 3).unwrap());
    let new_state: State = state.swap(0, 1).unwrap();
    let expected_state: State = 1.0 / 2.0
        * (State::new_zero(2).unwrap() + State::new_basis_n(2, 1).unwrap()
            - State::new_basis_n(2, 2).unwrap()
            - State::new_basis_n(2, 3).unwrap());
    assert_eq!(new_state, expected_state);

    // Base qubits = 2
    assert_eq!(SWAP.base_qubits(), 2);
}

#[test]
fn test_operator_toffoli_success() {
    // Toffoli gate (CCNOT) with control qubits 0 and 1, target qubit 2
    // |000> -> |000>
    let state: State = State::new_zero(3).unwrap();
    let new_state: State = state.toffoli(0, 1, 2).unwrap();
    let expected_state: State = State::new_zero(3).unwrap();
    assert_eq!(new_state, expected_state);

    // |001> -> |001>
    let state: State = State::new_basis_n(3, 1).unwrap();
    let new_state: State = state.toffoli(0, 1, 2).unwrap();
    let expected_state: State = State::new_basis_n(3, 1).unwrap();
    assert_eq!(new_state, expected_state);

    // |010> -> |010>
    let state: State = State::new_basis_n(3, 2).unwrap();
    let new_state: State = state.toffoli(0, 1, 2).unwrap();
    let expected_state: State = State::new_basis_n(3, 2).unwrap();
    assert_eq!(new_state, expected_state);

    // |011> -> |111>
    let state: State = State::new_basis_n(3, 3).unwrap();
    let new_state: State = state.toffoli(0, 1, 2).unwrap();
    let expected_state: State = State::new_basis_n(3, 7).unwrap();
    assert_eq!(new_state, expected_state);

    // |100> -> |100>
    let state: State = State::new_basis_n(3, 4).unwrap();
    let new_state: State = state.toffoli(0, 1, 2).unwrap();
    let expected_state: State = State::new_basis_n(3, 4).unwrap();
    assert_eq!(new_state, expected_state);

    // |101> -> |101>
    let state: State = State::new_basis_n(3, 5).unwrap();
    let new_state: State = state.toffoli(0, 1, 2).unwrap();
    let expected_state: State = State::new_basis_n(3, 5).unwrap();
    assert_eq!(new_state, expected_state);

    // |110> -> |110>
    let state: State = State::new_basis_n(3, 6).unwrap();
    let new_state: State = state.toffoli(0, 1, 2).unwrap();
    let expected_state: State = State::new_basis_n(3, 6).unwrap();
    assert_eq!(new_state, expected_state);

    // |111> -> |011>
    let state: State = State::new_basis_n(3, 7).unwrap();
    let new_state: State = state.toffoli(0, 1, 2).unwrap();
    let expected_state: State = State::new_basis_n(3, 3).unwrap();
    assert_eq!(new_state, expected_state);

    // Base qubits = 3
    assert_eq!(Toffoli.base_qubits(), 3);
}

#[test]
fn test_operate_operate_success() {
    // Test operate function with a single qubit gate and existing operator
    let h: Hadamard = Hadamard {};
    let state: State = State::new_zero(1).unwrap();
    let new_state: State = state.operate(h, &[0], &[]).unwrap();
    let expected_state: State = State::new_plus(1).unwrap();
    assert_eq!(new_state, expected_state);

    // Test operate function with a multi-qubit gate and existing operator
    let cnot: CNOT = CNOT {};
    let state: State = State::new_basis_n(2, 1).unwrap();
    let new_state: State = state.operate(cnot, &[1], &[0]).unwrap();
    let expected_state: State = State::new_basis_n(2, 3).unwrap();
    assert_eq!(new_state, expected_state);
}

// -- TEST ALL ERRORS --

// Macro to generate error test logic for single-qubit gates
macro_rules! generate_single_qubit_gate_error_assertions {
    // Match gates without angles
    ($state:ident, $gate_method:ident) => {
        paste::paste! { // Use paste crate to concatenate identifiers
            // Test single application with invalid index
            let result_single = $state.$gate_method(2); // Index 2 is invalid for a 2-qubit state
            assert!(matches!(result_single, Err(Error::InvalidQubitIndex(2, 2))),
                    "Gate '{}' single failed: Expected InvalidQubitIndex(2, 2), got {:?}", stringify!($gate_method), result_single);

            // Test multi application with invalid index
            let result_multi = $state.[<$gate_method _multi>](&[0, 2]); // Index 2 is invalid
            assert!(matches!(result_multi, Err(Error::InvalidQubitIndex(2, 2))),
                    "Gate '{}' multi failed: Expected InvalidQubitIndex(2, 2), got {:?}", stringify!($gate_method), result_multi);
        }
    };
    // Match gates with angles
    ($state:ident, $gate_method:ident, $angle:expr) => {
        paste::paste! { // Use paste crate to concatenate identifiers
            // Test single application with invalid index
            let result_single = $state.$gate_method(2, $angle); // Index 2 is invalid for a 2-qubit state
            assert!(matches!(result_single, Err(Error::InvalidQubitIndex(2, 2))),
                    "Gate '{}' single failed: Expected InvalidQubitIndex(2, 2), got {:?}", stringify!($gate_method), result_single);

            // Test multi application with invalid index
            let result_multi = $state.[<$gate_method _multi>](&[0, 2], $angle); // Index 2 is invalid
            assert!(matches!(result_multi, Err(Error::InvalidQubitIndex(2, 2))),
                    "Gate '{}' multi failed: Expected InvalidQubitIndex(2, 2), got {:?}", stringify!($gate_method), result_multi);
        }
    };
}

#[test]
fn test_single_qubit_gate_errors() {
    let state = State::new_zero(2).unwrap();
    let angle = PI / 4.0; // Example angle

    // Instantiate macro for each single-qubit gate
    generate_single_qubit_gate_error_assertions!(state, h);
    generate_single_qubit_gate_error_assertions!(state, x);
    generate_single_qubit_gate_error_assertions!(state, y);
    generate_single_qubit_gate_error_assertions!(state, z);
    generate_single_qubit_gate_error_assertions!(state, s);
    generate_single_qubit_gate_error_assertions!(state, t);
    generate_single_qubit_gate_error_assertions!(state, s_dag);
    generate_single_qubit_gate_error_assertions!(state, t_dag);
    generate_single_qubit_gate_error_assertions!(state, i);

    // Instantiate for gates requiring an angle
    generate_single_qubit_gate_error_assertions!(state, p, angle);
    generate_single_qubit_gate_error_assertions!(state, rx, angle);
    generate_single_qubit_gate_error_assertions!(state, ry, angle);
    generate_single_qubit_gate_error_assertions!(state, rz, angle);
}

#[test]
fn test_multi_qubit_gate_errors() {
    let num_qubits = 3;
    let invalid_index = 3; // Index 3 is out of bounds for a 3-qubit state (0, 1, 2)
    let state = State::new_zero(num_qubits).unwrap();

    // --- CNOT Gate Index Errors ---
    // Invalid control index
    let cnot_invalid_control = state.cnot(invalid_index, 0);
    assert!(
        matches!(
            cnot_invalid_control,
            Err(Error::InvalidQubitIndex(_invalid_index, _num_qubits))
        ),
        "CNOT failed (invalid control): Expected InvalidQubitIndex({}, {}), got {:?}",
        invalid_index,
        num_qubits,
        cnot_invalid_control
    );
    // Invalid target index
    let cnot_invalid_target = state.cnot(0, invalid_index);
    assert!(
        matches!(
            cnot_invalid_target,
            Err(Error::InvalidQubitIndex(_invalid_index, _num_qubits))
        ),
        "CNOT failed (invalid target): Expected InvalidQubitIndex({}, {}), got {:?}",
        invalid_index,
        num_qubits,
        cnot_invalid_target
    );

    // --- SWAP Gate Index Errors ---
    // Invalid first qubit index
    let swap_invalid_q1 = state.swap(invalid_index, 1);
    assert!(
        matches!(
            swap_invalid_q1,
            Err(Error::InvalidQubitIndex(_invalid_index, _num_qubits))
        ),
        "SWAP failed (invalid qubit1): Expected InvalidQubitIndex({}, {}), got {:?}",
        invalid_index,
        num_qubits,
        swap_invalid_q1
    );
    // Invalid second qubit index
    let swap_invalid_q2 = state.swap(0, invalid_index);
    assert!(
        matches!(
            swap_invalid_q2,
            Err(Error::InvalidQubitIndex(_invalid_index, _num_qubits))
        ),
        "SWAP failed (invalid qubit2): Expected InvalidQubitIndex({}, {}), got {:?}",
        invalid_index,
        num_qubits,
        swap_invalid_q2
    );

    // --- Toffoli Gate Index Errors ---
    // Invalid first control index
    let toffoli_invalid_c1 = state.toffoli(invalid_index, 1, 2);
    assert!(
        matches!(
            toffoli_invalid_c1,
            Err(Error::InvalidQubitIndex(_invalid_index, _num_qubits))
        ),
        "Toffoli failed (invalid control1): Expected InvalidQubitIndex({}, {}), got {:?}",
        invalid_index,
        num_qubits,
        toffoli_invalid_c1
    );
    // Invalid second control index
    let toffoli_invalid_c2 = state.toffoli(0, invalid_index, 2);
    assert!(
        matches!(
            toffoli_invalid_c2,
            Err(Error::InvalidQubitIndex(_invalid_index, _num_qubits))
        ),
        "Toffoli failed (invalid control2): Expected InvalidQubitIndex({}, {}), got {:?}",
        invalid_index,
        num_qubits,
        toffoli_invalid_c2
    );
    // Invalid target index
    let toffoli_invalid_target = state.toffoli(0, 1, invalid_index);
    assert!(
        matches!(
            toffoli_invalid_target,
            Err(Error::InvalidQubitIndex(_invalid_index, _num_qubits))
        ),
        "Toffoli failed (invalid target): Expected InvalidQubitIndex({}, {}), got {:?}",
        invalid_index,
        num_qubits,
        toffoli_invalid_target
    );
}
