use crate::{
    components::{
        ChainableState,
        operator::{
            CNOT, Hadamard, Identity, Operator, Pauli, PhaseS, PhaseSdag, PhaseShift, PhaseT,
            PhaseTdag, RotateX, RotateY, RotateZ, SWAP, Toffoli, Unitary2
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

    // ch(control = 0, target = 1)|+1> = |01>
    let state: State = State::new_plus(1).unwrap().tensor_product(&State::new_basis_n(1, 1).unwrap()).unwrap();
    let new_state: State = state.ch_multi(&[1], &[0]).unwrap();
    let expected_state: State = State::new_basis_n(2, 1).unwrap();
    assert_eq!(new_state, expected_state);

    // ch(control = 0, target = 1)|+0> = |+0>
    let state: State = State::new_basis_n(2, 0).unwrap().tensor_product(&State::new_plus(1).unwrap()).unwrap();
    let new_state: State = state.ch_multi(&[0], &[1]).unwrap();
    let expected_state: State = state.clone();
    assert_eq!(new_state, expected_state);

    // h(|0...0>) = |+...+> on 11 qubits [check if parallel method works (uncontrolled, multi-target)]
    let num_test_qubits_h_unc = 11;
    let state_h_unc = State::new_zero(num_test_qubits_h_unc).unwrap();
    let targets_h_unc: Vec<usize> = (0..num_test_qubits_h_unc).collect();
    let new_state_h_unc = state_h_unc.h_multi(&targets_h_unc).unwrap();
    let expected_state_h_unc = State::new_plus(num_test_qubits_h_unc).unwrap();
    assert_eq!(new_state_h_unc, expected_state_h_unc, "Uncontrolled H parallel (multi-target) failed");

    // ch(control=10, target=0) on 11 qubits. [check if controlled parallel method works]
    let num_test_qubits_ch = 11;
    let control_q_ch = &[num_test_qubits_ch - 1]; // qubit 10
    let target_q_ch = &[0]; // qubit 0

    // Case 1: Control is 0 (|0...0>)
    let initial_state_ch_c0 = State::new_zero(num_test_qubits_ch).unwrap();
    let new_state_ch_c0 = initial_state_ch_c0.ch_multi(target_q_ch, control_q_ch).unwrap();
    assert_eq!(new_state_ch_c0, initial_state_ch_c0, "Controlled H (c=0) parallel failed");

    // Case 2: Control is 1 (|...1...0...>)
    let mut initial_vec_ch_c1 = vec![Complex::new(0.0, 0.0); 1 << num_test_qubits_ch];
    let idx_ch_c1_t0 = 1 << control_q_ch[0]; // Control is 1 (at q10), target is 0 (at q0)
    initial_vec_ch_c1[idx_ch_c1_t0] = Complex::new(1.0, 0.0);
    let initial_state_ch_c1 = State { state_vector: initial_vec_ch_c1, num_qubits: num_test_qubits_ch };
    let new_state_ch_c1 = initial_state_ch_c1.ch_multi(target_q_ch, control_q_ch).unwrap();

    let mut expected_vec_ch_c1 = vec![Complex::new(0.0, 0.0); 1 << num_test_qubits_ch];
    let sqrt2_inv_h = Complex::new(1.0 / 2.0_f64.sqrt(), 0.0);
    expected_vec_ch_c1[idx_ch_c1_t0] = sqrt2_inv_h; // Control=1, Target=0 component of H|0>
    expected_vec_ch_c1[idx_ch_c1_t0 | (1 << target_q_ch[0])] = sqrt2_inv_h; // Control=1, Target=1 component of H|0>
    let expected_state_ch_c1 = State { state_vector: expected_vec_ch_c1, num_qubits: num_test_qubits_ch };
    assert_eq!(new_state_ch_c1, expected_state_ch_c1, "Controlled H (c=1) parallel failed");

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

    // cx(control = 0, target = 1)|11> = |01>
    let state: State = State::new_basis_n(2, 3).unwrap(); // |11>
    let new_state: State = state.cx_multi(&[1], &[0]).unwrap();
    let expected_state: State = State::new_basis_n(2, 1).unwrap(); // |01>
    assert_eq!(new_state, expected_state);

    // cx(control = 0, 1, target = 2)|011> = |111>
    let state: State = State::new_basis_n(3, 3).unwrap(); // |011>
    let new_state: State = state.cx_multi(&[2], &[0, 1]).unwrap();
    let expected_state: State = State::new_basis_n(3, 7).unwrap(); // |111>
    assert_eq!(new_state, expected_state);

    // cx(control = 0, target = 1)|10> = |10>
    let state: State = State::new_basis_n(2, 2).unwrap(); // |10>
    let new_state: State = state.cx_multi(&[1], &[0]).unwrap(); // |1> * |0> = |10>
    let expected_state: State = state.clone(); // |10>
    assert_eq!(new_state, expected_state);

    // Base qubits = 1
    assert_eq!(Pauli::X.base_qubits(), 1);

    // x_multi on 11 qubits [check if parallel method works (uncontrolled, multi-target)]
    let num_test_qubits_x_unc = 11;
    let initial_state_x_unc = State::new_zero(num_test_qubits_x_unc).unwrap();
    let targets_x_unc: Vec<usize> = (0..num_test_qubits_x_unc).collect();
    let new_state_x_unc = initial_state_x_unc.x_multi(&targets_x_unc).unwrap();
    let expected_state_x_unc = State::new_basis_n(num_test_qubits_x_unc, (1 << num_test_qubits_x_unc) - 1).unwrap(); // |1...1>
    assert_eq!(new_state_x_unc, expected_state_x_unc, "Uncontrolled X parallel (multi-target) failed");

    // cx(control=10, target=0) on 11 qubits. [check if controlled parallel method works]
    let num_test_qubits_cx = 11;
    let control_q_cx = &[num_test_qubits_cx - 1]; // qubit 10
    let target_q_cx = &[0]; // qubit 0

    // Case 1: Control is 0
    let initial_state_cx_c0 = State::new_zero(num_test_qubits_cx).unwrap();
    let new_state_cx_c0 = initial_state_cx_c0.cx_multi(target_q_cx, control_q_cx).unwrap();
    assert_eq!(new_state_cx_c0, initial_state_cx_c0, "Controlled X (c=0) parallel failed");

    // Case 2: Control is 1
    let mut initial_vec_cx_c1 = vec![Complex::new(0.0, 0.0); 1 << num_test_qubits_cx];
    let idx_cx_c1_t0 = 1 << control_q_cx[0]; // Control=1, Target=0
    initial_vec_cx_c1[idx_cx_c1_t0] = Complex::new(1.0, 0.0);
    let initial_state_cx_c1 = State { state_vector: initial_vec_cx_c1, num_qubits: num_test_qubits_cx };
    let new_state_cx_c1 = initial_state_cx_c1.cx_multi(target_q_cx, control_q_cx).unwrap();

    let mut expected_vec_cx_c1 = vec![Complex::new(0.0, 0.0); 1 << num_test_qubits_cx];
    expected_vec_cx_c1[idx_cx_c1_t0 | (1 << target_q_cx[0])] = Complex::new(1.0, 0.0); // Control=1, Target=1 (X|0> = |1>)
    let expected_state_cx_c1 = State { state_vector: expected_vec_cx_c1, num_qubits: num_test_qubits_cx };
    assert_eq!(new_state_cx_c1, expected_state_cx_c1, "Controlled X (c=1) parallel failed");
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

    // cy(control = 0, target = 1)|11> = -i|0> * |1> = -i|01>
    let state: State = State::new_basis_n(2, 3).unwrap(); // |11>
    let new_state: State = state.cy_multi(&[1], &[0]).unwrap(); // Y(|1>) * |1>
    let expected_state: State = (State::new_zero(1).unwrap() * neg_i).tensor_product(&State::new_basis_n(1, 1).unwrap()).unwrap(); // -i|0> * |1> = - i|01>
    assert_eq!(new_state, expected_state);

    // cy(control = 0, target = 1)|10> = |1> * |0> = |10>
    let state: State = State::new_basis_n(2, 2).unwrap(); // |10>
    let new_state: State = state.cy_multi(&[1], &[0]).unwrap();
    let expected_state: State = state.clone(); // |10>
    assert_eq!(new_state, expected_state);
    
    // Base qubits = 1
    assert_eq!(Pauli::Y.base_qubits(), 1);

    // y_multi on 11 qubits [check if parallel method works (uncontrolled, multi-target)]
    let num_test_qubits_y_unc = 11;
    let initial_state_y_unc = State::new_zero(num_test_qubits_y_unc).unwrap();
    let targets_y_unc: Vec<usize> = (0..num_test_qubits_y_unc).collect();
    let new_state_y_unc = initial_state_y_unc.y_multi(&targets_y_unc).unwrap();
    // Y|0> = i|1>. For Y applied to N |0> qubits, result is (i)^N |1...1>
    let i_complex_y = Complex::new(0.0, 1.0);
    let phase_factor_y = i_complex_y.powu(num_test_qubits_y_unc as u32);
    let basis_one_state_y = State::new_basis_n(num_test_qubits_y_unc, (1 << num_test_qubits_y_unc) - 1).unwrap();
    let expected_state_y_unc = basis_one_state_y * phase_factor_y;
    assert_eq!(new_state_y_unc, expected_state_y_unc, "Uncontrolled Y parallel (multi-target) failed");

    // cy(control=10, target=0) on 11 qubits. [check if controlled parallel method works]
    let num_test_qubits_cy = 11;
    let control_q_cy = &[num_test_qubits_cy - 1]; // qubit 10
    let target_q_cy = &[0]; // qubit 0

    // Case 1: Control is 0
    let initial_state_cy_c0 = State::new_zero(num_test_qubits_cy).unwrap();
    let new_state_cy_c0 = initial_state_cy_c0.cy_multi(target_q_cy, control_q_cy).unwrap();
    assert_eq!(new_state_cy_c0, initial_state_cy_c0, "Controlled Y (c=0) parallel failed");

    // Case 2: Control is 1
    let mut initial_vec_cy_c1 = vec![Complex::new(0.0, 0.0); 1 << num_test_qubits_cy];
    let idx_cy_c1_t0 = 1 << control_q_cy[0]; // Control=1, Target=0
    initial_vec_cy_c1[idx_cy_c1_t0] = Complex::new(1.0, 0.0);
    let initial_state_cy_c1 = State { state_vector: initial_vec_cy_c1, num_qubits: num_test_qubits_cy };
    let new_state_cy_c1 = initial_state_cy_c1.cy_multi(target_q_cy, control_q_cy).unwrap();

    // Y|0> = i|1>
    let mut expected_vec_cy_c1 = vec![Complex::new(0.0, 0.0); 1 << num_test_qubits_cy];
    expected_vec_cy_c1[idx_cy_c1_t0 | (1 << target_q_cy[0])] = i_complex_y; // Control=1, Target=1 component of Y|0>
    let expected_state_cy_c1 = State { state_vector: expected_vec_cy_c1, num_qubits: num_test_qubits_cy };
    assert_eq!(new_state_cy_c1, expected_state_cy_c1, "Controlled Y (c=1) parallel failed");
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

    // cz(control = 0, target = 1)|11> = -|1> * |1> = -|11>
    let state: State = State::new_basis_n(2, 3).unwrap(); // |11>
    let new_state: State = state.cz_multi(&[1], &[0]).unwrap(); // Z(|1>) * |1>
    let expected_state: State = (State::new_basis_n(1, 1).unwrap() * -1.0).tensor_product(&State::new_basis_n(1, 1).unwrap()).unwrap(); // -|1> * |1> = -|11>
    assert_eq!(new_state, expected_state);

    // cz(control = 0, target = 1)|10> = |10>
    let state: State = State::new_basis_n(2, 2).unwrap(); // |10>
    let new_state: State = state.cz_multi(&[1], &[0]).unwrap(); // Z(|0>) * |1>
    let expected_state: State = state.clone(); // |10>
    assert_eq!(new_state, expected_state);

    // Base qubits = 1
    assert_eq!(Pauli::Z.base_qubits(), 1);

    // z_multi on 11 qubits |0...0> [check if parallel method works (uncontrolled, multi-target)]
    // Z|0> = |0>, so Z applied to |0...0> is |0...0>
    let num_test_qubits_z_unc = 11;
    let initial_state_z_unc = State::new_zero(num_test_qubits_z_unc).unwrap();
    let targets_z_unc: Vec<usize> = (0..num_test_qubits_z_unc).collect();
    let new_state_z_unc = initial_state_z_unc.z_multi(&targets_z_unc).unwrap();
    assert_eq!(new_state_z_unc, initial_state_z_unc, "Uncontrolled Z parallel (multi-target) failed on |0...0>");

    // cz(control=10, target=0) on 11 qubits. [check if controlled parallel method works]
    let num_test_qubits_cz = 11;
    let control_q_cz = &[num_test_qubits_cz - 1]; // qubit 10
    let target_q_cz = &[0]; // qubit 0

    // Case 1: Control is 0 (|...0...0>)
    let initial_state_cz_c0_t0 = State::new_zero(num_test_qubits_cz).unwrap(); // Target is |0>
    let new_state_cz_c0_t0 = initial_state_cz_c0_t0.cz_multi(target_q_cz, control_q_cz).unwrap();
    assert_eq!(new_state_cz_c0_t0, initial_state_cz_c0_t0, "Controlled Z (c=0, t=0) parallel failed");

    let mut initial_vec_cz_c0_t1 = vec![Complex::new(0.0,0.0); 1 << num_test_qubits_cz];
    initial_vec_cz_c0_t1[1 << target_q_cz[0]] = Complex::new(1.0,0.0); // Control 0, Target 1
    let initial_state_cz_c0_t1 = State { state_vector: initial_vec_cz_c0_t1, num_qubits: num_test_qubits_cz };
    let new_state_cz_c0_t1 = initial_state_cz_c0_t1.cz_multi(target_q_cz, control_q_cz).unwrap();
    assert_eq!(new_state_cz_c0_t1, initial_state_cz_c0_t1, "Controlled Z (c=0, t=1) parallel failed");


    // Case 2: Control is 1
    // Target |0>: CZ|0> = |0>
    let mut initial_vec_cz_c1_t0 = vec![Complex::new(0.0, 0.0); 1 << num_test_qubits_cz];
    let idx_cz_c1_t0 = 1 << control_q_cz[0]; // Control=1, Target=0
    initial_vec_cz_c1_t0[idx_cz_c1_t0] = Complex::new(1.0, 0.0);
    let initial_state_cz_c1_t0 = State { state_vector: initial_vec_cz_c1_t0, num_qubits: num_test_qubits_cz };
    let new_state_cz_c1_t0 = initial_state_cz_c1_t0.cz_multi(target_q_cz, control_q_cz).unwrap();
    assert_eq!(new_state_cz_c1_t0, initial_state_cz_c1_t0, "Controlled Z (c=1, t=0) parallel failed");

    // Target |1>: CZ|1> = -|1>
    let mut initial_vec_cz_c1_t1 = vec![Complex::new(0.0, 0.0); 1 << num_test_qubits_cz];
    let idx_cz_c1_t1 = (1 << control_q_cz[0]) | (1 << target_q_cz[0]); // Control=1, Target=1
    initial_vec_cz_c1_t1[idx_cz_c1_t1] = Complex::new(1.0, 0.0);
    let initial_state_cz_c1_t1 = State { state_vector: initial_vec_cz_c1_t1, num_qubits: num_test_qubits_cz };
    let new_state_cz_c1_t1 = initial_state_cz_c1_t1.cz_multi(target_q_cz, control_q_cz).unwrap();

    let mut expected_vec_cz_c1_t1 = vec![Complex::new(0.0, 0.0); 1 << num_test_qubits_cz];
    expected_vec_cz_c1_t1[idx_cz_c1_t1] = Complex::new(-1.0, 0.0); // Phase flip
    let expected_state_cz_c1_t1 = State { state_vector: expected_vec_cz_c1_t1, num_qubits: num_test_qubits_cz };
    assert_eq!(new_state_cz_c1_t1, expected_state_cz_c1_t1, "Controlled Z (c=1, t=1) parallel failed");
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

    // i_multi on 11 qubits |0...0> [check if parallel method works (uncontrolled, multi-target)]
    // I|psi> = |psi>, so I applied to |0...0> is |0...0>
    let num_test_qubits_i_unc = 11;
    let initial_state_i_unc = State::new_zero(num_test_qubits_i_unc).unwrap();
    let targets_i_unc: Vec<usize> = (0..num_test_qubits_i_unc).collect();
    let new_state_i_unc = initial_state_i_unc.i_multi(&targets_i_unc).unwrap();
    assert_eq!(new_state_i_unc, initial_state_i_unc, "Uncontrolled I parallel (multi-target) failed");

    // ci (controlled identity) on 11 qubits. [check if controlled parallel method works]
    // Controlled Identity is always identity.
    let num_test_qubits_ci = 11;
    let control_q_ci = &[num_test_qubits_ci - 1]; 
    let target_q_ci = &[0]; 

    // Case 1: Control is 0
    let initial_state_ci_c0 = State::new_zero(num_test_qubits_ci).unwrap();
    let new_state_ci_c0 = initial_state_ci_c0.ci_multi(target_q_ci, control_q_ci).unwrap();
    assert_eq!(new_state_ci_c0, initial_state_ci_c0, "Controlled I (c=0) parallel failed");

    // Case 2: Control is 1
    let mut initial_vec_ci_c1 = vec![Complex::new(0.0, 0.0); 1 << num_test_qubits_ci];
    let idx_ci_c1_t0 = 1 << control_q_ci[0]; 
    initial_vec_ci_c1[idx_ci_c1_t0] = Complex::new(1.0, 0.0);
    let initial_state_ci_c1 = State { state_vector: initial_vec_ci_c1, num_qubits: num_test_qubits_ci };
    let new_state_ci_c1 = initial_state_ci_c1.ci_multi(target_q_ci, control_q_ci).unwrap();
    assert_eq!(new_state_ci_c1, initial_state_ci_c1, "Controlled I (c=1) parallel failed");
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

    // cs(control = 0, target = 1)|+1> = (|0>/sqrt2 + i|1>/sqrt2)|1>
    let state: State = State::new_plus(1).unwrap().tensor_product(&State::new_basis_n(1, 1).unwrap()).unwrap();
    let new_state: State = state.cs_multi(&[1], &[0]).unwrap();
    let invrt2: Complex<f64> = Complex::new(1.0 / 2.0_f64.sqrt(), 0.0);
    let expected_state: State = (invrt2 * (State::new_zero(1).unwrap() + i * State::new_basis_n(1, 1).unwrap())).tensor_product(&State::new_basis_n(1, 1).unwrap()).unwrap(); // (|0>/sqrt2 + i|1>/sqrt2)|1>
    assert_eq!(new_state, expected_state);

    // cs(control = 0, target = 1)|+0> = |+0>
    let state: State = State::new_basis_n(2, 0).unwrap().tensor_product(&State::new_plus(1).unwrap()).unwrap();
    let new_state: State = state.cs_multi(&[0], &[1]).unwrap();
    let expected_state: State = state.clone(); // |+0>
    assert_eq!(new_state, expected_state);

    // Base qubits = 1
    assert_eq!(PhaseS {}.base_qubits(), 1);

    // s_multi on 11 qubits |0...0> [check if parallel method works (uncontrolled, multi-target)]
    // S|0> = |0>, so S applied to |0...0> is |0...0>
    let num_test_qubits_s_unc = 11;
    let initial_state_s_unc = State::new_zero(num_test_qubits_s_unc).unwrap();
    let targets_s_unc: Vec<usize> = (0..num_test_qubits_s_unc).collect();
    let new_state_s_unc = initial_state_s_unc.s_multi(&targets_s_unc).unwrap();
    assert_eq!(new_state_s_unc, initial_state_s_unc, "Uncontrolled S parallel (multi-target) on |0...0> failed");

    // cs(control=10, target=0) on 11 qubits. [check if controlled parallel method works]
    let num_test_qubits_cs = 11;
    let control_q_cs = &[num_test_qubits_cs - 1]; // qubit 10
    let target_q_cs = &[0]; // qubit 0

    // Case 1: Control is 0
    let initial_state_cs_c0_t0 = State::new_zero(num_test_qubits_cs).unwrap(); // Target |0>
    let new_state_cs_c0_t0 = initial_state_cs_c0_t0.cs_multi(target_q_cs, control_q_cs).unwrap();
    assert_eq!(new_state_cs_c0_t0, initial_state_cs_c0_t0, "Controlled S (c=0, t=0) parallel failed");
    
    let mut initial_vec_cs_c0_t1 = vec![Complex::new(0.0,0.0); 1 << num_test_qubits_cs];
    initial_vec_cs_c0_t1[1 << target_q_cs[0]] = Complex::new(1.0,0.0); // Control 0, Target 1
    let initial_state_cs_c0_t1 = State { state_vector: initial_vec_cs_c0_t1, num_qubits: num_test_qubits_cs };
    let new_state_cs_c0_t1 = initial_state_cs_c0_t1.cs_multi(target_q_cs, control_q_cs).unwrap();
    assert_eq!(new_state_cs_c0_t1, initial_state_cs_c0_t1, "Controlled S (c=0, t=1) parallel failed");


    // Case 2: Control is 1
    // Target |0>: CS|0> = |0>
    let mut initial_vec_cs_c1_t0 = vec![Complex::new(0.0, 0.0); 1 << num_test_qubits_cs];
    let idx_cs_c1_t0 = 1 << control_q_cs[0]; // Control=1, Target=0
    initial_vec_cs_c1_t0[idx_cs_c1_t0] = Complex::new(1.0, 0.0);
    let initial_state_cs_c1_t0 = State { state_vector: initial_vec_cs_c1_t0, num_qubits: num_test_qubits_cs };
    let new_state_cs_c1_t0 = initial_state_cs_c1_t0.cs_multi(target_q_cs, control_q_cs).unwrap();
    assert_eq!(new_state_cs_c1_t0, initial_state_cs_c1_t0, "Controlled S (c=1, t=0) parallel failed");

    // Target |1>: CS|1> = i|1>
    let mut initial_vec_cs_c1_t1 = vec![Complex::new(0.0, 0.0); 1 << num_test_qubits_cs];
    let idx_cs_c1_t1 = (1 << control_q_cs[0]) | (1 << target_q_cs[0]); // Control=1, Target=1
    initial_vec_cs_c1_t1[idx_cs_c1_t1] = Complex::new(1.0, 0.0);
    let initial_state_cs_c1_t1 = State { state_vector: initial_vec_cs_c1_t1, num_qubits: num_test_qubits_cs };
    let new_state_cs_c1_t1 = initial_state_cs_c1_t1.cs_multi(target_q_cs, control_q_cs).unwrap();

    let mut expected_vec_cs_c1_t1 = vec![Complex::new(0.0, 0.0); 1 << num_test_qubits_cs];
    expected_vec_cs_c1_t1[idx_cs_c1_t1] = Complex::new(0.0, 1.0); // i factor
    let expected_state_cs_c1_t1 = State { state_vector: expected_vec_cs_c1_t1, num_qubits: num_test_qubits_cs };
    assert_eq!(new_state_cs_c1_t1, expected_state_cs_c1_t1, "Controlled S (c=1, t=1) parallel failed");
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

    // ct(control = 0, target = 1)|+1> = (|0>/sqrt2 + e^(ipi/4)|1>/sqrt2)|1>
    let state: State = State::new_plus(1).unwrap().tensor_product(&State::new_basis_n(1, 1).unwrap()).unwrap();
    let new_state: State = state.ct_multi(&[1], &[0]).unwrap();
    let invrt2: Complex<f64> = Complex::new(1.0 / 2.0_f64.sqrt(), 0.0);
    let expected_state: State = (invrt2 * (State::new_zero(1).unwrap() + eipi4 * State::new_basis_n(1, 1).unwrap())).tensor_product(&State::new_basis_n(1, 1).unwrap()).unwrap(); // (|0>/sqrt2 + i|1>/sqrt2)|1>
    assert_eq!(new_state, expected_state);

    // ct(control = 0, target = 1)|+0> = |+0>
    let state: State = State::new_basis_n(2, 0).unwrap().tensor_product(&State::new_plus(1).unwrap()).unwrap();
    let new_state: State = state.ct_multi(&[0], &[1]).unwrap();
    let expected_state: State = state.clone(); // |+0>
    assert_eq!(new_state, expected_state);

    // Base qubits = 1
    assert_eq!(PhaseT {}.base_qubits(), 1);

    // t_multi on 11 qubits |0...0> [check if parallel method works (uncontrolled, multi-target)]
    // T|0> = |0>, so T applied to |0...0> is |0...0>
    let num_test_qubits_t_unc = 11;
    let initial_state_t_unc = State::new_zero(num_test_qubits_t_unc).unwrap();
    let targets_t_unc: Vec<usize> = (0..num_test_qubits_t_unc).collect();
    let new_state_t_unc = initial_state_t_unc.t_multi(&targets_t_unc).unwrap();
    assert_eq!(new_state_t_unc, initial_state_t_unc, "Uncontrolled T parallel (multi-target) on |0...0> failed");

    // ct(control=10, target=0) on 11 qubits. [check if controlled parallel method works]
    let num_test_qubits_ct = 11;
    let control_q_ct = &[num_test_qubits_ct - 1]; // qubit 10
    let target_q_ct = &[0]; // qubit 0

    // Case 1: Control is 0
    let initial_state_ct_c0_t0 = State::new_zero(num_test_qubits_ct).unwrap(); // Target |0>
    let new_state_ct_c0_t0 = initial_state_ct_c0_t0.ct_multi(target_q_ct, control_q_ct).unwrap();
    assert_eq!(new_state_ct_c0_t0, initial_state_ct_c0_t0, "Controlled T (c=0, t=0) parallel failed");

    let mut initial_vec_ct_c0_t1 = vec![Complex::new(0.0,0.0); 1 << num_test_qubits_ct];
    initial_vec_ct_c0_t1[1 << target_q_ct[0]] = Complex::new(1.0,0.0); // Control 0, Target 1
    let initial_state_ct_c0_t1 = State { state_vector: initial_vec_ct_c0_t1, num_qubits: num_test_qubits_ct };
    let new_state_ct_c0_t1 = initial_state_ct_c0_t1.ct_multi(target_q_ct, control_q_ct).unwrap();
    assert_eq!(new_state_ct_c0_t1, initial_state_ct_c0_t1, "Controlled T (c=0, t=1) parallel failed");

    // Case 2: Control is 1
    // Target |0>: CT|0> = |0>
    let mut initial_vec_ct_c1_t0 = vec![Complex::new(0.0, 0.0); 1 << num_test_qubits_ct];
    let idx_ct_c1_t0 = 1 << control_q_ct[0]; // Control=1, Target=0
    initial_vec_ct_c1_t0[idx_ct_c1_t0] = Complex::new(1.0, 0.0);
    let initial_state_ct_c1_t0 = State { state_vector: initial_vec_ct_c1_t0, num_qubits: num_test_qubits_ct };
    let new_state_ct_c1_t0 = initial_state_ct_c1_t0.ct_multi(target_q_ct, control_q_ct).unwrap();
    assert_eq!(new_state_ct_c1_t0, initial_state_ct_c1_t0, "Controlled T (c=1, t=0) parallel failed");

    // Target |1>: CT|1> = e^(i*PI/4)|1>
    let mut initial_vec_ct_c1_t1 = vec![Complex::new(0.0, 0.0); 1 << num_test_qubits_ct];
    let idx_ct_c1_t1 = (1 << control_q_ct[0]) | (1 << target_q_ct[0]); // Control=1, Target=1
    initial_vec_ct_c1_t1[idx_ct_c1_t1] = Complex::new(1.0, 0.0);
    let initial_state_ct_c1_t1 = State { state_vector: initial_vec_ct_c1_t1, num_qubits: num_test_qubits_ct };
    let new_state_ct_c1_t1 = initial_state_ct_c1_t1.ct_multi(target_q_ct, control_q_ct).unwrap();

    let mut expected_vec_ct_c1_t1 = vec![Complex::new(0.0, 0.0); 1 << num_test_qubits_ct];
    let eipi4_ct = Complex::new(0.0, PI / 4.0).exp();
    expected_vec_ct_c1_t1[idx_ct_c1_t1] = eipi4_ct;
    let expected_state_ct_c1_t1 = State { state_vector: expected_vec_ct_c1_t1, num_qubits: num_test_qubits_ct };
    assert_eq!(new_state_ct_c1_t1, expected_state_ct_c1_t1, "Controlled T (c=1, t=1) parallel failed");
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

    // csdag(control = 0, target = 1)|+1> = (|0>/sqrt2 - i|1>/sqrt2)|1>
    let state: State = State::new_plus(1).unwrap().tensor_product(&State::new_basis_n(1, 1).unwrap()).unwrap();
    let new_state: State = state.cs_dag_multi(&[1], &[0]).unwrap();
    let invrt2: Complex<f64> = Complex::new(1.0 / 2.0_f64.sqrt(), 0.0);
    let expected_state: State = (invrt2 * (State::new_zero(1).unwrap() - i * State::new_basis_n(1, 1).unwrap())).tensor_product(&State::new_basis_n(1, 1).unwrap()).unwrap(); // (|0>/sqrt2 - i|1>/sqrt2)|1>
    assert_eq!(new_state, expected_state);

    // csdag(control = 0, target = 1)|+0> = |+0>
    let state: State = State::new_basis_n(2, 0).unwrap().tensor_product(&State::new_plus(1).unwrap()).unwrap();
    let new_state: State = state.cs_dag_multi(&[0], &[1]).unwrap();
    let expected_state: State = state.clone(); // |+0>
    assert_eq!(new_state, expected_state);

    // Base qubits = 1
    assert_eq!(PhaseSdag {}.base_qubits(), 1);

    // s_dag_multi on 11 qubits |0...0> [check if parallel method works (uncontrolled, multi-target)]
    let num_test_qubits_sdag_unc = 11;
    let initial_state_sdag_unc = State::new_zero(num_test_qubits_sdag_unc).unwrap();
    let targets_sdag_unc: Vec<usize> = (0..num_test_qubits_sdag_unc).collect();
    let new_state_sdag_unc = initial_state_sdag_unc.s_dag_multi(&targets_sdag_unc).unwrap();
    assert_eq!(new_state_sdag_unc, initial_state_sdag_unc, "Uncontrolled Sdag parallel (multi-target) on |0...0> failed");

    // cs_dag(control=10, target=0) on 11 qubits. [check if controlled parallel method works]
    let num_test_qubits_csdag = 11;
    let control_q_csdag = &[num_test_qubits_csdag - 1];
    let target_q_csdag = &[0];

    // Case 1: Control is 0
    let initial_state_csdag_c0_t0 = State::new_zero(num_test_qubits_csdag).unwrap();
    let new_state_csdag_c0_t0 = initial_state_csdag_c0_t0.cs_dag_multi(target_q_csdag, control_q_csdag).unwrap();
    assert_eq!(new_state_csdag_c0_t0, initial_state_csdag_c0_t0, "Controlled Sdag (c=0, t=0) parallel failed");

    // Case 2: Control is 1
    // Target |1>: CSdag|1> = -i|1>
    let mut initial_vec_csdag_c1_t1 = vec![Complex::new(0.0, 0.0); 1 << num_test_qubits_csdag];
    let idx_csdag_c1_t1 = (1 << control_q_csdag[0]) | (1 << target_q_csdag[0]);
    initial_vec_csdag_c1_t1[idx_csdag_c1_t1] = Complex::new(1.0, 0.0);
    let initial_state_csdag_c1_t1 = State { state_vector: initial_vec_csdag_c1_t1, num_qubits: num_test_qubits_csdag };
    let new_state_csdag_c1_t1 = initial_state_csdag_c1_t1.cs_dag_multi(target_q_csdag, control_q_csdag).unwrap();
    let mut expected_vec_csdag_c1_t1 = vec![Complex::new(0.0, 0.0); 1 << num_test_qubits_csdag];
    expected_vec_csdag_c1_t1[idx_csdag_c1_t1] = Complex::new(0.0, -1.0); // -i factor
    let expected_state_csdag_c1_t1 = State { state_vector: expected_vec_csdag_c1_t1, num_qubits: num_test_qubits_csdag };
    assert_eq!(new_state_csdag_c1_t1, expected_state_csdag_c1_t1, "Controlled Sdag (c=1, t=1) parallel failed");
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

    // ctdag(control = 0, target = 1)|+1> = (|0>/sqrt2 + e^(-i*PI/4)|1>/sqrt2)|1>
    let state: State = State::new_plus(1).unwrap().tensor_product(&State::new_basis_n(1, 1).unwrap()).unwrap();
    let new_state: State = state.ct_dag_multi(&[1], &[0]).unwrap();
    let invrt2: Complex<f64> = Complex::new(1.0 / 2.0_f64.sqrt(), 0.0);
    let expected_state: State = (invrt2 * (State::new_zero(1).unwrap() + enegipi4 * State::new_basis_n(1, 1).unwrap())).tensor_product(&State::new_basis_n(1, 1).unwrap()).unwrap(); // (|0>/sqrt2 + e^(-i*PI/4)|1>/sqrt2)|1>
    assert_eq!(new_state, expected_state);

    // Base qubits = 1
    assert_eq!(PhaseTdag {}.base_qubits(), 1);

    // t_dag_multi on 11 qubits |0...0> [check if parallel method works (uncontrolled, multi-target)]
    let num_test_qubits_tdag_unc = 11;
    let initial_state_tdag_unc = State::new_zero(num_test_qubits_tdag_unc).unwrap();
    let targets_tdag_unc: Vec<usize> = (0..num_test_qubits_tdag_unc).collect();
    let new_state_tdag_unc = initial_state_tdag_unc.t_dag_multi(&targets_tdag_unc).unwrap();
    assert_eq!(new_state_tdag_unc, initial_state_tdag_unc, "Uncontrolled Tdag parallel (multi-target) on |0...0> failed");

    // ct_dag(control=10, target=0) on 11 qubits. [check if controlled parallel method works]
    let num_test_qubits_ctdag = 11;
    let control_q_ctdag = &[num_test_qubits_ctdag - 1];
    let target_q_ctdag = &[0];

    // Case 1: Control is 0
    let initial_state_ctdag_c0_t0 = State::new_zero(num_test_qubits_ctdag).unwrap();
    let new_state_ctdag_c0_t0 = initial_state_ctdag_c0_t0.ct_dag_multi(target_q_ctdag, control_q_ctdag).unwrap();
    assert_eq!(new_state_ctdag_c0_t0, initial_state_ctdag_c0_t0, "Controlled Tdag (c=0, t=0) parallel failed");

    // Case 2: Control is 1
    // Target |1>: CTdag|1> = e^(-i*PI/4)|1>
    let mut initial_vec_ctdag_c1_t1 = vec![Complex::new(0.0, 0.0); 1 << num_test_qubits_ctdag];
    let idx_ctdag_c1_t1 = (1 << control_q_ctdag[0]) | (1 << target_q_ctdag[0]);
    initial_vec_ctdag_c1_t1[idx_ctdag_c1_t1] = Complex::new(1.0, 0.0);
    let initial_state_ctdag_c1_t1 = State { state_vector: initial_vec_ctdag_c1_t1, num_qubits: num_test_qubits_ctdag };
    let new_state_ctdag_c1_t1 = initial_state_ctdag_c1_t1.ct_dag_multi(target_q_ctdag, control_q_ctdag).unwrap();
    let mut expected_vec_ctdag_c1_t1 = vec![Complex::new(0.0, 0.0); 1 << num_test_qubits_ctdag];
    let enegipi4_ctdag = Complex::new(0.0, -PI / 4.0).exp();
    expected_vec_ctdag_c1_t1[idx_ctdag_c1_t1] = enegipi4_ctdag;
    let expected_state_ctdag_c1_t1 = State { state_vector: expected_vec_ctdag_c1_t1, num_qubits: num_test_qubits_ctdag };
    assert_eq!(new_state_ctdag_c1_t1, expected_state_ctdag_c1_t1, "Controlled Tdag (c=1, t=1) parallel failed");
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

    // cp(control = 0, target = 1, theta = pi/2)|+1> = (|0>/sqrt2 + i|1>/sqrt2)|1>
    let state: State = State::new_plus(1).unwrap().tensor_product(&State::new_basis_n(1, 1).unwrap()).unwrap();
    let new_state: State = state.cp_multi(&[1], &[0], PI / 2.0).unwrap();
    let invrt2: Complex<f64> = Complex::new(1.0 / 2.0_f64.sqrt(), 0.0);
    let expected_state: State = (invrt2 * (State::new_zero(1).unwrap() + Complex::new(0.0, PI / 2.0).exp() * State::new_basis_n(1, 1).unwrap())).tensor_product(&State::new_basis_n(1, 1).unwrap()).unwrap(); // (|0>/sqrt2 + i|1>/sqrt2)|1>
    assert_eq!(new_state, expected_state);

    // cp(control = 0, target = 1, theta = -pi/2)|+0> = |+0>
    let state: State = State::new_basis_n(2, 0).unwrap().tensor_product(&State::new_plus(1).unwrap()).unwrap();
    let new_state: State = state.cp_multi(&[0], &[1], -PI / 2.0).unwrap();
    let expected_state: State = state.clone(); // |+0>
    assert_eq!(new_state, expected_state);

    // Base qubits = 1
    assert_eq!(PhaseShift::new(theta).base_qubits(), 1);

    // p_multi on 11 qubits |0...0> with angle theta [check if parallel method works (uncontrolled, multi-target)]
    let num_test_qubits_p_unc = 11;
    let angle_p_unc = PI / 3.0;
    let initial_state_p_unc = State::new_zero(num_test_qubits_p_unc).unwrap();
    let targets_p_unc: Vec<usize> = (0..num_test_qubits_p_unc).collect();
    let new_state_p_unc = initial_state_p_unc.p_multi(&targets_p_unc, angle_p_unc).unwrap();
    assert_eq!(new_state_p_unc, initial_state_p_unc, "Uncontrolled PhaseShift parallel (multi-target) on |0...0> failed");

    // cp(control=10, target=0, angle=theta) on 11 qubits. [check if controlled parallel method works]
    let num_test_qubits_cp = 11;
    let angle_cp = PI / 5.0;
    let control_q_cp = &[num_test_qubits_cp - 1];
    let target_q_cp = &[0];

    // Case 1: Control is 0
    let initial_state_cp_c0_t0 = State::new_zero(num_test_qubits_cp).unwrap();
    let new_state_cp_c0_t0 = initial_state_cp_c0_t0.cp_multi(target_q_cp, control_q_cp, angle_cp).unwrap();
    assert_eq!(new_state_cp_c0_t0, initial_state_cp_c0_t0, "Controlled PhaseShift (c=0, t=0) parallel failed");

    // Case 2: Control is 1
    // Target |1>: CP|1> = e^(i*theta)|1>
    let mut initial_vec_cp_c1_t1 = vec![Complex::new(0.0, 0.0); 1 << num_test_qubits_cp];
    let idx_cp_c1_t1 = (1 << control_q_cp[0]) | (1 << target_q_cp[0]);
    initial_vec_cp_c1_t1[idx_cp_c1_t1] = Complex::new(1.0, 0.0);
    let initial_state_cp_c1_t1 = State { state_vector: initial_vec_cp_c1_t1, num_qubits: num_test_qubits_cp };
    let new_state_cp_c1_t1 = initial_state_cp_c1_t1.cp_multi(target_q_cp, control_q_cp, angle_cp).unwrap();
    let mut expected_vec_cp_c1_t1 = vec![Complex::new(0.0, 0.0); 1 << num_test_qubits_cp];
    let eitheta_cp = Complex::new(0.0, angle_cp).exp();
    expected_vec_cp_c1_t1[idx_cp_c1_t1] = eitheta_cp;
    let expected_state_cp_c1_t1 = State { state_vector: expected_vec_cp_c1_t1, num_qubits: num_test_qubits_cp };
    assert_eq!(new_state_cp_c1_t1, expected_state_cp_c1_t1, "Controlled PhaseShift (c=1, t=1) parallel failed");
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

    // crx(control = 0, target = 1, theta = pi)|11> = -i|0> * |1> = -i|0> * |1> = -i|01>
    let state: State = State::new_basis_n(2, 3).unwrap(); // |11>
    let new_state: State = state.crx_multi(&[1], &[0], PI).unwrap(); // RX(|1>) * |1>
    let expected_state: State = (State::new_zero(1).unwrap() * -i).tensor_product(&State::new_basis_n(1, 1).unwrap()).unwrap(); // -i|0> * |1> = -i|01>
    assert_eq!(new_state, expected_state);

    // crx(control = 0, target = 1, theta = -pi)|+0> = |+0>
    let state: State = State::new_basis_n(2, 0).unwrap().tensor_product(&State::new_plus(1).unwrap()).unwrap();
    let new_state: State = state.crx_multi(&[0], &[1], -PI).unwrap(); // RX(|0>) * |+0>
    let expected_state: State = state.clone(); // |+0>
    assert_eq!(new_state, expected_state);

    // Base qubits = 1
    assert_eq!(RotateX::new(theta).base_qubits(), 1);

    // rx_multi on 11 qubits |0...0> [check if parallel method works (uncontrolled, multi-target)]
    let num_test_qubits_rx_unc = 11;
    let angle_rx_unc = PI / 1.5;
    let initial_state_rx_unc = State::new_zero(num_test_qubits_rx_unc).unwrap();
    let targets_rx_unc: Vec<usize> = (0..num_test_qubits_rx_unc).collect();
    let new_state_rx_unc = initial_state_rx_unc.rx_multi(&targets_rx_unc, angle_rx_unc).unwrap();
    // Expected: (Rx(angle)|0>)^(\otimes N)
    let rx_on_zero_one_q = State::new_zero(1).unwrap().rx(0, angle_rx_unc).unwrap();
    let mut expected_state_rx_unc = rx_on_zero_one_q.clone();
    if num_test_qubits_rx_unc > 0 { // Should always be true here (11)
        for _ in 1..num_test_qubits_rx_unc {
            expected_state_rx_unc = expected_state_rx_unc.tensor_product(&rx_on_zero_one_q).unwrap();
        }
    } else {
        // Handle 0 qubit case if necessary, though num_test_qubits_rx_unc is 11
        // For 0 qubits, perhaps an empty state or error, but test setup implies >0
        panic!("num_test_qubits_rx_unc is 0, which is not expected for this test");
    }
    assert_eq!(new_state_rx_unc, expected_state_rx_unc, "Uncontrolled Rx parallel (multi-target) failed");

    // crx(control=10, target=0, angle=theta) on 11 qubits. [check if controlled parallel method works]
    let num_test_qubits_crx = 11;
    let angle_crx = PI / 2.2;
    let control_q_crx = &[num_test_qubits_crx - 1];
    let target_q_crx = &[0];

    // Case 1: Control is 0
    let initial_state_crx_c0 = State::new_zero(num_test_qubits_crx).unwrap();
    let new_state_crx_c0 = initial_state_crx_c0.crx_multi(target_q_crx, control_q_crx, angle_crx).unwrap();
    assert_eq!(new_state_crx_c0, initial_state_crx_c0, "Controlled Rx (c=0) parallel failed");

    // Case 2: Control is 1
    let mut initial_vec_crx_c1 = vec![Complex::new(0.0, 0.0); 1 << num_test_qubits_crx];
    let idx_crx_c1_t0 = 1 << control_q_crx[0];
    initial_vec_crx_c1[idx_crx_c1_t0] = Complex::new(1.0, 0.0);
    let initial_state_crx_c1 = State { state_vector: initial_vec_crx_c1, num_qubits: num_test_qubits_crx };
    let new_state_crx_c1 = initial_state_crx_c1.crx_multi(target_q_crx, control_q_crx, angle_crx).unwrap();
    
    let rx_on_zero_for_crx = State::new_zero(1).unwrap().rx(0, angle_crx).unwrap();
    let mut expected_vec_crx_c1 = vec![Complex::new(0.0, 0.0); 1 << num_test_qubits_crx];
    if rx_on_zero_for_crx.state_vector[0].norm_sqr() > 1e-9 {
        expected_vec_crx_c1[idx_crx_c1_t0] = rx_on_zero_for_crx.state_vector[0];
    }
    if rx_on_zero_for_crx.state_vector[1].norm_sqr() > 1e-9 {
        expected_vec_crx_c1[idx_crx_c1_t0 | (1 << target_q_crx[0])] = rx_on_zero_for_crx.state_vector[1];
    }
    let expected_state_crx_c1 = State { state_vector: expected_vec_crx_c1, num_qubits: num_test_qubits_crx };
    assert_eq!(new_state_crx_c1, expected_state_crx_c1, "Controlled Rx (c=1) parallel failed");
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

    // cry(control = 0, target = 1, theta = pi)|11> = -|0> * |1> = -|01>
    let state: State = State::new_basis_n(2, 3).unwrap(); // |11>
    let new_state: State = state.cry_multi(&[1], &[0], PI).unwrap(); // RY(|1>) * |1>
    let expected_state: State = (State::new_zero(1).unwrap() * -1.0).tensor_product(&State::new_basis_n(1, 1).unwrap()).unwrap(); // -|0> * |1> = -|01>
    assert_eq!(new_state, expected_state);

    // cry(control = 0, target = 1, theta = -pi)|+0> = |+0>
    let state: State = State::new_basis_n(2, 0).unwrap().tensor_product(&State::new_plus(1).unwrap()).unwrap();
    let new_state: State = state.cry_multi(&[0], &[1], -PI).unwrap();
    let expected_state: State = state.clone(); // |+0>
    assert_eq!(new_state, expected_state);

    // Base qubits = 1
    assert_eq!(RotateY::new(theta).base_qubits(), 1);

    // ry_multi on 11 qubits |0...0> [check if parallel method works (uncontrolled, multi-target)]
    let num_test_qubits_ry_unc = 11;
    let angle_ry_unc = PI / 1.8;
    let initial_state_ry_unc = State::new_zero(num_test_qubits_ry_unc).unwrap();
    let targets_ry_unc: Vec<usize> = (0..num_test_qubits_ry_unc).collect();
    let new_state_ry_unc = initial_state_ry_unc.ry_multi(&targets_ry_unc, angle_ry_unc).unwrap();
    let ry_on_zero_one_q = State::new_zero(1).unwrap().ry(0, angle_ry_unc).unwrap();
    let mut expected_state_ry_unc = ry_on_zero_one_q.clone();
    if num_test_qubits_ry_unc > 0 { // Should always be true here (11)
        for _ in 1..num_test_qubits_ry_unc {
            expected_state_ry_unc = expected_state_ry_unc.tensor_product(&ry_on_zero_one_q).unwrap();
        }
    } else {
        panic!("num_test_qubits_ry_unc is 0, which is not expected for this test");
    }
    assert_eq!(new_state_ry_unc, expected_state_ry_unc, "Uncontrolled Ry parallel (multi-target) failed");

    // cry(control=10, target=0, angle=theta) on 11 qubits. [check if controlled parallel method works]
    let num_test_qubits_cry = 11;
    let angle_cry = PI / 2.8;
    let control_q_cry = &[num_test_qubits_cry - 1];
    let target_q_cry = &[0];

    // Case 1: Control is 0
    let initial_state_cry_c0 = State::new_zero(num_test_qubits_cry).unwrap();
    let new_state_cry_c0 = initial_state_cry_c0.cry_multi(target_q_cry, control_q_cry, angle_cry).unwrap();
    assert_eq!(new_state_cry_c0, initial_state_cry_c0, "Controlled Ry (c=0) parallel failed");

    // Case 2: Control is 1
    let mut initial_vec_cry_c1 = vec![Complex::new(0.0, 0.0); 1 << num_test_qubits_cry];
    let idx_cry_c1_t0 = 1 << control_q_cry[0];
    initial_vec_cry_c1[idx_cry_c1_t0] = Complex::new(1.0, 0.0);
    let initial_state_cry_c1 = State { state_vector: initial_vec_cry_c1, num_qubits: num_test_qubits_cry };
    let new_state_cry_c1 = initial_state_cry_c1.cry_multi(target_q_cry, control_q_cry, angle_cry).unwrap();
    
    let ry_on_zero_for_cry = State::new_zero(1).unwrap().ry(0, angle_cry).unwrap();
    let mut expected_vec_cry_c1 = vec![Complex::new(0.0, 0.0); 1 << num_test_qubits_cry];
    if ry_on_zero_for_cry.state_vector[0].norm_sqr() > 1e-9 {
        expected_vec_cry_c1[idx_cry_c1_t0] = ry_on_zero_for_cry.state_vector[0];
    }
    if ry_on_zero_for_cry.state_vector[1].norm_sqr() > 1e-9 {
        expected_vec_cry_c1[idx_cry_c1_t0 | (1 << target_q_cry[0])] = ry_on_zero_for_cry.state_vector[1];
    }
    let expected_state_cry_c1 = State { state_vector: expected_vec_cry_c1, num_qubits: num_test_qubits_cry };
    assert_eq!(new_state_cry_c1, expected_state_cry_c1, "Controlled Ry (c=1) parallel failed");
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

    // crz(control = 0, target = 1, theta = pi)|11> = i|1> * |1> = i|11>
    let state: State = State::new_basis_n(2, 3).unwrap(); // |11>
    let new_state: State = state.crz_multi(&[1], &[0], PI).unwrap(); // RZ(|1>) * |1>
    let i: Complex<f64> = Complex::new(0.0, 1.0);
    let expected_state: State = (State::new_basis_n(1, 1).unwrap() * i).tensor_product(&State::new_basis_n(1, 1).unwrap()).unwrap(); // i|1> * |1> = i|11>
    assert_eq!(new_state, expected_state);

    // crz(control = 0, target = 1, theta = -pi)|+0> = |+0>
    let state: State = State::new_basis_n(2, 0).unwrap().tensor_product(&State::new_plus(1).unwrap()).unwrap();
    let new_state: State = state.crz_multi(&[0], &[1], -PI).unwrap(); // RZ(|0>) * |+0>
    let expected_state: State = state.clone(); // |+0>
    assert_eq!(new_state, expected_state);

    // Base qubits = 1
    assert_eq!(RotateZ::new(theta).base_qubits(), 1);

    // rz_multi on 11 qubits |0...0> [check if parallel method works (uncontrolled, multi-target)]
    let num_test_qubits_rz_unc = 11;
    let angle_rz_unc = PI / 1.2;
    let initial_state_rz_unc = State::new_zero(num_test_qubits_rz_unc).unwrap();
    let targets_rz_unc: Vec<usize> = (0..num_test_qubits_rz_unc).collect();
    let new_state_rz_unc = initial_state_rz_unc.rz_multi(&targets_rz_unc, angle_rz_unc).unwrap();
    let rz_on_zero_one_q = State::new_zero(1).unwrap().rz(0, angle_rz_unc).unwrap();
    let mut expected_state_rz_unc = rz_on_zero_one_q.clone();
    if num_test_qubits_rz_unc > 0 { // Should always be true here (11)
        for _ in 1..num_test_qubits_rz_unc {
            expected_state_rz_unc = expected_state_rz_unc.tensor_product(&rz_on_zero_one_q).unwrap();
        }
    } else {
        panic!("num_test_qubits_rz_unc is 0, which is not expected for this test");
    }
    assert_eq!(new_state_rz_unc, expected_state_rz_unc, "Uncontrolled Rz parallel (multi-target) failed");

    // crz(control=10, target=0, angle=theta) on 11 qubits. [check if controlled parallel method works]
    let num_test_qubits_crz = 11;
    let angle_crz = PI / 3.3;
    let control_q_crz = &[num_test_qubits_crz - 1];
    let target_q_crz = &[0];

    // Case 1: Control is 0
    let initial_state_crz_c0 = State::new_zero(num_test_qubits_crz).unwrap();
    let new_state_crz_c0 = initial_state_crz_c0.crz_multi(target_q_crz, control_q_crz, angle_crz).unwrap();
    assert_eq!(new_state_crz_c0, initial_state_crz_c0, "Controlled Rz (c=0) parallel failed");

    // Case 2: Control is 1
    // Target |0>: CRz|0> = e^(-i*theta/2)|0>
    let mut initial_vec_crz_c1_t0 = vec![Complex::new(0.0, 0.0); 1 << num_test_qubits_crz];
    let idx_crz_c1_t0 = 1 << control_q_crz[0];
    initial_vec_crz_c1_t0[idx_crz_c1_t0] = Complex::new(1.0, 0.0);
    let initial_state_crz_c1_t0 = State { state_vector: initial_vec_crz_c1_t0, num_qubits: num_test_qubits_crz };
    let new_state_crz_c1_t0 = initial_state_crz_c1_t0.crz_multi(target_q_crz, control_q_crz, angle_crz).unwrap();
    
    let rz_on_zero_for_crz = State::new_zero(1).unwrap().rz(0, angle_crz).unwrap(); // Rz|0>
    let mut expected_vec_crz_c1_t0_calc = vec![Complex::new(0.0, 0.0); 1 << num_test_qubits_crz];
    expected_vec_crz_c1_t0_calc[idx_crz_c1_t0] = rz_on_zero_for_crz.state_vector[0];
    let expected_state_crz_c1_t0 = State { state_vector: expected_vec_crz_c1_t0_calc, num_qubits: num_test_qubits_crz };
    assert_eq!(new_state_crz_c1_t0, expected_state_crz_c1_t0, "Controlled Rz (c=1, t=0) parallel failed");
}

#[test]
fn test_operator_unitary2_success() {
    // U = X = [[0, 1], [1, 0]]
    // U(|0>) = |1>
    // U(|1>) = |0>
    // U(|+>) = |+>
    // U(|->) = -|->

    let zero_state: State = State::new_zero(1).unwrap();
    let one_state: State = State::new_basis_n(1, 1).unwrap();
    let plus_state: State = State::new_plus(1).unwrap();
    let minus_state: State = State::new_minus(1).unwrap();

    let u: [[Complex<f64>; 2]; 2] = [
        [Complex::new(0.0, 0.0), Complex::new(1.0, 0.0)],
        [Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)],
    ];
    let expected_zero: State = one_state.clone();
    let expected_one: State = zero_state.clone();
    let expected_plus: State = plus_state.clone();
    let expected_minus: State = minus_state.clone() * -1.0;

    assert_eq!(zero_state.unitary(0, u).unwrap(), expected_zero.clone());
    assert_eq!(one_state.unitary(0, u).unwrap(), expected_one.clone());
    assert_eq!(plus_state.unitary(0, u).unwrap(), expected_plus.clone());
    assert_eq!(minus_state.unitary(0, u).unwrap(), expected_minus.clone());

    // u(|00>) = |11>
    let two_qubit_state: State = State::new_zero(2).unwrap();
    let new_state: State = two_qubit_state.unitary_multi(&[0, 1], u).unwrap();
    let expected_state: State = State::new_basis_n(2, 3).unwrap(); // |11>
    assert_eq!(new_state, expected_state);

    // cu(control = 0, target = 1, U = X)|11> = |01>
    let state: State = State::new_basis_n(2, 3).unwrap(); // |11>
    let new_state: State = state.cunitary_multi(&[1], &[0], u).unwrap(); // X(|1>) * |1>
    let expected_state: State = State::new_basis_n(2, 1).unwrap(); // |01>
    assert_eq!(new_state, expected_state);

    // cu(control = 0, target = 1, U = X)|+0> = |+0>
    let state: State = State::new_basis_n(2, 0).unwrap().tensor_product(&State::new_plus(1).unwrap()).unwrap();
    let new_state: State = state.cunitary_multi(&[0], &[1], u).unwrap(); // X(|0>) * |+0>
    let expected_state: State = state.clone(); // |+0>
    assert_eq!(new_state, expected_state);

    // Base qubits = 1
    assert_eq!(Unitary2::new(u).unwrap().base_qubits(), 1);

    // Test errors
    let invalid_u: [[Complex<f64>; 2]; 2] = [
        [Complex::new(0.0, 0.0), Complex::new(1.0, 0.0)],
        [Complex::new(1.0, 0.0), Complex::new(1.0, 0.0)],
    ];
    let invalid_unitary = Unitary2::new(invalid_u);
    assert!(invalid_unitary.is_err(), "Invalid unitary matrix should return an error.");

    // unitary_multi (with X matrix) on 11 qubits |0...0> [check if parallel method works (uncontrolled, multi-target)]
    let num_test_qubits_u_unc = 11;
    let u_matrix_unc = u; // X matrix from above
    let initial_state_u_unc = State::new_zero(num_test_qubits_u_unc).unwrap();
    let targets_u_unc: Vec<usize> = (0..num_test_qubits_u_unc).collect();
    let new_state_u_unc = initial_state_u_unc.unitary_multi(&targets_u_unc, u_matrix_unc).unwrap();
    let expected_state_u_unc = State::new_basis_n(num_test_qubits_u_unc, (1 << num_test_qubits_u_unc) - 1).unwrap(); // |1...1>
    assert_eq!(new_state_u_unc, expected_state_u_unc, "Uncontrolled Unitary2 (X) parallel (multi-target) failed");

    // cunitary (with X matrix, so CX) (control=10, target=0) on 11 qubits. [check if controlled parallel method works]
    let num_test_qubits_cu = 11;
    let u_matrix_cu = u; // X matrix
    let control_q_cu = &[num_test_qubits_cu - 1];
    let target_q_cu = &[0];

    // Case 1: Control is 0
    let initial_state_cu_c0 = State::new_zero(num_test_qubits_cu).unwrap();
    let new_state_cu_c0 = initial_state_cu_c0.cunitary_multi(target_q_cu, control_q_cu, u_matrix_cu).unwrap();
    assert_eq!(new_state_cu_c0, initial_state_cu_c0, "Controlled Unitary2 (X) (c=0) parallel failed");

    // Case 2: Control is 1
    let mut initial_vec_cu_c1 = vec![Complex::new(0.0, 0.0); 1 << num_test_qubits_cu];
    let idx_cu_c1_t0 = 1 << control_q_cu[0];
    initial_vec_cu_c1[idx_cu_c1_t0] = Complex::new(1.0, 0.0);
    let initial_state_cu_c1 = State { state_vector: initial_vec_cu_c1, num_qubits: num_test_qubits_cu };
    let new_state_cu_c1 = initial_state_cu_c1.cunitary_multi(target_q_cu, control_q_cu, u_matrix_cu).unwrap();
    
    let u_on_zero_for_cu = State::new_zero(1).unwrap().unitary(0, u_matrix_cu).unwrap(); // X|0> = |1>
    let mut expected_vec_cu_c1 = vec![Complex::new(0.0, 0.0); 1 << num_test_qubits_cu];
    if u_on_zero_for_cu.state_vector[0].norm_sqr() > 1e-9 {
        expected_vec_cu_c1[idx_cu_c1_t0] = u_on_zero_for_cu.state_vector[0];
    }
    if u_on_zero_for_cu.state_vector[1].norm_sqr() > 1e-9 {
        expected_vec_cu_c1[idx_cu_c1_t0 | (1 << target_q_cu[0])] = u_on_zero_for_cu.state_vector[1];
    }
    let expected_state_cu_c1 = State { state_vector: expected_vec_cu_c1, num_qubits: num_test_qubits_cu };
    assert_eq!(new_state_cu_c1, expected_state_cu_c1, "Controlled Unitary2 (X) (c=1) parallel failed");
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

    // CNOT(control=10, target=0) on 11 qubits [check if parallel method works for CNOT's own logic]
    let num_test_qubits_cnot_par = 11;
    let control_cnot_par = num_test_qubits_cnot_par - 1; // q10
    let target_cnot_par = 0; // q0

    // Case 1: Control is 0
    let initial_state_cnot_par_c0 = State::new_zero(num_test_qubits_cnot_par).unwrap();
    let new_state_cnot_par_c0 = initial_state_cnot_par_c0.cnot(control_cnot_par, target_cnot_par).unwrap();
    assert_eq!(new_state_cnot_par_c0, initial_state_cnot_par_c0, "CNOT parallel (control=0) failed");

    // Case 2: Control is 1
    let mut initial_vec_cnot_par_c1 = vec![Complex::new(0.0, 0.0); 1 << num_test_qubits_cnot_par];
    let idx_cnot_par_c1_t0 = 1 << control_cnot_par; // Control=1, Target=0
    initial_vec_cnot_par_c1[idx_cnot_par_c1_t0] = Complex::new(1.0, 0.0);
    let initial_state_cnot_par_c1 = State { state_vector: initial_vec_cnot_par_c1, num_qubits: num_test_qubits_cnot_par };
    let new_state_cnot_par_c1 = initial_state_cnot_par_c1.cnot(control_cnot_par, target_cnot_par).unwrap();
    let mut expected_vec_cnot_par_c1 = vec![Complex::new(0.0, 0.0); 1 << num_test_qubits_cnot_par];
    expected_vec_cnot_par_c1[idx_cnot_par_c1_t0 | (1 << target_cnot_par)] = Complex::new(1.0, 0.0); // Target flipped
    let expected_state_cnot_par_c1 = State { state_vector: expected_vec_cnot_par_c1, num_qubits: num_test_qubits_cnot_par };
    assert_eq!(new_state_cnot_par_c1, expected_state_cnot_par_c1, "CNOT parallel (control=1) failed");

    // Controlled CNOT (Toffoli) on 11 qubits [check if controlled parallel method works for CNOT, i.e. Toffoli]
    // Toffoli(c1=10, c2=9, t=0)
    let num_test_qubits_ccnot_par = 11;
    let c1_ccnot_par = num_test_qubits_ccnot_par - 1; // q10
    let c2_ccnot_par = num_test_qubits_ccnot_par - 2; // q9
    let t_ccnot_par = 0; // q0

    // Case 1: Controls 00 (one control for Toffoli is 0)
    let initial_state_ccnot_par_c00 = State::new_zero(num_test_qubits_ccnot_par).unwrap();
    let new_state_ccnot_par_c00 = initial_state_ccnot_par_c00.toffoli(c1_ccnot_par, c2_ccnot_par, t_ccnot_par).unwrap();
    assert_eq!(new_state_ccnot_par_c00, initial_state_ccnot_par_c00, "Toffoli (as controlled CNOT) parallel (controls 00) failed");
    
    // Case 2: Controls 11
    let mut initial_vec_ccnot_par_c11 = vec![Complex::new(0.0, 0.0); 1 << num_test_qubits_ccnot_par];
    let idx_ccnot_par_c11_t0 = (1 << c1_ccnot_par) | (1 << c2_ccnot_par); // Controls 11, Target 0
    initial_vec_ccnot_par_c11[idx_ccnot_par_c11_t0] = Complex::new(1.0, 0.0);
    let initial_state_ccnot_par_c11 = State { state_vector: initial_vec_ccnot_par_c11, num_qubits: num_test_qubits_ccnot_par };
    let new_state_ccnot_par_c11 = initial_state_ccnot_par_c11.toffoli(c1_ccnot_par, c2_ccnot_par, t_ccnot_par).unwrap();
    let mut expected_vec_ccnot_par_c11 = vec![Complex::new(0.0, 0.0); 1 << num_test_qubits_ccnot_par];
    expected_vec_ccnot_par_c11[idx_ccnot_par_c11_t0 | (1 << t_ccnot_par)] = Complex::new(1.0, 0.0); // Target flipped
    let expected_state_ccnot_par_c11 = State { state_vector: expected_vec_ccnot_par_c11, num_qubits: num_test_qubits_ccnot_par };
    assert_eq!(new_state_ccnot_par_c11, expected_state_ccnot_par_c11, "Toffoli (as controlled CNOT) parallel (controls 11) failed");
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

    // cswap(control = 0, target1 = 1, target2 = 2)
    // |011> -> |101>
    let state: State = State::new_basis_n(3, 3).unwrap(); // |011>
    let new_state: State = state.cswap(1, 2, &[0]).unwrap(); // SWAP(|1>, |0>)
    let expected_state: State = State::new_basis_n(3, 5).unwrap(); // |101>
    assert_eq!(new_state, expected_state);

    // cswap(control = 0, target1 = 1, target2 = 2)
    // |100> -> |100>
    let state: State = State::new_basis_n(3, 4).unwrap(); // |100>
    let new_state: State = state.cswap(1, 2, &[0]).unwrap(); // SWAP(|0>, |1>)
    let expected_state: State = State::new_basis_n(3, 4).unwrap(); // |100>
    assert_eq!(new_state, expected_state);

    // Base qubits = 2
    assert_eq!(SWAP.base_qubits(), 2);

    // SWAP(target1=0, target2=1) on 11 qubits [check if parallel method works for SWAP's own logic]
    let num_test_qubits_swap_par = 11;
    let t1_swap_par = 0; let t2_swap_par = 1;
    let mut initial_vec_swap_par = vec![Complex::new(0.0, 0.0); 1 << num_test_qubits_swap_par];
    initial_vec_swap_par[1 << t1_swap_par] = Complex::new(1.0, 0.0); // |...0010> (1 at t1_swap_par)
    let initial_state_swap_par = State { state_vector: initial_vec_swap_par, num_qubits: num_test_qubits_swap_par };
    let new_state_swap_par = initial_state_swap_par.swap(t1_swap_par, t2_swap_par).unwrap();
    let mut expected_vec_swap_par = vec![Complex::new(0.0, 0.0); 1 << num_test_qubits_swap_par];
    expected_vec_swap_par[1 << t2_swap_par] = Complex::new(1.0, 0.0); // |...0100> (1 at t2_swap_par)
    let expected_state_swap_par = State { state_vector: expected_vec_swap_par, num_qubits: num_test_qubits_swap_par };
    assert_eq!(new_state_swap_par, expected_state_swap_par, "SWAP parallel failed");

    // CSWAP(control=10, target1=0, target2=1) on 11 qubits [check if controlled parallel method works for SWAP, i.e. CSWAP]
    let num_test_qubits_cswap_par = 11;
    let control_cswap_par = &[num_test_qubits_cswap_par - 1]; // q10
    let t1_cswap_par = 0; let t2_cswap_par = 1;

    // Case 1: Control is 0
    let mut initial_vec_cswap_par_c0 = vec![Complex::new(0.0, 0.0); 1 << num_test_qubits_cswap_par];
    initial_vec_cswap_par_c0[1 << t1_cswap_par] = Complex::new(1.0, 0.0); // Control is 0, t1 is 1, t2 is 0
    let initial_state_cswap_par_c0 = State { state_vector: initial_vec_cswap_par_c0, num_qubits: num_test_qubits_cswap_par };
    let new_state_cswap_par_c0 = initial_state_cswap_par_c0.cswap(t1_cswap_par, t2_cswap_par, control_cswap_par).unwrap();
    assert_eq!(new_state_cswap_par_c0, initial_state_cswap_par_c0, "CSWAP parallel (control=0) failed");

    // Case 2: Control is 1
    let mut initial_vec_cswap_par_c1 = vec![Complex::new(0.0, 0.0); 1 << num_test_qubits_cswap_par];
    let idx_cswap_par_c1_t1_is_1 = (1 << control_cswap_par[0]) | (1 << t1_cswap_par); // Control 1, t1 is 1, t2 is 0
    initial_vec_cswap_par_c1[idx_cswap_par_c1_t1_is_1] = Complex::new(1.0, 0.0);
    let initial_state_cswap_par_c1 = State { state_vector: initial_vec_cswap_par_c1, num_qubits: num_test_qubits_cswap_par };
    let new_state_cswap_par_c1 = initial_state_cswap_par_c1.cswap(t1_cswap_par, t2_cswap_par, control_cswap_par).unwrap();
    let mut expected_vec_cswap_par_c1 = vec![Complex::new(0.0, 0.0); 1 << num_test_qubits_cswap_par];
    expected_vec_cswap_par_c1[(1 << control_cswap_par[0]) | (1 << t2_cswap_par)] = Complex::new(1.0, 0.0); // Control 1, t1 is 0, t2 is 1
    let expected_state_cswap_par_c1 = State { state_vector: expected_vec_cswap_par_c1, num_qubits: num_test_qubits_cswap_par };
    assert_eq!(new_state_cswap_par_c1, expected_state_cswap_par_c1, "CSWAP parallel (control=1) failed");
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

    // Toffoli(c1=10, c2=9, t=0) on 11 qubits [check if parallel method works for Toffoli's own logic]
    let num_test_qubits_toffoli_par = 11;
    let c1_toffoli_par = num_test_qubits_toffoli_par - 1; // q10
    let c2_toffoli_par = num_test_qubits_toffoli_par - 2; // q9
    let t_toffoli_par = 0; // q0

    // Case 1: Controls 00 (c1=0, c2=0)
    let initial_state_toffoli_par_c00 = State::new_zero(num_test_qubits_toffoli_par).unwrap();
    let new_state_toffoli_par_c00 = initial_state_toffoli_par_c00.toffoli(c1_toffoli_par, c2_toffoli_par, t_toffoli_par).unwrap();
    assert_eq!(new_state_toffoli_par_c00, initial_state_toffoli_par_c00, "Toffoli parallel (controls 00) failed");
    
    // Case 2: Controls 11 (c1=1, c2=1), target 0
    let mut initial_vec_toffoli_par_c11 = vec![Complex::new(0.0, 0.0); 1 << num_test_qubits_toffoli_par];
    let idx_toffoli_par_c11_t0 = (1 << c1_toffoli_par) | (1 << c2_toffoli_par); // Controls 11, Target 0
    initial_vec_toffoli_par_c11[idx_toffoli_par_c11_t0] = Complex::new(1.0, 0.0);
    let initial_state_toffoli_par_c11 = State { state_vector: initial_vec_toffoli_par_c11, num_qubits: num_test_qubits_toffoli_par };
    let new_state_toffoli_par_c11 = initial_state_toffoli_par_c11.toffoli(c1_toffoli_par, c2_toffoli_par, t_toffoli_par).unwrap();
    let mut expected_vec_toffoli_par_c11 = vec![Complex::new(0.0, 0.0); 1 << num_test_qubits_toffoli_par];
    expected_vec_toffoli_par_c11[idx_toffoli_par_c11_t0 | (1 << t_toffoli_par)] = Complex::new(1.0, 0.0); // Target flipped
    let expected_state_toffoli_par_c11 = State { state_vector: expected_vec_toffoli_par_c11, num_qubits: num_test_qubits_toffoli_par };
    assert_eq!(new_state_toffoli_par_c11, expected_state_toffoli_par_c11, "Toffoli parallel (controls 11) failed");

    // C-Toffoli (CCCX) (controls q10,q9,q8, target q0) on 11 qubits [check if controlled parallel method works for Toffoli, i.e. CCCX]
    // This uses Pauli::X.apply with 3 control qubits.
    let num_test_qubits_cccx_par = 11;
    let cccx_controls_par = &[num_test_qubits_cccx_par - 1, num_test_qubits_cccx_par - 2, num_test_qubits_cccx_par - 3]; // q10, q9, q8
    let cccx_target_par = &[0]; // q0

    // Case 1: One control is 0 (e.g., q8 is 0)
    let mut initial_vec_cccx_par_one_c0 = vec![Complex::new(0.0, 0.0); 1 << num_test_qubits_cccx_par];
    let idx_cccx_par_one_c0 = (1 << cccx_controls_par[0]) | (1 << cccx_controls_par[1]); // q10=1, q9=1, q8=0, target=0
    initial_vec_cccx_par_one_c0[idx_cccx_par_one_c0] = Complex::new(1.0, 0.0);
    let initial_state_cccx_par_one_c0 = State { state_vector: initial_vec_cccx_par_one_c0, num_qubits: num_test_qubits_cccx_par };
    let new_state_cccx_par_one_c0 = Pauli::X.apply(&initial_state_cccx_par_one_c0, cccx_target_par, cccx_controls_par).unwrap();
    assert_eq!(new_state_cccx_par_one_c0, initial_state_cccx_par_one_c0, "CCCX parallel (one control=0) failed");

    // Case 2: All controls are 1
    let mut initial_vec_cccx_par_all_c1 = vec![Complex::new(0.0, 0.0); 1 << num_test_qubits_cccx_par];
    let idx_cccx_par_all_c1_t0 = (1 << cccx_controls_par[0]) | (1 << cccx_controls_par[1]) | (1 << cccx_controls_par[2]); // q10,q9,q8 = 1, target=0
    initial_vec_cccx_par_all_c1[idx_cccx_par_all_c1_t0] = Complex::new(1.0, 0.0);
    let initial_state_cccx_par_all_c1 = State { state_vector: initial_vec_cccx_par_all_c1, num_qubits: num_test_qubits_cccx_par };
    let new_state_cccx_par_all_c1 = Pauli::X.apply(&initial_state_cccx_par_all_c1, cccx_target_par, cccx_controls_par).unwrap();
    let mut expected_vec_cccx_par_all_c1 = vec![Complex::new(0.0, 0.0); 1 << num_test_qubits_cccx_par];
    expected_vec_cccx_par_all_c1[idx_cccx_par_all_c1_t0 | (1 << cccx_target_par[0])] = Complex::new(1.0, 0.0); // Target flipped
    let expected_state_cccx_par_all_c1 = State { state_vector: expected_vec_cccx_par_all_c1, num_qubits: num_test_qubits_cccx_par };
    assert_eq!(new_state_cccx_par_all_c1, expected_state_cccx_par_all_c1, "CCCX parallel (all controls=1) failed");
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
