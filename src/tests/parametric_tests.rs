use crate::{components::parametric::{
    parameter::Parameter,
}, CircuitBuilder};

#[test]
fn test_parametric_parameter_new() {
    // This should compile and run
    let _parameter = Parameter::new([0.5, 1.0]);
}

#[test]
fn test_parametric_parameter_get() {
    let parameter = Parameter::new([0.5, 1.0]);
    let values = parameter.get();
    assert_eq!(values, [0.5, 1.0]);
}

#[test]
fn test_parametric_parameter_set() {
    let parameter = Parameter::new([0.5, 1.0]);
    parameter.set([0.6, 1.1]);
    let values = parameter.get();
    assert_eq!(values, [0.6, 1.1]);
}

#[test]
fn test_parametric_parametric_ryphase() {
    let parameter = Parameter::new([0.5, 1.0]);

    let circuit_ry_phase = CircuitBuilder::new(1)
        .parametric_ry_phase_gate(0, parameter.clone())
        .build_final().expect("Failed to build circuit").to_concrete_circuit();

    let concrete_circuit_ry_phase = circuit! {
        qubits: 1,
        ry_phase(0, parameter.get()[0], parameter.get()[1])
    }.expect("Failed to create concrete circuit");

    assert_eq!(format!("{:?}", circuit_ry_phase), format!("{:?}", concrete_circuit_ry_phase));

    let circuit_cry_phase = CircuitBuilder::new(3)
        .parametric_cry_phase_gates(vec![0, 1], vec![2], vec![parameter.clone(), parameter.clone()]).unwrap()
        .build_final().expect("Failed to build circuit").to_concrete_circuit();

    let concrete_circuit_cry_phase = circuit! {
        qubits: 3,
        cry_phase([0, 1], 2, parameter.get()[0], parameter.get()[1]),
    }.expect("Failed to create concrete circuit");

    assert_eq!(format!("{:?}", circuit_cry_phase), format!("{:?}", concrete_circuit_cry_phase));

    let circuit = CircuitBuilder::new(2)
        .parametric_ry_phase_gates(vec![0, 1], vec![parameter.clone(), parameter.clone()]).unwrap()
        .build_final().expect("Failed to build circuit").to_concrete_circuit();

    let concrete_circuit = circuit! {
        qubits: 2,
        ry_phase([0, 1], parameter.get()[0], parameter.get()[1])
    }.expect("Failed to create concrete circuit");

    assert_eq!(format!("{:?}", circuit), format!("{:?}", concrete_circuit));
}

#[test]
fn test_parametric_parametric_matchgate() {
    let parameter = Parameter::new([0.5, 1.0, 1.5]);

    let circuit = CircuitBuilder::new(2)
        .parametric_matchgate(0, parameter.clone())
        .build_final().expect("Failed to build circuit").to_concrete_circuit();

    let concrete_circuit = circuit! {
        qubits: 2,
        matchgate(0, parameter.get()[0], parameter.get()[1], parameter.get()[2])
    }.expect("Failed to create concrete circuit");

    assert_eq!(format!("{:?}", circuit), format!("{:?}", concrete_circuit));

    let circuit_controlled = CircuitBuilder::new(3)
        .parametric_cmatchgate(0, vec![2], parameter.clone())
        .build_final().expect("Failed to build circuit").to_concrete_circuit();

    let concrete_circuit_controlled = circuit! {
        qubits: 3,
        cmatchgate(0, [2], parameter.get()[0], parameter.get()[1], parameter.get()[2])
    }.expect("Failed to create concrete circuit");

    assert_eq!(format!("{:?}", circuit_controlled), format!("{:?}", concrete_circuit_controlled));
}

#[test]
fn test_parametric_change_parameter_val() {
    let parameter_ryp = Parameter::new([0.5, 1.0]);
    let parameter_mc = Parameter::new([0.5, 1.0, 1.5]);
    let circuit = CircuitBuilder::new(3)
        .parametric_cry_phase_gates(vec![0, 1], vec![2], vec![parameter_ryp.clone(), parameter_ryp.clone()])
        .expect("Failed to create parametric cry phase gates")
        .parametric_matchgate(0, parameter_mc.clone())
        .build_final().expect("Failed to build circuit");

    let circuit_1 = circuit! {
        qubits: 3,
        cry_phase([0, 1], 2, parameter_ryp.get()[0], parameter_ryp.get()[1]),
        matchgate(0, parameter_mc.get()[0], parameter_mc.get()[1], parameter_mc.get()[2])
    }.expect("Failed to create concrete circuit");

    assert_eq!(format!("{:?}", circuit.to_concrete_circuit()), format!("{:?}", circuit_1));

    // Change param values
    parameter_ryp.set([0.6, 1.1]);
    parameter_mc.set([0.6, 1.1, 1.6]);

    let circuit_2 = circuit! {
        qubits: 3,
        cry_phase([0, 1], 2, parameter_ryp.get()[0], parameter_ryp.get()[1]),
        matchgate(0, parameter_mc.get()[0], parameter_mc.get()[1], parameter_mc.get()[2])
    }.expect("Failed to create concrete circuit");

    assert_eq!(format!("{:?}", circuit.to_concrete_circuit()), format!("{:?}", circuit_2));
}

#[test]
fn test_parametric_mismatched_parameters_error() {
    let parameter = Parameter::new([0.5, 1.0]);
    let mut builder = CircuitBuilder::new(3);
    let err_cry = builder.parametric_cry_phase_gates(vec![0, 1], vec![2], vec![parameter.clone()]);

    if let Err(err) = err_cry {
        assert_eq!(
            err.to_string(),
            "Mismatched number of parameters: expected 2, got 1"
        );
    } else {
        panic!("Expected an error due to mismatched parameters");
    }

    let mut builder = CircuitBuilder::new(3);
    let err_ry = builder.parametric_ry_phase_gates(vec![0, 1], vec![parameter.clone()]);
    if let Err(err) = err_ry {
        assert_eq!(
            err.to_string(),
            "Mismatched number of parameters: expected 2, got 1"
        );
    } else {
        panic!("Expected an error due to mismatched parameters");
    }

    let parameter_1d = Parameter::new([0.5]);

    // Test crx_gates
    let mut builder_crx = CircuitBuilder::new(3);
    let err_crx = builder_crx.parametric_crx_gates(vec![0, 1], vec![2], vec![parameter_1d.clone()]);
    if let Err(err) = err_crx {
        assert_eq!(
            err.to_string(),
            "Mismatched number of parameters: expected 2, got 1"
        );
    } else {
        panic!("Expected an error due to mismatched parameters for crx");
    }

    // Test cry_gates
    let mut builder_cry = CircuitBuilder::new(3);
    let err_cry = builder_cry.parametric_cry_gates(vec![0, 1], vec![2], vec![parameter_1d.clone()]);
    if let Err(err) = err_cry {
        assert_eq!(
            err.to_string(),
            "Mismatched number of parameters: expected 2, got 1"
        );
    } else {
        panic!("Expected an error due to mismatched parameters for cry");
    }

    // Test crz_gates
    let mut builder_crz = CircuitBuilder::new(3);
    let err_crz = builder_crz.parametric_crz_gates(vec![0, 1], vec![2], vec![parameter_1d.clone()]);
    if let Err(err) = err_crz {
        assert_eq!(
            err.to_string(),
            "Mismatched number of parameters: expected 2, got 1"
        );
    } else {
        panic!("Expected an error due to mismatched parameters for crz");
    }

    // Test cp_gates
    let mut builder_cp = CircuitBuilder::new(3);
    let err_cp = builder_cp.parametric_cp_gates(vec![0, 1], vec![2], vec![parameter_1d.clone()]);
    if let Err(err) = err_cp {
        assert_eq!(
            err.to_string(),
            "Mismatched number of parameters: expected 2, got 1"
        );
    } else {
        panic!("Expected an error due to mismatched parameters for cp");
    }
}

#[test]
fn test_parametric_rotations() {
    let parameter = Parameter::new([0.5]);

    // Rx
    let circuit_rx = CircuitBuilder::new(1)
        .parametric_rx_gate(0, parameter.clone())
        .build_final()
        .expect("Failed to build circuit")
        .to_concrete_circuit();
    let concrete_circuit_rx = circuit! {
        qubits: 1,
        rx(0, parameter.get()[0])
    }
    .expect("Failed to create concrete circuit");
    assert_eq!(
        format!("{:?}", circuit_rx),
        format!("{:?}", concrete_circuit_rx)
    );

    // cRx
    let circuit_crx = CircuitBuilder::new(2)
        .parametric_crx_gates(vec![0], vec![1], vec![parameter.clone()])
        .unwrap()
        .build_final()
        .expect("Failed to build circuit")
        .to_concrete_circuit();
    let concrete_circuit_crx = circuit! {
        qubits: 2,
        crx(0, 1, parameter.get()[0])
    }
    .expect("Failed to create concrete circuit");
    assert_eq!(
        format!("{:?}", circuit_crx),
        format!("{:?}", concrete_circuit_crx)
    );

    // Ry
    let circuit_ry = CircuitBuilder::new(1)
        .parametric_ry_gate(0, parameter.clone())
        .build_final()
        .expect("Failed to build circuit")
        .to_concrete_circuit();
    let concrete_circuit_ry = circuit! {
        qubits: 1,
        ry(0, parameter.get()[0])
    }
    .expect("Failed to create concrete circuit");
    assert_eq!(
        format!("{:?}", circuit_ry),
        format!("{:?}", concrete_circuit_ry)
    );

    // cRy
    let circuit_cry = CircuitBuilder::new(2)
        .parametric_cry_gates(vec![0], vec![1], vec![parameter.clone()])
        .unwrap()
        .build_final()
        .expect("Failed to build circuit")
        .to_concrete_circuit();
    let concrete_circuit_cry = circuit! {
        qubits: 2,
        cry(0, 1, parameter.get()[0])
    }
    .expect("Failed to create concrete circuit");
    assert_eq!(
        format!("{:?}", circuit_cry),
        format!("{:?}", concrete_circuit_cry)
    );

    // Rz
    let circuit_rz = CircuitBuilder::new(1)
        .parametric_rz_gate(0, parameter.clone())
        .build_final()
        .expect("Failed to build circuit")
        .to_concrete_circuit();
    let concrete_circuit_rz = circuit! {
        qubits: 1,
        rz(0, parameter.get()[0])
    }
    .expect("Failed to create concrete circuit");
    assert_eq!(
        format!("{:?}", circuit_rz),
        format!("{:?}", concrete_circuit_rz)
    );

    // cRz
    let circuit_crz = CircuitBuilder::new(2)
        .parametric_crz_gates(vec![0], vec![1], vec![parameter.clone()])
        .unwrap()
        .build_final()
        .expect("Failed to build circuit")
        .to_concrete_circuit();
    let concrete_circuit_crz = circuit! {
        qubits: 2,
        crz(0, 1, parameter.get()[0])
    }
    .expect("Failed to create concrete circuit");
    assert_eq!(
        format!("{:?}", circuit_crz),
        format!("{:?}", concrete_circuit_crz)
    );
}

#[test]
fn test_parametric_phase() {
    let parameter = Parameter::new([0.5]);

    // P
    let circuit_p = CircuitBuilder::new(1)
        .parametric_p_gate(0, parameter.clone())
        .build_final()
        .expect("Failed to build circuit")
        .to_concrete_circuit();
    let concrete_circuit_p = circuit! {
        qubits: 1,
        p(0, parameter.get()[0])
    }
    .expect("Failed to create concrete circuit");
    assert_eq!(
        format!("{:?}", circuit_p),
        format!("{:?}", concrete_circuit_p)
    );

    // cP
    let circuit_cp = CircuitBuilder::new(2)
        .parametric_cp_gates(vec![0], vec![1], vec![parameter.clone()])
        .unwrap()
        .build_final()
        .expect("Failed to build circuit")
        .to_concrete_circuit();
    let concrete_circuit_cp = circuit! {
        qubits: 2,
        cp(0, 1, parameter.get()[0])
    }
    .expect("Failed to create concrete circuit");
    assert_eq!(
        format!("{:?}", circuit_cp),
        format!("{:?}", concrete_circuit_cp)
    );
}