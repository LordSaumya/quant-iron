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
}