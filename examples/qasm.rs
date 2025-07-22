use std::f64::consts::PI;
use quant_iron::{CircuitBuilder, MeasurementBasis};

fn main() {
    // Create a circuit
    let circuit = CircuitBuilder::new(5)
    .h_gate(0)
    .cx_gates(vec![1, 2], vec![0])
    .p_gates(vec![0,1,2], PI)
    .measure_gate(MeasurementBasis::X, vec![0, 1])
    .swap_gate(3, 4)
    .ctdag_gates(vec![2, 3], vec![0, 3])
    .crz_gates(vec![4], vec![0, 1], PI)
    .measure_gate(MeasurementBasis::Custom([[0.0.into(), 1.0.into()], [1.0.into(), 0.0.into()]]), vec![2, 3, 4])
    .build_final();

    // Convert circuit to OpenQASM 3.0
    println!("{}", circuit.expect("Could not build circuit").to_qasm(None::<&str>).expect("Could not convert circuit to QASM"));
}