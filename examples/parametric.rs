use quant_iron::{parameter::Parameter, CircuitBuilder};
use std::f64::consts::PI;
fn main() {
    // Create new parameters
    let angles_1 = Parameter::new([PI / 4.0, PI / 2.0]); // Theta & Phi

    let angles_2 = Parameter::new([PI / 3.0, PI / 6.0, PI / 12.0]); // Theta, Phi1 & Phi2

    let _circuit = CircuitBuilder::new(3)
        .parametric_ry_phase_gate(0, angles_1.clone()) // Add a parametrised ry_phase gate
        .parametric_ry_phase_gate(2, angles_1.clone()) // Add a parametrised ry_phase gate with shared parameters
        .parametric_matchgate(1, angles_2.clone()) // Add a parametrised matchgate
        .build_final()
        .expect("Failed to build circuit");

    angles_1.set([PI / 2.0, PI / 3.0]); // Update parameters
    
    println!("{:?}", angles_2.get()); // Get parameters
}
