use quant_iron::heisenberg::heisenberg_1d;
use quant_iron::State;

/// Create a Hamiltonian for the 1D Heisenberg model and execute it on a state
fn main() {
    // Define a Hamiltonian for the 1D Heisenberg model
    let number_of_spins = 3;
    let coupling_constant_x = 1.0;
    let coupling_constant_y = 2.0;
    let coupling_constant_z = 3.0;
    let field_strength = 0.5;
    let magnetic_field = 0.1;

    let hamiltonian = heisenberg_1d(number_of_spins, coupling_constant_x,
                                    coupling_constant_y, coupling_constant_z, field_strength, magnetic_field)
        .expect("Could not get heisenberg constant value");

    // Initialise a |+++> state
    let state = State::new_plus(3).expect("Could not initialise state");
    // Apply the Hamiltonian to the state
    let modified_state = hamiltonian.apply(&state).expect("Could not apply hamiltonian");
    // Print the modified state
    println!("Modified state:\n{:?}", modified_state);
}
