use crate::components::gate::Gate;
use crate::components::parametric::parameter::Parameter;
use std::fmt::Debug;

/// A trait for gates that can be parametrised.
///
/// The trait defines the `to_concrete_gate` method for creating a concrete gate from the parametrised gate.
/// This is required for applying the parametrised gate in a circuit or compiling it into OpenQASM.
pub trait ParametricGate: Debug + Send + Sync {
    /// Creates a concrete `Gate` using the current parameter values.
    ///
    /// # Arguments
    ///
    /// * `target_indices` - The indices of the target qubits.
    /// * `control_indices` - The indices of the control qubits.
    ///
    /// # Returns
    ///
    /// Concrete `Gate::Operator` variants of the parameterised gate instance.
    fn to_concrete_gates(&self, target_indices: &[usize], control_indices: &[usize]) -> Vec<Gate>;

    /// Clones the trait object.
    /// This is required by the `Gate` trait.
    fn box_clone(&self) -> Box<dyn ParametricGate>;
}

impl Clone for Box<dyn ParametricGate> {
    fn clone(&self) -> Self {
        self.box_clone()
    }
}

/// A parametrised RY phase gate
///
/// This operator can be decomposed into a rotation around the Y axis followed by a phase shift.
/// The enclosed unitary matrix is guaranteed to be unitary.
///
/// # Fields
/// - `parameter`: A `Parameter<2>` instance that holds the rotation angle (theta) and phase shift (phi).
#[derive(Debug, Clone)]
pub struct ParametricRyPhase {
    pub parameter: Parameter<2>, // theta, phi
}

impl ParametricGate for ParametricRyPhase {
    fn to_concrete_gates(&self, target_indices: &[usize], control_indices: &[usize]) -> Vec<Gate> {
        let params = self.parameter.get();
        Gate::ry_phase_controlled_gates(
            target_indices.to_vec(),
            control_indices.to_vec(),
            params[0],
            params[1],
        )
    }

    fn box_clone(&self) -> Box<dyn ParametricGate> {
        Box::new(self.clone())
    }
}

/// A parametrised matchgate
///
/// A two-qubit operator that applies a matchgate transformation to the adjacent target qubits.
/// This gate can be decomposed into a two-qubit rotation and phase shifts.
///
/// # Fields
/// - `parameter`: A `Parameter<3>` instance that holds the rotation angle (theta) and phase shifts (phi1, phi2).
#[derive(Debug, Clone)]
pub struct ParametricMatchgate {
    pub parameter: Parameter<3>, // theta, phi1, phi2
}

impl ParametricGate for ParametricMatchgate {
    fn to_concrete_gates(&self, target_indices: &[usize], control_indices: &[usize]) -> Vec<Gate> {
        let params = self.parameter.get();
        vec![Gate::controlled_matchgate(
            target_indices[0],
            control_indices.to_vec(),
            params[0],
            params[1],
            params[2],
        )]
    }

    fn box_clone(&self) -> Box<dyn ParametricGate> {
        Box::new(self.clone())
    }
}
