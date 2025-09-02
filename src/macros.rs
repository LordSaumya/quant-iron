/// Creates a `Circuit` object using a domain-specific language.
///
/// The macro provides a concise, intuitive way to define a quantum circuit.
///
/// # Syntax
///
/// The basic structure is:
/// ```ignore
/// circuit! {
///     qubits: <number_of_qubits>,
///     <operation_1>,
///     <operation_2>,
///     ...
/// }
/// ```
///
/// Where `<operation>` can be one of the following:
///
/// ## Single-Qubit Gates
///
/// Apply a gate to a single qubit or multiple qubits individually.
/// - `h(qubit)` or `h([qubit1, qubit2, ...])`
/// - `x(qubit)` or `x([qubit1, qubit2, ...])`
/// - `y(qubit)` or `y([qubit1, qubit2, ...])`
/// - `z(qubit)` or `z([qubit1, qubit2, ...])`
/// - `s(qubit)` or `s([qubit1, qubit2, ...])`
/// - `t(qubit)` or `t([qubit1, qubit2, ...])`
/// - `sdag(qubit)` or `sdag([qubit1, qubit2, ...])`
/// - `tdag(qubit)` or `tdag([qubit1, qubit2, ...])`
/// - `id(qubit)` or `id([qubit1, qubit2, ...])`
///
/// ## Two-Qubit Gates
///
/// - `cnot(control, target)`
/// - `swap(qubit1, qubit2)`
///
/// ## Controlled Gates
///
/// Controlled gates can have single or multiple targets and controls.
/// - `ch(target, control)` or `ch([targets], [controls])`
/// - `cx(target, control)` or `cx([targets], [controls])`
/// - `cy(target, control)` or `cy([targets], [controls])`
/// - `cz(target, control)` or `cz([targets], [controls])`
/// - `cs(target, control)` or `cs([targets], [controls])`
/// - `csdag(target, control)` or `csdag([targets], [controls])`
/// - `ct(target, control)` or `ct([targets], [controls])`
/// - `ctdag(target, control)` or `ctdag([targets], [controls])`
///
/// ## Gates with Angles
///
/// Angles are `f64` values.
/// - `rx(qubit, angle)` or `rx([qubits], angle)`
/// - `ry(qubit, angle)` or `ry([qubits], angle)`
/// - `rz(qubit, angle)` or `rz([qubits], angle)`
/// - `p(qubit, angle)` or `p([qubits], angle)`
/// - `ry_phase(qubit, theta, phi)` or `ry_phase([qubits], theta, phi)`
/// - `ry_phase_dag(qubit, theta, phi)` or `ry_phase_dag([qubits], theta, phi)`
///
/// ## Controlled Gates with Angles
///
/// - `crx(target, control, angle)` or `crx([targets], [controls], angle)`
/// - `cry(target, control, angle)` or `cry([targets], [controls], angle)`
/// - `crz(target, control, angle)` or `crz([targets], [controls], angle)`
/// - `cp(target, control, angle)` or `cp([targets], [controls], angle)`
/// - `cry_phase(target, control, theta, phi)` or `cry_phase([targets], [controls], theta, phi)`
/// - `cry_phase_dag(target, control, theta, phi)` or `cry_phase_dag([targets], [controls], theta, phi)`
///
/// ## Special Gates
///
/// - `toffoli(target, control1, control2)`
/// - `cswap(target1, target2, control)` or `cswap(target1, target2, [controls])`
/// - `matchgate(target, theta, phi1, phi2)`
/// - `cmatchgate(target, control, theta, phi1, phi2)` or `cmatchgate(target, [controls], theta, phi1, phi2)`
/// - `pauli_string(pauli_string)`
///
/// ## Unitary Gates
///
/// Apply a custom unitary matrix.
/// - `unitary(qubit, matrix)`
/// - `unitary([qubits], matrix)`
/// - `cunitary(target, control, matrix)` or `cunitary([targets], [controls], matrix)`
/// 
/// ### Warning
/// 
/// * The `matrix` must be a valid unitary matrix. if the matrix is not unitary, the macro will panic.
///
/// ## Measurement Gates
///
/// - `measurex(qubit)` or `measurex([qubits])`
/// - `measurey(qubit)` or `measurey([qubits])`
/// - `measurez(qubit)` or `measurez([qubits])`
/// - `measure_custom(qubit, matrix)` or `measure_custom([qubits], matrix)`
/// 
/// # Returns
/// 
/// * `Result<Circuit, Error>` - A result containing the constructed `Circuit` or an error if the circuit cannot be built.
#[macro_export]
macro_rules! circuit {
    (qubits: $num_qubits:expr, $($rest:tt)*) => {
        {
            let mut builder = $crate::circuit::CircuitBuilder::new($num_qubits);
            $crate::circuit_internal!(builder, $($rest)*);
            builder.build_final()
        }
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! circuit_internal {
    // Base case: No more tokens to process.
    ($builder:ident,) => {};

    // --- Gate Overloading rules (with trailing comma) ---
    // For each gate, most specific form (arrays) -> least specific form (single qubits) for disambiguation.

    // Multi-qubit gates
    ($builder:ident, h([$($qubits:expr),*]), $($rest:tt)*) => { $builder.h_gates(vec![$($qubits),*]); $crate::circuit_internal!($builder, $($rest)*); };
    ($builder:ident, x([$($qubits:expr),*]), $($rest:tt)*) => { $builder.x_gates(vec![$($qubits),*]); $crate::circuit_internal!($builder, $($rest)*); };
    ($builder:ident, y([$($qubits:expr),*]), $($rest:tt)*) => { $builder.y_gates(vec![$($qubits),*]); $crate::circuit_internal!($builder, $($rest)*); };
    ($builder:ident, z([$($qubits:expr),*]), $($rest:tt)*) => { $builder.z_gates(vec![$($qubits),*]); $crate::circuit_internal!($builder, $($rest)*); };
    ($builder:ident, s([$($qubits:expr),*]), $($rest:tt)*) => { $builder.s_gates(vec![$($qubits),*]); $crate::circuit_internal!($builder, $($rest)*); };
    ($builder:ident, t([$($qubits:expr),*]), $($rest:tt)*) => { $builder.t_gates(vec![$($qubits),*]); $crate::circuit_internal!($builder, $($rest)*); };
    ($builder:ident, id([$($qubits:expr),*]), $($rest:tt)*) => { $builder.id_gates(vec![$($qubits),*]); $crate::circuit_internal!($builder, $($rest)*); };
    ($builder:ident, sdag([$($qubits:expr),*]), $($rest:tt)*) => { $builder.sdag_gates(vec![$($qubits),*]); $crate::circuit_internal!($builder, $($rest)*); };
    ($builder:ident, tdag([$($qubits:expr),*]), $($rest:tt)*) => { $builder.tdag_gates(vec![$($qubits),*]); $crate::circuit_internal!($builder, $($rest)*); };
    ($builder:ident, pauli_string($pauli_string:expr), $($rest:tt)*) => { $builder.pauli_string_gate($pauli_string); $crate::circuit_internal!($builder, $($rest)*); };

    // Single-qubit gates
    ($builder:ident, h($qubit:expr), $($rest:tt)*) => { $builder.h_gate($qubit); $crate::circuit_internal!($builder, $($rest)*); };
    ($builder:ident, x($qubit:expr), $($rest:tt)*) => { $builder.x_gate($qubit); $crate::circuit_internal!($builder, $($rest)*); };
    ($builder:ident, y($qubit:expr), $($rest:tt)*) => { $builder.y_gate($qubit); $crate::circuit_internal!($builder, $($rest)*); };
    ($builder:ident, z($qubit:expr), $($rest:tt)*) => { $builder.z_gate($qubit); $crate::circuit_internal!($builder, $($rest)*); };
    ($builder:ident, s($qubit:expr), $($rest:tt)*) => { $builder.s_gate($qubit); $crate::circuit_internal!($builder, $($rest)*); };
    ($builder:ident, t($qubit:expr), $($rest:tt)*) => { $builder.t_gate($qubit); $crate::circuit_internal!($builder, $($rest)*); };
    ($builder:ident, id($qubit:expr), $($rest:tt)*) => { $builder.id_gate($qubit); $crate::circuit_internal!($builder, $($rest)*); };
    ($builder:ident, sdag($qubit:expr), $($rest:tt)*) => { $builder.sdag_gate($qubit); $crate::circuit_internal!($builder, $($rest)*); };
    ($builder:ident, tdag($qubit:expr), $($rest:tt)*) => { $builder.tdag_gate($qubit); $crate::circuit_internal!($builder, $($rest)*); };

    // Two-argument gates
    ($builder:ident, cnot($arg1:expr, $arg2:expr), $($rest:tt)*) => { $builder.cnot_gate($arg1, $arg2); $crate::circuit_internal!($builder, $($rest)*); };
    ($builder:ident, swap($arg1:expr, $arg2:expr), $($rest:tt)*) => { $builder.swap_gate($arg1, $arg2); $crate::circuit_internal!($builder, $($rest)*); };

    // --- Controlled gate overloading ---
    ($builder:ident, ch([$($targets:expr),*], [$($controls:expr),*]), $($rest:tt)*) => { $builder.ch_gates(vec![$($targets),*], vec![$($controls),*]); $crate::circuit_internal!($builder, $($rest)*); };
    ($builder:ident, ch([$($targets:expr),*], $control:expr), $($rest:tt)*) => { $builder.ch_gates(vec![$($targets),*], vec![$control]); $crate::circuit_internal!($builder, $($rest)*); };
    ($builder:ident, ch($target:expr, [$($controls:expr),*]), $($rest:tt)*) => { $builder.ch_gates(vec![$target], vec![$($controls),*]); $crate::circuit_internal!($builder, $($rest)*); };
    ($builder:ident, ch($target:expr, $control:expr), $($rest:tt)*) => { $builder.ch_gates(vec![$target], vec![$control]); $crate::circuit_internal!($builder, $($rest)*); };
    ($builder:ident, cx([$($targets:expr),*], [$($controls:expr),*]), $($rest:tt)*) => { $builder.cx_gates(vec![$($targets),*], vec![$($controls),*]); $crate::circuit_internal!($builder, $($rest)*); };
    ($builder:ident, cx([$($targets:expr),*], $control:expr), $($rest:tt)*) => { $builder.cx_gates(vec![$($targets),*], vec![$control]); $crate::circuit_internal!($builder, $($rest)*); };
    ($builder:ident, cx($target:expr, [$($controls:expr),*]), $($rest:tt)*) => { $builder.cx_gates(vec![$target], vec![$($controls),*]); $crate::circuit_internal!($builder, $($rest)*); };
    ($builder:ident, cx($target:expr, $control:expr), $($rest:tt)*) => { $builder.cx_gates(vec![$target], vec![$control]); $crate::circuit_internal!($builder, $($rest)*); };
    ($builder:ident, cy([$($targets:expr),*], [$($controls:expr),*]), $($rest:tt)*) => { $builder.cy_gates(vec![$($targets),*], vec![$($controls),*]); $crate::circuit_internal!($builder, $($rest)*); };
    ($builder:ident, cy([$($targets:expr),*], $control:expr), $($rest:tt)*) => { $builder.cy_gates(vec![$($targets),*], vec![$control]); $crate::circuit_internal!($builder, $($rest)*); };
    ($builder:ident, cy($target:expr, [$($controls:expr),*]), $($rest:tt)*) => { $builder.cy_gates(vec![$target], vec![$($controls),*]); $crate::circuit_internal!($builder, $($rest)*); };
    ($builder:ident, cy($target:expr, $control:expr), $($rest:tt)*) => { $builder.cy_gates(vec![$target], vec![$control]); $crate::circuit_internal!($builder, $($rest)*); };
    ($builder:ident, cz([$($targets:expr),*], [$($controls:expr),*]), $($rest:tt)*) => { $builder.cz_gates(vec![$($targets),*], vec![$($controls),*]); $crate::circuit_internal!($builder, $($rest)*); };
    ($builder:ident, cz([$($targets:expr),*], $control:expr), $($rest:tt)*) => { $builder.cz_gates(vec![$($targets),*], vec![$control]); $crate::circuit_internal!($builder, $($rest)*); };
    ($builder:ident, cz($target:expr, [$($controls:expr),*]), $($rest:tt)*) => { $builder.cz_gates(vec![$target], vec![$($controls),*]); $crate::circuit_internal!($builder, $($rest)*); };
    ($builder:ident, cz($target:expr, $control:expr), $($rest:tt)*) => { $builder.cz_gates(vec![$target], vec![$control]); $crate::circuit_internal!($builder, $($rest)*); };
    ($builder:ident, cs([$($targets:expr),*], [$($controls:expr),*]), $($rest:tt)*) => { $builder.cs_gates(vec![$($targets),*], vec![$($controls),*]); $crate::circuit_internal!($builder, $($rest)*); };
    ($builder:ident, cs([$($targets:expr),*], $control:expr), $($rest:tt)*) => { $builder.cs_gates(vec![$($targets),*], vec![$control]); $crate::circuit_internal!($builder, $($rest)*); };
    ($builder:ident, cs($target:expr, [$($controls:expr),*]), $($rest:tt)*) => { $builder.cs_gates(vec![$target], vec![$($controls),*]); $crate::circuit_internal!($builder, $($rest)*); };
    ($builder:ident, cs($target:expr, $control:expr), $($rest:tt)*) => { $builder.cs_gates(vec![$target], vec![$control]); $crate::circuit_internal!($builder, $($rest)*); };
    ($builder:ident, csdag([$($targets:expr),*], [$($controls:expr),*]), $($rest:tt)*) => { $builder.csdag_gates(vec![$($targets),*], vec![$($controls),*]); $crate::circuit_internal!($builder, $($rest)*); };
    ($builder:ident, csdag([$($targets:expr),*], $control:expr), $($rest:tt)*) => { $builder.csdag_gates(vec![$($targets),*], vec![$control]); $crate::circuit_internal!($builder, $($rest)*); };
    ($builder:ident, csdag($target:expr, [$($controls:expr),*]), $($rest:tt)*) => { $builder.csdag_gates(vec![$target], vec![$($controls),*]); $crate::circuit_internal!($builder, $($rest)*); };
    ($builder:ident, csdag($target:expr, $control:expr), $($rest:tt)*) => { $builder.csdag_gates(vec![$target], vec![$control]); $crate::circuit_internal!($builder, $($rest)*); };
    ($builder:ident, ct([$($targets:expr),*], [$($controls:expr),*]), $($rest:tt)*) => { $builder.ct_gates(vec![$($targets),*], vec![$($controls),*]); $crate::circuit_internal!($builder, $($rest)*); };
    ($builder:ident, ct([$($targets:expr),*], $control:expr), $($rest:tt)*) => { $builder.ct_gates(vec![$($targets),*], vec![$control]); $crate::circuit_internal!($builder, $($rest)*); };
    ($builder:ident, ct($target:expr, [$($controls:expr),*]), $($rest:tt)*) => { $builder.ct_gates(vec![$target], vec![$($controls),*]); $crate::circuit_internal!($builder, $($rest)*); };
    ($builder:ident, ct($target:expr, $control:expr), $($rest:tt)*) => { $builder.ct_gates(vec![$target], vec![$control]); $crate::circuit_internal!($builder, $($rest)*); };
    ($builder:ident, ctdag([$($targets:expr),*], [$($controls:expr),*]), $($rest:tt)*) => { $builder.ctdag_gates(vec![$($targets),*], vec![$($controls),*]); $crate::circuit_internal!($builder, $($rest)*); };
    ($builder:ident, ctdag([$($targets:expr),*], $control:expr), $($rest:tt)*) => { $builder.ctdag_gates(vec![$($targets),*], vec![$control]); $crate::circuit_internal!($builder, $($rest)*); };
    ($builder:ident, ctdag($target:expr, [$($controls:expr),*]), $($rest:tt)*) => { $builder.ctdag_gates(vec![$target], vec![$($controls),*]); $crate::circuit_internal!($builder, $($rest)*); };
    ($builder:ident, ctdag($target:expr, $control:expr), $($rest:tt)*) => { $builder.ctdag_gates(vec![$target], vec![$control]); $crate::circuit_internal!($builder, $($rest)*); };

    // Gates with angles
    ($builder:ident, rx([$($qubits:expr),*], $angle:expr), $($rest:tt)*) => { $builder.rx_gates(vec![$($qubits),*], $angle); $crate::circuit_internal!($builder, $($rest)*); };
    ($builder:ident, ry([$($qubits:expr),*], $angle:expr), $($rest:tt)*) => { $builder.ry_gates(vec![$($qubits),*], $angle); $crate::circuit_internal!($builder, $($rest)*); };
    ($builder:ident, rz([$($qubits:expr),*], $angle:expr), $($rest:tt)*) => { $builder.rz_gates(vec![$($qubits),*], $angle); $crate::circuit_internal!($builder, $($rest)*); };
    ($builder:ident, p([$($qubits:expr),*], $angle:expr), $($rest:tt)*) => { $builder.p_gates(vec![$($qubits),*], $angle); $crate::circuit_internal!($builder, $($rest)*); };
    ($builder:ident, ry_phase([$($qubits:expr),*], $theta:expr, $phi:expr), $($rest:tt)*) => { $builder.ry_phase_gates(vec![$($qubits),*], $theta, $phi); $crate::circuit_internal!($builder, $($rest)*); };
    ($builder:ident, ry_phase_dag([$($qubits:expr),*], $theta:expr, $phi:expr), $($rest:tt)*) => { $builder.ry_phase_dag_gates(vec![$($qubits),*], $theta, $phi); $crate::circuit_internal!($builder, $($rest)*); };
    ($builder:ident, rx($qubit:expr, $angle:expr), $($rest:tt)*) => { $builder.rx_gate($qubit, $angle); $crate::circuit_internal!($builder, $($rest)*); };
    ($builder:ident, ry($qubit:expr, $angle:expr), $($rest:tt)*) => { $builder.ry_gate($qubit, $angle); $crate::circuit_internal!($builder, $($rest)*); };
    ($builder:ident, rz($qubit:expr, $angle:expr), $($rest:tt)*) => { $builder.rz_gate($qubit, $angle); $crate::circuit_internal!($builder, $($rest)*); };
    ($builder:ident, p($qubit:expr, $angle:expr), $($rest:tt)*) => { $builder.p_gate($qubit, $angle); $crate::circuit_internal!($builder, $($rest)*); };
    ($builder:ident, ry_phase($qubit:expr, $theta:expr, $phi:expr), $($rest:tt)*) => { $builder.ry_phase_gate($qubit, $theta, $phi); $crate::circuit_internal!($builder, $($rest)*); };
    ($builder:ident, ry_phase_dag($qubit:expr, $theta:expr, $phi:expr), $($rest:tt)*) => { $builder.ry_phase_dag_gate($qubit, $theta, $phi); $crate::circuit_internal!($builder, $($rest)*); };

    // --- Controlled angle gate overloading ---
    ($builder:ident, crx([$($targets:expr),*], [$($controls:expr),*], $angle:expr), $($rest:tt)*) => { $builder.crx_gates(vec![$($targets),*], vec![$($controls),*], $angle); $crate::circuit_internal!($builder, $($rest)*); };
    ($builder:ident, crx([$($targets:expr),*], $control:expr, $angle:expr), $($rest:tt)*) => { $builder.crx_gates(vec![$($targets),*], vec![$control], $angle); $crate::circuit_internal!($builder, $($rest)*); };
    ($builder:ident, crx($target:expr, [$($controls:expr),*], $angle:expr), $($rest:tt)*) => { $builder.crx_gates(vec![$target], vec![$($controls),*], $angle); $crate::circuit_internal!($builder, $($rest)*); };
    ($builder:ident, crx($target:expr, $control:expr, $angle:expr), $($rest:tt)*) => { $builder.crx_gates(vec![$target], vec![$control], $angle); $crate::circuit_internal!($builder, $($rest)*); };
    ($builder:ident, cry([$($targets:expr),*], [$($controls:expr),*], $angle:expr), $($rest:tt)*) => { $builder.cry_gates(vec![$($targets),*], vec![$($controls),*], $angle); $crate::circuit_internal!($builder, $($rest)*); };
    ($builder:ident, cry([$($targets:expr),*], $control:expr, $angle:expr), $($rest:tt)*) => { $builder.cry_gates(vec![$($targets),*], vec![$control], $angle); $crate::circuit_internal!($builder, $($rest)*); };
    ($builder:ident, cry($target:expr, [$($controls:expr),*], $angle:expr), $($rest:tt)*) => { $builder.cry_gates(vec![$target], vec![$($controls),*], $angle); $crate::circuit_internal!($builder, $($rest)*); };
    ($builder:ident, cry($target:expr, $control:expr, $angle:expr), $($rest:tt)*) => { $builder.cry_gates(vec![$target], vec![$control], $angle); $crate::circuit_internal!($builder, $($rest)*); };
    ($builder:ident, crz([$($targets:expr),*], [$($controls:expr),*], $angle:expr), $($rest:tt)*) => { $builder.crz_gates(vec![$($targets),*], vec![$($controls),*], $angle); $crate::circuit_internal!($builder, $($rest)*); };
    ($builder:ident, crz([$($targets:expr),*], $control:expr, $angle:expr), $($rest:tt)*) => { $builder.crz_gates(vec![$($targets),*], vec![$control], $angle); $crate::circuit_internal!($builder, $($rest)*); };
    ($builder:ident, crz($target:expr, [$($controls:expr),*], $angle:expr), $($rest:tt)*) => { $builder.crz_gates(vec![$target], vec![$($controls),*], $angle); $crate::circuit_internal!($builder, $($rest)*); };
    ($builder:ident, crz($target:expr, $control:expr, $angle:expr), $($rest:tt)*) => { $builder.crz_gates(vec![$target], vec![$control], $angle); $crate::circuit_internal!($builder, $($rest)*); };
    ($builder:ident, cp([$($targets:expr),*], [$($controls:expr),*], $angle:expr), $($rest:tt)*) => { $builder.cp_gates(vec![$($targets),*], vec![$($controls),*], $angle); $crate::circuit_internal!($builder, $($rest)*); };
    ($builder:ident, cp([$($targets:expr),*], $control:expr, $angle:expr), $($rest:tt)*) => { $builder.cp_gates(vec![$($targets),*], vec![$control], $angle); $crate::circuit_internal!($builder, $($rest)*); };
    ($builder:ident, cp($target:expr, [$($controls:expr),*], $angle:expr), $($rest:tt)*) => { $builder.cp_gates(vec![$target], vec![$($controls),*], $angle); $crate::circuit_internal!($builder, $($rest)*); };
    ($builder:ident, cp($target:expr, $control:expr, $angle:expr), $($rest:tt)*) => { $builder.cp_gates(vec![$target], vec![$control], $angle); $crate::circuit_internal!($builder, $($rest)*); };
    ($builder:ident, cry_phase([$($targets:expr),*], [$($controls:expr),*], $theta:expr, $phi:expr), $($rest:tt)*) => { $builder.cry_phase_gates(vec![$($targets),*], vec![$($controls),*], $theta, $phi); $crate::circuit_internal!($builder, $($rest)*); };
    ($builder:ident, cry_phase([$($targets:expr),*], $control:expr, $theta:expr, $phi:expr), $($rest:tt)*) => { $builder.cry_phase_gates(vec![$($targets),*], vec![$control], $theta, $phi); $crate::circuit_internal!($builder, $($rest)*); };
    ($builder:ident, cry_phase($target:expr, [$($controls:expr),*], $theta:expr, $phi:expr), $($rest:tt)*) => { $builder.cry_phase_gates(vec![$target], vec![$($controls),*], $theta, $phi); $crate::circuit_internal!($builder, $($rest)*); };
    ($builder:ident, cry_phase($target:expr, $control:expr, $theta:expr, $phi:expr), $($rest:tt)*) => { $builder.cry_phase_gates(vec![$target], vec![$control], $theta, $phi); $crate::circuit_internal!($builder, $($rest)*); };
    ($builder:ident, cry_phase_dag([$($targets:expr),*], [$($controls:expr),*], $theta:expr, $phi:expr), $($rest:tt)*) => { $builder.cry_phase_dag_gates(vec![$($targets),*], vec![$($controls),*], $theta, $phi); $crate::circuit_internal!($builder, $($rest)*); };
    ($builder:ident, cry_phase_dag([$($targets:expr),*], $control:expr, $theta:expr, $phi:expr), $($rest:tt)*) => { $builder.cry_phase_dag_gates(vec![$($targets),*], vec![$control], $theta, $phi); $crate::circuit_internal!($builder, $($rest)*); };
    ($builder:ident, cry_phase_dag($target:expr, [$($controls:expr),*], $theta:expr, $phi:expr), $($rest:tt)*) => { $builder.cry_phase_dag_gates(vec![$target], vec![$($controls),*], $theta, $phi); $crate::circuit_internal!($builder, $($rest)*); };
    ($builder:ident, cry_phase_dag($target:expr, $control:expr, $theta:expr, $phi:expr), $($rest:tt)*) => { $builder.cry_phase_dag_gates(vec![$target], vec![$control], $theta, $phi); $crate::circuit_internal!($builder, $($rest)*); };

    // --- Unitary and controlled unitary gates ---
    ($builder:ident, unitary([$($qubits:expr),*], $matrix:expr), $($rest:tt)*) => { $builder.unitary_gates(vec![$($qubits),*], $matrix).unwrap(); $crate::circuit_internal!($builder, $($rest)*); };
    ($builder:ident, unitary($qubit:expr, $matrix:expr), $($rest:tt)*) => { $builder.unitary_gate($qubit, $matrix).unwrap(); $crate::circuit_internal!($builder, $($rest)*); };
    ($builder:ident, cunitary([$($targets:expr),*], [$($controls:expr),*], $matrix:expr), $($rest:tt)*) => { $builder.cunitary_gates(vec![$($targets),*], vec![$($controls),*], $matrix).unwrap(); $crate::circuit_internal!($builder, $($rest)*); };
    ($builder:ident, cunitary([$($targets:expr),*], $control:expr, $matrix:expr), $($rest:tt)*) => { $builder.cunitary_gates(vec![$($targets),*], vec![$control], $matrix).unwrap(); $crate::circuit_internal!($builder, $($rest)*); };
    ($builder:ident, cunitary($target:expr, [$($controls:expr),*], $matrix:expr), $($rest:tt)*) => { $builder.cunitary_gates(vec![$target], vec![$($controls),*], $matrix).unwrap(); $crate::circuit_internal!($builder, $($rest)*); };
    ($builder:ident, cunitary($target:expr, $control:expr, $matrix:expr), $($rest:tt)*) => { $builder.cunitary_gates(vec![$target], vec![$control], $matrix).unwrap(); $crate::circuit_internal!($builder, $($rest)*); };

    // --- Special gates ---
    ($builder:ident, toffoli($target:expr, $control1:expr, $control2:expr), $($rest:tt)*) => { $builder.toffoli_gate($target, $control1, $control2); $crate::circuit_internal!($builder, $($rest)*); };
    ($builder:ident, cswap($target1:expr, $target2:expr, [$($controls:expr),*]), $($rest:tt)*) => { $builder.cswap_gate($target1, $target2, vec![$($controls),*]); $crate::circuit_internal!($builder, $($rest)*); };
    ($builder:ident, cswap($target1:expr, $target2:expr, $control:expr), $($rest:tt)*) => { $builder.cswap_gate($target1, $target2, vec![$control]); $crate::circuit_internal!($builder, $($rest)*); };
    ($builder:ident, matchgate($target1:expr, $theta:expr, $phi1:expr, $phi2:expr), $($rest:tt)*) => { $builder.matchgate($target1, $theta, $phi1, $phi2); $crate::circuit_internal!($builder, $($rest)*); };
    ($builder:ident, cmatchgate($target1:expr, [$($controls:expr),*], $theta:expr, $phi1:expr, $phi2:expr), $($rest:tt)*) => { $builder.cmatchgate($target1, vec![$($controls),*], $theta, $phi1, $phi2); $crate::circuit_internal!($builder, $($rest)*); };
    ($builder:ident, cmatchgate($target1:expr, $control:expr, $theta:expr, $phi1:expr, $phi2:expr), $($rest:tt)*) => { $builder.cmatchgate($target1, vec![$control], $theta, $phi1, $phi2); $crate::circuit_internal!($builder, $($rest)*); };

    // --- Measurement rules ---
    ($builder:ident, measurex([$($qubits:expr),*]), $($rest:tt)*) => { $builder.measure_gate($crate::components::measurement::MeasurementBasis::X, vec![$($qubits),*]); $crate::circuit_internal!($builder, $($rest)*); };
    ($builder:ident, measurex($qubit:expr), $($rest:tt)*) => { $builder.measure_gate($crate::components::measurement::MeasurementBasis::X, vec![$qubit]); $crate::circuit_internal!($builder, $($rest)*); };
    ($builder:ident, measurey([$($qubits:expr),*]), $($rest:tt)*) => { $builder.measure_gate($crate::components::measurement::MeasurementBasis::Y, vec![$($qubits),*]); $crate::circuit_internal!($builder, $($rest)*); };
    ($builder:ident, measurey($qubit:expr), $($rest:tt)*) => { $builder.measure_gate($crate::components::measurement::MeasurementBasis::Y, vec![$qubit]); $crate::circuit_internal!($builder, $($rest)*); };
    ($builder:ident, measurez([$($qubits:expr),*]), $($rest:tt)*) => { $builder.measure_gate($crate::components::measurement::MeasurementBasis::Computational, vec![$($qubits),*]); $crate::circuit_internal!($builder, $($rest)*); };
    ($builder:ident, measurez($qubit:expr), $($rest:tt)*) => { $builder.measure_gate($crate::components::measurement::MeasurementBasis::Computational, vec![$qubit]); $crate::circuit_internal!($builder, $($rest)*); };
    ($builder:ident, measure_custom([$($qubits:expr),*], $matrix:expr), $($rest:tt)*) => { $builder.measure_gate($crate::components::measurement::MeasurementBasis::Custom($matrix), vec![$($qubits),*]); $crate::circuit_internal!($builder, $($rest)*); };
    ($builder:ident, measure_custom($qubit:expr, $matrix:expr), $($rest:tt)*) => { $builder.measure_gate($crate::components::measurement::MeasurementBasis::Custom($matrix), vec![$qubit]); $crate::circuit_internal!($builder, $($rest)*); };

    // --- Terminating Rules (no trailing comma) ---

    // Multi-qubit gates
    ($builder:ident, h([$($qubits:expr),*])) => { $builder.h_gates(vec![$($qubits),*]); };
    ($builder:ident, x([$($qubits:expr),*])) => { $builder.x_gates(vec![$($qubits),*]); };
    ($builder:ident, y([$($qubits:expr),*])) => { $builder.y_gates(vec![$($qubits),*]); };
    ($builder:ident, z([$($qubits:expr),*])) => { $builder.z_gates(vec![$($qubits),*]); };
    ($builder:ident, s([$($qubits:expr),*])) => { $builder.s_gates(vec![$($qubits),*]); };
    ($builder:ident, t([$($qubits:expr),*])) => { $builder.t_gates(vec![$($qubits),*]); };
    ($builder:ident, id([$($qubits:expr),*])) => { $builder.id_gates(vec![$($qubits),*]); };
    ($builder:ident, sdag([$($qubits:expr),*])) => { $builder.sdag_gates(vec![$($qubits),*]); };
    ($builder:ident, tdag([$($qubits:expr),*])) => { $builder.tdag_gates(vec![$($qubits),*]); };
    ($builder:ident, pauli_string($pauli_string:expr)) => { $builder.pauli_string_gate($pauli_string); };

    // Single-qubit gates
    ($builder:ident, h($qubit:expr)) => { $builder.h_gate($qubit); };
    ($builder:ident, x($qubit:expr)) => { $builder.x_gate($qubit); };
    ($builder:ident, y($qubit:expr)) => { $builder.y_gate($qubit); };
    ($builder:ident, z($qubit:expr)) => { $builder.z_gate($qubit); };
    ($builder:ident, s($qubit:expr)) => { $builder.s_gate($qubit); };
    ($builder:ident, t($qubit:expr)) => { $builder.t_gate($qubit); };
    ($builder:ident, id($qubit:expr)) => { $builder.id_gate($qubit); };
    ($builder:ident, sdag($qubit:expr)) => { $builder.sdag_gate($qubit); };
    ($builder:ident, tdag($qubit:expr)) => { $builder.tdag_gate($qubit); };

    // Two-argument gates
    ($builder:ident, cnot($arg1:expr, $arg2:expr)) => { $builder.cnot_gate($arg1, $arg2); };
    ($builder:ident, swap($arg1:expr, $arg2:expr)) => { $builder.swap_gate($arg1, $arg2); };

    // Controlled gates
    ($builder:ident, ch([$($targets:expr),*], [$($controls:expr),*])) => { $builder.ch_gates(vec![$($targets),*], vec![$($controls),*]); };
    ($builder:ident, ch([$($targets:expr),*], $control:expr)) => { $builder.ch_gates(vec![$($targets),*], vec![$control]); };
    ($builder:ident, ch($target:expr, [$($controls:expr),*])) => { $builder.ch_gates(vec![$target], vec![$($controls),*]); };
    ($builder:ident, ch($target:expr, $control:expr)) => { $builder.ch_gates(vec![$target], vec![$control]); };
    ($builder:ident, cx([$($targets:expr),*], [$($controls:expr),*])) => { $builder.cx_gates(vec![$($targets),*], vec![$($controls),*]); };
    ($builder:ident, cx([$($targets:expr),*], $control:expr)) => { $builder.cx_gates(vec![$($targets),*], vec![$control]); };
    ($builder:ident, cx($target:expr, [$($controls:expr),*])) => { $builder.cx_gates(vec![$target], vec![$($controls),*]); };
    ($builder:ident, cx($target:expr, $control:expr)) => { $builder.cx_gates(vec![$target], vec![$control]); };
    ($builder:ident, cy([$($targets:expr),*], [$($controls:expr),*])) => { $builder.cy_gates(vec![$($targets),*], vec![$($controls),*]); };
    ($builder:ident, cy([$($targets:expr),*], $control:expr)) => { $builder.cy_gates(vec![$($targets),*], vec![$control]); };
    ($builder:ident, cy($target:expr, [$($controls:expr),*])) => { $builder.cy_gates(vec![$target], vec![$($controls),*]); };
    ($builder:ident, cy($target:expr, $control:expr)) => { $builder.cy_gates(vec![$target], vec![$control]); };
    ($builder:ident, cz([$($targets:expr),*], [$($controls:expr),*])) => { $builder.cz_gates(vec![$($targets),*], vec![$($controls),*]); };
    ($builder:ident, cz([$($targets:expr),*], $control:expr)) => { $builder.cz_gates(vec![$($targets),*], vec![$control]); };
    ($builder:ident, cz($target:expr, [$($controls:expr),*])) => { $builder.cz_gates(vec![$target], vec![$($controls),*]); };
    ($builder:ident, cz($target:expr, $control:expr)) => { $builder.cz_gates(vec![$target], vec![$control]); };
    ($builder:ident, cs([$($targets:expr),*], [$($controls:expr),*])) => { $builder.cs_gates(vec![$($targets),*], vec![$($controls),*]); };
    ($builder:ident, cs([$($targets:expr),*], $control:expr)) => { $builder.cs_gates(vec![$($targets),*], vec![$control]); };
    ($builder:ident, cs($target:expr, [$($controls:expr),*])) => { $builder.cs_gates(vec![$target], vec![$($controls),*]); };
    ($builder:ident, cs($target:expr, $control:expr)) => { $builder.cs_gates(vec![$target], vec![$control]); };
    ($builder:ident, csdag([$($targets:expr),*], [$($controls:expr),*])) => { $builder.csdag_gates(vec![$($targets),*], vec![$($controls),*]); };
    ($builder:ident, csdag([$($targets:expr),*], $control:expr)) => { $builder.csdag_gates(vec![$($targets),*], vec![$control]); };
    ($builder:ident, csdag($target:expr, [$($controls:expr),*])) => { $builder.csdag_gates(vec![$target], vec![$($controls),*]); };
    ($builder:ident, csdag($target:expr, $control:expr)) => { $builder.csdag_gates(vec![$target], vec![$control]); };
    ($builder:ident, ct([$($targets:expr),*], [$($controls:expr),*])) => { $builder.ct_gates(vec![$($targets),*], vec![$($controls),*]); };
    ($builder:ident, ct([$($targets:expr),*], $control:expr)) => { $builder.ct_gates(vec![$($targets),*], vec![$control]); };
    ($builder:ident, ct($target:expr, [$($controls:expr),*])) => { $builder.ct_gates(vec![$target], vec![$($controls),*]); };
    ($builder:ident, ct($target:expr, $control:expr)) => { $builder.ct_gates(vec![$target], vec![$control]); };
    ($builder:ident, ctdag([$($targets:expr),*], [$($controls:expr),*])) => { $builder.ctdag_gates(vec![$($targets),*], vec![$($controls),*]); };
    ($builder:ident, ctdag([$($targets:expr),*], $control:expr)) => { $builder.ctdag_gates(vec![$($targets),*], vec![$control]); };
    ($builder:ident, ctdag($target:expr, [$($controls:expr),*])) => { $builder.ctdag_gates(vec![$target], vec![$($controls),*]); };
    ($builder:ident, ctdag($target:expr, $control:expr)) => { $builder.ctdag_gates(vec![$target], vec![$control]); };

    // Gates with angles
    ($builder:ident, rx([$($qubits:expr),*], $angle:expr)) => { $builder.rx_gates(vec![$($qubits),*], $angle); };
    ($builder:ident, ry([$($qubits:expr),*], $angle:expr)) => { $builder.ry_gates(vec![$($qubits),*], $angle); };
    ($builder:ident, rz([$($qubits:expr),*], $angle:expr)) => { $builder.rz_gates(vec![$($qubits),*], $angle); };
    ($builder:ident, p([$($qubits:expr),*], $angle:expr)) => { $builder.p_gates(vec![$($qubits),*], $angle); };
    ($builder:ident, ry_phase([$($qubits:expr),*], $theta:expr, $phi:expr)) => { $builder.ry_phase_gates(vec![$($qubits),*], $theta, $phi); };
    ($builder:ident, ry_phase_dag([$($qubits:expr),*], $theta:expr, $phi:expr)) => { $builder.ry_phase_dag_gates(vec![$($qubits),*], $theta, $phi); };
    ($builder:ident, rx($qubit:expr, $angle:expr)) => { $builder.rx_gate($qubit, $angle); };
    ($builder:ident, ry($qubit:expr, $angle:expr)) => { $builder.ry_gate($qubit, $angle); };
    ($builder:ident, rz($qubit:expr, $angle:expr)) => { $builder.rz_gate($qubit, $angle); };
    ($builder:ident, p($qubit:expr, $angle:expr)) => { $builder.p_gate($qubit, $angle); };
    ($builder:ident, ry_phase($qubit:expr, $theta:expr, $phi:expr)) => { $builder.ry_phase_gate($qubit, $theta, $phi); };
    ($builder:ident, ry_phase_dag($qubit:expr, $theta:expr, $phi:expr)) => { $builder.ry_phase_dag_gate($qubit, $theta, $phi); };

    // Controlled angle gates
    ($builder:ident, crx([$($targets:expr),*], [$($controls:expr),*], $angle:expr)) => { $builder.crx_gates(vec![$($targets),*], vec![$($controls),*], $angle); };
    ($builder:ident, crx([$($targets:expr),*], $control:expr, $angle:expr)) => { $builder.crx_gates(vec![$($targets),*], vec![$control], $angle); };
    ($builder:ident, crx($target:expr, [$($controls:expr),*], $angle:expr)) => { $builder.crx_gates(vec![$target], vec![$($controls),*], $angle); };
    ($builder:ident, crx($target:expr, $control:expr, $angle:expr)) => { $builder.crx_gates(vec![$target], vec![$control], $angle); };
    ($builder:ident, cry([$($targets:expr),*], [$($controls:expr),*], $angle:expr)) => { $builder.cry_gates(vec![$($targets),*], vec![$($controls),*], $angle); };
    ($builder:ident, cry([$($targets:expr),*], $control:expr, $angle:expr)) => { $builder.cry_gates(vec![$($targets),*], vec![$control], $angle); };
    ($builder:ident, cry($target:expr, [$($controls:expr),*], $angle:expr)) => { $builder.cry_gates(vec![$target], vec![$($controls),*], $angle); };
    ($builder:ident, cry($target:expr, $control:expr, $angle:expr)) => { $builder.cry_gates(vec![$target], vec![$control], $angle); };
    ($builder:ident, crz([$($targets:expr),*], [$($controls:expr),*], $angle:expr)) => { $builder.crz_gates(vec![$($targets),*], vec![$($controls),*], $angle); };
    ($builder:ident, crz([$($targets:expr),*], $control:expr, $angle:expr)) => { $builder.crz_gates(vec![$($targets),*], vec![$control], $angle); };
    ($builder:ident, crz($target:expr, [$($controls:expr),*], $angle:expr)) => { $builder.crz_gates(vec![$target], vec![$($controls),*], $angle); };
    ($builder:ident, crz($target:expr, $control:expr, $angle:expr)) => { $builder.crz_gates(vec![$target], vec![$control], $angle); };
    ($builder:ident, cp([$($targets:expr),*], [$($controls:expr),*], $angle:expr)) => { $builder.cp_gates(vec![$($targets),*], vec![$($controls),*], $angle); };
    ($builder:ident, cp([$($targets:expr),*], $control:expr, $angle:expr)) => { $builder.cp_gates(vec![$($targets),*], vec![$control], $angle); };
    ($builder:ident, cp($target:expr, [$($controls:expr),*], $angle:expr)) => { $builder.cp_gates(vec![$target], vec![$($controls),*], $angle); };
    ($builder:ident, cp($target:expr, $control:expr, $angle:expr)) => { $builder.cp_gates(vec![$target], vec![$control], $angle); };
    ($builder:ident, cry_phase([$($targets:expr),*], [$($controls:expr),*], $theta:expr, $phi:expr)) => { $builder.cry_phase_gates(vec![$($targets),*], vec![$($controls),*], $theta, $phi); };
    ($builder:ident, cry_phase([$($targets:expr),*], $control:expr, $theta:expr, $phi:expr)) => { $builder.cry_phase_gates(vec![$($targets),*], vec![$control], $theta, $phi); };
    ($builder:ident, cry_phase($target:expr, [$($controls:expr),*], $theta:expr, $phi:expr)) => { $builder.cry_phase_gates(vec![$target], vec![$($controls),*], $theta, $phi); };
    ($builder:ident, cry_phase($target:expr, $control:expr, $theta:expr, $phi:expr)) => { $builder.cry_phase_gates(vec![$target], vec![$control], $theta, $phi); };
    ($builder:ident, cry_phase_dag([$($targets:expr),*], [$($controls:expr),*], $theta:expr, $phi:expr)) => { $builder.cry_phase_dag_gates(vec![$($targets),*], vec![$($controls),*], $theta, $phi); };
    ($builder:ident, cry_phase_dag([$($targets:expr),*], $control:expr, $theta:expr, $phi:expr)) => { $builder.cry_phase_dag_gates(vec![$($targets),*], vec![$control], $theta, $phi); };
    ($builder:ident, cry_phase_dag($target:expr, [$($controls:expr),*], $theta:expr, $phi:expr)) => { $builder.cry_phase_dag_gates(vec![$target], vec![$($controls),*], $theta, $phi); };
    ($builder:ident, cry_phase_dag($target:expr, $control:expr, $theta:expr, $phi:expr)) => { $builder.cry_phase_dag_gates(vec![$target], vec![$control], $theta, $phi); };

    // Unitary and Controlled Unitary Gates
    ($builder:ident, unitary([$($qubits:expr),*], $matrix:expr)) => { $builder.unitary_gates(vec![$($qubits),*], $matrix).unwrap(); };
    ($builder:ident, unitary($qubit:expr, $matrix:expr)) => { $builder.unitary_gate($qubit, $matrix).unwrap(); };
    ($builder:ident, cunitary([$($targets:expr),*], [$($controls:expr),*], $matrix:expr)) => { $builder.cunitary_gates(vec![$($targets),*], vec![$($controls),*], $matrix).unwrap(); };
    ($builder:ident, cunitary([$($targets:expr),*], $control:expr, $matrix:expr)) => { $builder.cunitary_gates(vec![$($targets),*], vec![$control], $matrix).unwrap(); };
    ($builder:ident, cunitary($target:expr, [$($controls:expr),*], $matrix:expr)) => { $builder.cunitary_gates(vec![$target], vec![$($controls),*], $matrix).unwrap(); };
    ($builder:ident, cunitary($target:expr, $control:expr, $matrix:expr)) => { $builder.cunitary_gates(vec![$target], vec![$control], $matrix).unwrap(); };

    // Special Gates
    ($builder:ident, toffoli($target:expr, $control1:expr, $control2:expr)) => { $builder.toffoli_gate($target, $control1, $control2); };
    ($builder:ident, cswap($target1:expr, $target2:expr, [$($controls:expr),*])) => { $builder.cswap_gate($target1, $target2, vec![$($controls),*]); };
    ($builder:ident, cswap($target1:expr, $target2:expr, $control:expr)) => { $builder.cswap_gate($target1, $target2, vec![$control]); };
    ($builder:ident, matchgate($target1:expr, $theta:expr, $phi1:expr, $phi2:expr)) => { $builder.matchgate($target1, $theta, $phi1, $phi2); };
    ($builder:ident, cmatchgate($target1:expr, [$($controls:expr),*], $theta:expr, $phi1:expr, $phi2:expr)) => { $builder.cmatchgate($target1, vec![$($controls),*], $theta, $phi1, $phi2); };
    ($builder:ident, cmatchgate($target1:expr, $control:expr, $theta:expr, $phi1:expr, $phi2:expr)) => { $builder.cmatchgate($target1, vec![$control], $theta, $phi1, $phi2); };

    // Measurement gates
    ($builder:ident, measurex([$($qubits:expr),*])) => { $builder.measure_gate($crate::components::measurement::MeasurementBasis::X, vec![$($qubits),*]); };
    ($builder:ident, measurex($qubit:expr)) => { $builder.measure_gate($crate::components::measurement::MeasurementBasis::X, vec![$qubit]); };
    ($builder:ident, measurey([$($qubits:expr),*])) => { $builder.measure_gate($crate::components::measurement::MeasurementBasis::Y, vec![$($qubits),*]); };
    ($builder:ident, measurey($qubit:expr)) => { $builder.measure_gate($crate::components::measurement::MeasurementBasis::Y, vec![$qubit]); };
    ($builder:ident, measurez([$($qubits:expr),*])) => { $builder.measure_gate($crate::components::measurement::MeasurementBasis::Computational, vec![$($qubits),*]); };
    ($builder:ident, measurez($qubit:expr)) => { $builder.measure_gate($crate::components::measurement::MeasurementBasis::Computational, vec![$qubit]); };
    ($builder:ident, measure_custom([$($qubits:expr),*], $matrix:expr)) => { $builder.measure_gate($crate::components::measurement::MeasurementBasis::Custom($matrix), vec![$($qubits),*]); };
    ($builder:ident, measure_custom($qubit:expr, $matrix:expr)) => { $builder.measure_gate($crate::components::measurement::MeasurementBasis::Custom($matrix), vec![$qubit]); };

    // --- Error Handling ---
    // This is a compile-time error, as it indicates a failure to match any known gate pattern.
    ($builder:ident, $bad_token:tt, $($rest:tt)*) => {
        compile_error!(concat!("Unrecognised gate or syntax: `", stringify!($bad_token), "`"));
    };
}