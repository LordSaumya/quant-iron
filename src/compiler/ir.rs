use crate::{compiler::qasm::QasmInstruction, components::measurement::MeasurementBasis, errors::CompilerError};
use num_complex::Complex;

#[derive(Debug, Clone, PartialEq)]
pub enum InstructionIR {
    // GATES
    /// IR representation for a Hadamard gate on qubit index with controls
    Hadamard(usize, Vec<usize>),
    /// IR representation for a Pauli-X gate on qubit index with controls
    PauliX(usize, Vec<usize>),
    /// IR representation for a Pauli-Y gate on qubit index with controls
    PauliY(usize, Vec<usize>),
    /// IR representation for a Pauli-Z gate on qubit index with controls
    PauliZ(usize, Vec<usize>),
    /// IR representation for a Phase S gate on qubit index with controls
    S(usize, Vec<usize>),
    /// IR representation for a Phase T gate on qubit index with controls
    T(usize, Vec<usize>),
    /// IR representation for a Phase S-dagger gate on qubit index with controls
    Sdg(usize, Vec<usize>),
    /// IR representation for a Phase T-dagger gate on qubit index with controls
    Tdg(usize, Vec<usize>),
    /// IR representation for a Phase Shift (P) gate with an angle, on qubit index with controls
    Phase(f64, usize, Vec<usize>),
    /// IR representation for a RotateX gate with an angle, on qubit index with controls
    Rx(f64, usize, Vec<usize>),
    /// IR representation for a RotateY gate with an angle, on qubit index with controls
    Ry(f64, usize, Vec<usize>),
    /// IR representation for a RotateZ gate with an angle, on qubit index with controls
    Rz(f64, usize, Vec<usize>),
    /// IR representation for an Identity gate on qubit index with controls
    Id(usize, Vec<usize>),
    /// IR representation for a SWAP gate on two target qubits, with controls
    Swap(usize, usize, Vec<usize>),
    /// IR representation for a custom 2x2 Unitary gate, on qubit index with controls
    Unitary([[Complex<f64>; 2]; 2], usize, Vec<usize>),

    // MEASUREMENT
    /// IR representation for a measurement operation on a vector of qubits with a given basis
    Measurement(Vec<usize>, MeasurementBasis),

    // Allow Raw QASM Strings (without validation) for custom instructions
    /// IR representation for a raw QASM string
    ///
    /// This allows for custom instructions that may not be directly supported by the existing IR format.
    ///
    /// WARNING: This does not validate the QASM string, so it should be used with caution.
    RawQASM(String),
}

impl InstructionIR {
    /// Private method to generates QASM string and comment string for control qubits.
    fn format_complex(c: Complex<f64>) -> String {
        const EPSILON: f64 = 1e-9;
        let r = c.re;
        let i = c.im;

        if r.abs() < EPSILON && i.abs() < EPSILON {
            "0".to_string()
        } else if i.abs() < EPSILON {
            format!("{:.3}", r)
        } else if r.abs() < EPSILON {
            format!("{:.3}i", i)
        } else {
            format!(
                "{:.3} {} {:.3}i",
                r,
                if i.is_sign_negative() { "-" } else { "+" },
                i.abs()
            )
        }
    }

    /// Private method to generates QASM string and comment string for control qubits.
    fn generate_control_qasm_strings(controls: &[usize]) -> (String, String) {
        if controls.is_empty() {
            (String::new(), String::new())
        } else {
            let mut ctrl_qasm_str = format!("ctrl({}) @ ", controls.len());
            for (i, c) in controls.iter().enumerate() {
                if i > 0 {
                    ctrl_qasm_str.push_str(", ");
                }
                ctrl_qasm_str.push_str(&format!("q[{}]", c));
            }

            let mut ctrl_comment_str = "with control qubits: ".to_string();
            for (i, c) in controls.iter().enumerate() {
                if i > 0 {
                    ctrl_comment_str.push_str(", ");
                }
                ctrl_comment_str.push_str(&c.to_string());
            }

            (ctrl_qasm_str, ctrl_comment_str)
        }
    }

    /// Converts the IR instruction to a QASM string representation.
    ///
    /// # Returns
    ///
    /// A `QasmInstruction` enum that represents the QASM equivalent of the IR instruction, or an error if the conversion fails.
    pub(crate) fn to_qasm(&self) -> Result<Vec<QasmInstruction>, CompilerError> {
        // This function should be implemented to convert the IR instruction to a QASM string.
        let mut qasm_instructions: Vec<QasmInstruction> = Vec::new();
        match self {
            // If it's a raw QASM string, return it as-is.
            InstructionIR::RawQASM(qasmstr) => {
                qasm_instructions.push(QasmInstruction::RawQASM(qasmstr.clone()));
            }

            // Handle gates
            InstructionIR::Hadamard(index, controls) => {
                let (ctrl_qasm_str, ctrl_comment_str) =
                    Self::generate_control_qasm_strings(controls);

                qasm_instructions.push(QasmInstruction::GateDeclaration(format!(
                    "{} h q[{}] // {} {} {}",
                    ctrl_qasm_str, index, "Hadamard gate on qubit", index, ctrl_comment_str
                )));
            }

            InstructionIR::PauliX(index, controls) => {
                let (ctrl_qasm_str, ctrl_comment_str) =
                    Self::generate_control_qasm_strings(controls);
                
                qasm_instructions.push(QasmInstruction::GateDeclaration(format!(
                    "{} x q[{}] // {} {} {}",
                    ctrl_qasm_str, index, "Pauli-X gate on qubit", index, ctrl_comment_str
                )));
            },

            InstructionIR::PauliY(index, controls) => {
                let (ctrl_qasm_str, ctrl_comment_str) =
                    Self::generate_control_qasm_strings(controls);
                
                qasm_instructions.push(QasmInstruction::GateDeclaration(format!(
                    "{} y q[{}] // {} {} {}",
                    ctrl_qasm_str, index, "Pauli-Y gate on qubit", index, ctrl_comment_str
                )));
            },

            InstructionIR::PauliZ(index, controls) => {
                let (ctrl_qasm_str, ctrl_comment_str) =
                    Self::generate_control_qasm_strings(controls);
                
                qasm_instructions.push(QasmInstruction::GateDeclaration(format!(
                    "{} z q[{}] // {} {} {}",
                    ctrl_qasm_str, index, "Pauli-Z gate on qubit", index, ctrl_comment_str
                )));
            },

            InstructionIR::S(index, controls) => {
                let (ctrl_qasm_str, ctrl_comment_str) =
                    Self::generate_control_qasm_strings(controls);
                
                qasm_instructions.push(QasmInstruction::GateDeclaration(format!(
                    "{} s q[{}] // {} {} {}",
                    ctrl_qasm_str, index, "Phase S gate on qubit", index, ctrl_comment_str
                )));
            },

            InstructionIR::T(index, controls) => {
                let (ctrl_qasm_str, ctrl_comment_str) =
                    Self::generate_control_qasm_strings(controls);
                
                qasm_instructions.push(QasmInstruction::GateDeclaration(format!(
                    "{} t q[{}] // {} {} {}",
                    ctrl_qasm_str, index, "Phase T gate on qubit", index, ctrl_comment_str
                )));
            },

            InstructionIR::Sdg(index, controls) => {
                let (ctrl_qasm_str, ctrl_comment_str) =
                    Self::generate_control_qasm_strings(controls);
                
                qasm_instructions.push(QasmInstruction::GateDeclaration(format!(
                    "{} sdg q[{}] // {} {} {}",
                    ctrl_qasm_str, index, "Phase S-dagger gate on qubit", index, ctrl_comment_str
                )));
            },

            InstructionIR::Tdg(index, controls) => {
                let (ctrl_qasm_str, ctrl_comment_str) =
                    Self::generate_control_qasm_strings(controls);
                
                qasm_instructions.push(QasmInstruction::GateDeclaration(format!(
                    "{} tdg q[{}] // {} {} {}",
                    ctrl_qasm_str, index, "Phase T-dagger gate on qubit", index, ctrl_comment_str
                )));
            },

            InstructionIR::Phase(angle, index, controls) => {
                let (ctrl_qasm_str, ctrl_comment_str) =
                    Self::generate_control_qasm_strings(controls);
                
                
                qasm_instructions.push(QasmInstruction::GateDeclaration(format!(
                    "{} p({}) q[{}] // {} {} {} {} {}",
                    ctrl_qasm_str, angle, index,
                    "Phase gate with angle", angle, "on qubit", index, ctrl_comment_str
                )));
            },

            InstructionIR::Rx(angle, index, controls) => {
                let (ctrl_qasm_str, ctrl_comment_str) =
                    Self::generate_control_qasm_strings(controls);
                
                qasm_instructions.push(QasmInstruction::GateDeclaration(format!(
                    "{} rx({}) q[{}] // {} {} {} {} {}",
                    ctrl_qasm_str, angle, index,
                    "Rotate-X gate with angle", angle, "on qubit", index, ctrl_comment_str
                )));
            },

            InstructionIR::Ry(angle, index, controls) => {
                let (ctrl_qasm_str, ctrl_comment_str) =
                    Self::generate_control_qasm_strings(controls);
                
                qasm_instructions.push(QasmInstruction::GateDeclaration(format!(
                    "{} ry({}) q[{}] // {} {} {} {} {}",
                    ctrl_qasm_str, angle, index,
                    "Rotate-Y gate with angle", angle, "on qubit", index, ctrl_comment_str
                )));
            },

            InstructionIR::Rz(angle, index, controls) => {
                let (ctrl_qasm_str, ctrl_comment_str) =
                    Self::generate_control_qasm_strings(controls);
                
                qasm_instructions.push(QasmInstruction::GateDeclaration(format!(
                    "{} rz({}) q[{}] // {} {} {} {} {}",
                    ctrl_qasm_str, angle, index,
                    "Rotate-Z gate with angle", angle, "on qubit", index, ctrl_comment_str
                )));
            },

            InstructionIR::Id(index, controls) => {
                let (ctrl_qasm_str, ctrl_comment_str) =
                    Self::generate_control_qasm_strings(controls);
                
                qasm_instructions.push(QasmInstruction::GateDeclaration(format!(
                    "{} id q[{}] // {} {} {}",
                    ctrl_qasm_str, index, "Identity gate on qubit", index, ctrl_comment_str
                )));
            },

            InstructionIR::Swap(index1, index2, controls) => {
                let (ctrl_qasm_str, ctrl_comment_str) =
                    Self::generate_control_qasm_strings(controls);
                
                qasm_instructions.push(QasmInstruction::GateDeclaration(format!(
                    "{} swap q[{}], q[{}] // {} {} {} {} {}",
                    ctrl_qasm_str, index1, index2,
                    "SWAP gate between qubits", index1, "and", index2, ctrl_comment_str
                )));
            },

            InstructionIR::Unitary(matrix_val, target_idx, controls_vec) => {
                let (ctrl_qasm_str, ctrl_comment_str) =
                    Self::generate_control_qasm_strings(controls_vec);

                let a: Complex<f64> = matrix_val[0][0];
                let b: Complex<f64> = matrix_val[0][1];
                let c: Complex<f64> = matrix_val[1][0];
                let d: Complex<f64> = matrix_val[1][1];

                let theta: f64;
                let phi: f64;
                let lambda: f64;
                let alpha: f64;

                const EPSILON: f64 = 1e-9; // For norm comparisons

                if (1.0 - a.norm()).abs() < EPSILON { // Case 1: |a| ~ 1 (matrix is almost diagonal)
                    theta = 0.0;
                    phi = 0.0; // Conventional choice
                    lambda = d.arg() - a.arg();
                } else if (1.0 - c.norm()).abs() < EPSILON { // Case 2: |c| ~ 1 (matrix is almost anti-diagonal)
                    theta = std::f64::consts::PI;
                    alpha = std::f64::consts::FRAC_PI_2;
                    phi = c.arg() - alpha;
                    lambda = b.arg() - alpha + std::f64::consts::PI;
                } else { // General Case
                    // Check for invalid unitary (first column is zero vector)
                    if a.norm() < EPSILON && c.norm() < EPSILON {
                        return Err(CompilerError::UnsupportedOperator(
                            "Custom Unitary with zero first column".to_string(),
                        ));
                    }
                    theta = 2.0 * c.norm().atan2(a.norm());
                    alpha = a.arg();
                    phi = c.arg() - alpha; // Equivalent to c.arg() - a.arg()
                    lambda = b.arg() - alpha + std::f64::consts::PI; // Equivalent to b.arg() - a.arg() + pi
                }
                
                let base_comment = format!(
                    "Custom Unitary U({}, {}, {}, {}) on qubit {}",
                    Self::format_complex(a),
                    Self::format_complex(b),
                    Self::format_complex(c),
                    Self::format_complex(d),
                    target_idx
                );
                let full_comment = if controls_vec.is_empty() {
                    base_comment
                } else {
                    format!("{} {}", base_comment, ctrl_comment_str)
                };

                qasm_instructions.push(QasmInstruction::GateDeclaration(format!(
                    "{} U({:.3},{:.3},{:.3}) q[{}] // {}",
                    ctrl_qasm_str, theta, phi, lambda, target_idx, full_comment
                )));
            }

            // Handle measurements
            InstructionIR::Measurement(target_qubit_indices, basis_enum) => {
                // Indicate that a classical register is needed for this measurement's result.
                qasm_instructions.push(QasmInstruction::BitRegisterDeclaration(
                    target_qubit_indices.len(),
                ));
                qasm_instructions.push(QasmInstruction::StartMeasurementGroup);

                // Generate QASM instructions for measurement in the specified basis.
                for &target_qubit_idx in target_qubit_indices {
                    match basis_enum {
                        MeasurementBasis::Computational => {
                            let core_qasm_command = format!("measure q[{}]", target_qubit_idx);
                            qasm_instructions
                                .push(QasmInstruction::MeasurementDeclaration(core_qasm_command));
                        }
                        MeasurementBasis::X => {
                            let core_qasm_command = format!("xmeasure q[{}]", target_qubit_idx);
                            qasm_instructions
                                .push(QasmInstruction::MeasurementDeclaration(core_qasm_command));
                        }
                        MeasurementBasis::Y => {
                            let core_qasm_command = format!("ymeasure q[{}]", target_qubit_idx);
                            qasm_instructions
                                .push(QasmInstruction::MeasurementDeclaration(core_qasm_command));
                        }
                        MeasurementBasis::Custom(custom_matrix) => {
                            // For custom basis, apply U_inverse, then measure in Z basis.
                            let u_inv = [
                                [custom_matrix[0][0].conj(), custom_matrix[1][0].conj()],
                                [custom_matrix[0][1].conj(), custom_matrix[1][1].conj()],
                            ];

                            // Create uncontrolled Unitary IR instruction for inverse operation.
                            let unitary_op_ir =
                                InstructionIR::Unitary(u_inv, target_qubit_idx, vec![]);
                            qasm_instructions.extend(unitary_op_ir.to_qasm()?);

                            // Measure in the Z-basis.
                            let core_qasm_command = format!("measure q[{}]", target_qubit_idx);
                            qasm_instructions
                                .push(QasmInstruction::MeasurementDeclaration(core_qasm_command));
                        }
                    }
                }
            }
        }
        Ok(qasm_instructions)
    }
}