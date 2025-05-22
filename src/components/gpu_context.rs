#![cfg(feature = "gpu")]

use crate::errors::Error;
use ocl::{ProQue, Buffer, flags, prm::Float2};
use once_cell::sync::Lazy;
use std::sync::Mutex;


/// Defines the available GPU kernels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum KernelType {
    Hadamard,
    PauliX,
    PauliY,
    PauliZ,
    PhaseSOrSdag,
    PhaseShift,
    Swap,
    RotateX,
    RotateY,
    RotateZ,
}

impl std::fmt::Display for KernelType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KernelType::Hadamard => write!(f, "Hadamard"),
            KernelType::PauliX => write!(f, "PauliX"),
            KernelType::PauliY => write!(f, "PauliY"),
            KernelType::PauliZ => write!(f, "PauliZ"),
            KernelType::PhaseSOrSdag => write!(f, "PhaseSOrSdag"),
            KernelType::PhaseShift => write!(f, "PhaseShift"),
            KernelType::Swap => write!(f, "Swap"),
            KernelType::RotateX => write!(f, "RotateX"),
            KernelType::RotateY => write!(f, "RotateY"),
            KernelType::RotateZ => write!(f, "RotateZ"),
        }
    }
}

impl KernelType {
    // Kernel sources - assumes .cl files are in src/components/kernels/
    pub(crate) fn src(&self) -> &'static str {
        match self {
            KernelType::Hadamard => include_str!("kernels/hadamard.cl"),
            KernelType::PauliX => include_str!("kernels/pauli_x.cl"),
            KernelType::PauliY => include_str!("kernels/pauli_y.cl"),
            KernelType::PauliZ => include_str!("kernels/pauli_z.cl"),
            KernelType::PhaseSOrSdag => include_str!("kernels/phase_s_sdag.cl"),
            KernelType::PhaseShift => include_str!("kernels/phase_shift.cl"),
            KernelType::Swap => include_str!("kernels/swap.cl"),
            KernelType::RotateX => include_str!("kernels/rotate_x.cl"),
            KernelType::RotateY => include_str!("kernels/rotate_y.cl"),
            KernelType::RotateZ => include_str!("kernels/rotate_z.cl"),
        }
    }

    pub(crate) fn name(&self) -> &'static str {
        match self {
            KernelType::Hadamard => "hadamard_kernel",
            KernelType::PauliX => "pauli_x_kernel",
            KernelType::PauliY => "pauli_y_kernel",
            KernelType::PauliZ => "pauli_z_kernel",
            KernelType::PhaseSOrSdag => "phase_s_sdag_kernel",
            KernelType::PhaseShift => "phase_shift_kernel",
            KernelType::Swap => "swap_kernel",
            KernelType::RotateX => "rotate_x_kernel",
            KernelType::RotateY => "rotate_y_kernel",
            KernelType::RotateZ => "rotate_z_kernel",
        }
    }
}

/// Represents the arguments for GPU kernels.
pub(crate) enum GpuKernelArgs {
    None,
    SOrSdag { sign: f32 },
    PhaseShift { cos_angle: f32, sin_angle: f32 },
    SwapTarget { q1: i32 }, // For SWAP gate, q0 is the standard target_qubit
    RotationGate { cos_half_angle: f32, sin_half_angle: f32 },
}

pub(crate) struct GpuContext {
    pub pro_que: ProQue,
    pub state_buffer: Option<Buffer<Float2>>,
    pub control_buffer: Option<Buffer<i32>>,
    pub current_num_qubits: usize, // To track buffer size for state_buffer
    pub current_control_len: usize, // To track buffer size for control_buffer
}

impl GpuContext {
    fn new() -> Result<Self, Error> {
        let all_kernels_src = format!(
            "{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}",
            KernelType::Hadamard.src(),
            KernelType::PauliX.src(),
            KernelType::PauliY.src(),
            KernelType::PauliZ.src(),
            KernelType::PhaseSOrSdag.src(),
            KernelType::PhaseShift.src(),
            KernelType::Swap.src(),
            KernelType::RotateX.src(),
            KernelType::RotateY.src(),
            KernelType::RotateZ.src()
        );

        let pro_que = ProQue::builder()
            .src(all_kernels_src)
            .build()
            .map_err(|e| Error::OpenCLError(format!("Failed to build ProQue: {}", e)))?;

        Ok(GpuContext {
            pro_que,
            state_buffer: None,
            control_buffer: None,
            current_num_qubits: 0,
            current_control_len: 0,
        })
    }

    // Ensure state buffer is allocated and has the correct size
    pub fn ensure_state_buffer(&mut self, num_elements: usize) -> Result<&mut Buffer<Float2>, Error> {
        let num_qubits_for_buffer = if num_elements > 0 { num_elements.trailing_zeros() as usize } else { 0 };
        if self.state_buffer.is_none() || self.current_num_qubits != num_qubits_for_buffer || self.state_buffer.as_ref().unwrap().len() != num_elements {
            let buffer = Buffer::builder()
                .queue(self.pro_que.queue().clone())
                .flags(flags::MEM_READ_WRITE) 
                .len(num_elements)
                .build()
                .map_err(|e| Error::OpenCLError(format!("Failed to create state buffer: {}", e)))?;
            self.state_buffer = Some(buffer);
            self.current_num_qubits = num_qubits_for_buffer;
        }
        Ok(self.state_buffer.as_mut().unwrap())
    }

    // Ensure control buffer is allocated and has the correct size
    pub fn ensure_control_buffer(&mut self, num_elements: usize) -> Result<&mut Buffer<i32>, Error> {
        let actual_num_elements = if num_elements == 0 { 1 } else { num_elements }; // Min size 1 for dummy
        if self.control_buffer.is_none() || self.current_control_len != actual_num_elements || self.control_buffer.as_ref().unwrap().len() != actual_num_elements {
            let buffer = Buffer::builder()
                .queue(self.pro_que.queue().clone())
                .flags(flags::MEM_READ_ONLY)
                .len(actual_num_elements)
                .build()
                .map_err(|e| Error::OpenCLError(format!("Failed to create control buffer: {}", e)))?;
            self.control_buffer = Some(buffer);
            self.current_control_len = actual_num_elements;
        }
        Ok(self.control_buffer.as_mut().unwrap())
    }
}

pub(crate) static GPU_CONTEXT: Lazy<Mutex<Result<GpuContext, Error>>> = Lazy::new(|| {
    Mutex::new(GpuContext::new())
});
