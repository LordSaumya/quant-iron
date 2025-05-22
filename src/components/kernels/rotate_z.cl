__kernel void rotate_z_kernel(
    __global float2 *state_vector,
    const int num_qubits,
    const int target_qubit,
    __global const int *control_qubits,
    const int num_control_qubits,
    const float cos_half_angle,
    const float sin_half_angle
) {
    int global_id = get_global_id(0); // Corresponds to the basis state index i

    // Check control qubits
    bool controls_met = true;
    if (num_control_qubits > 0) {
        for (int i = 0; i < num_control_qubits; ++i) {
            if (!((global_id >> control_qubits[i]) & 1)) {
                controls_met = false;
                break;
            }
        }
    }

    if (controls_met) {
        float2 amp = state_vector[global_id];
        float phase_re;
        float phase_im;

        // Determine phase based on target qubit state
        if (!((global_id >> target_qubit) & 1)) { // Target qubit is |0>
            phase_re = cos_half_angle;
            phase_im = -sin_half_angle; // exp(-i * angle / 2)
        } else { // Target qubit is |1>
            phase_re = cos_half_angle;
            phase_im = sin_half_angle;  // exp(i * angle / 2)
        }

        // Apply phase: (a+bi) * (phase_re + i*phase_im)
        // = (a*phase_re - b*phase_im) + i*(a*phase_im + b*phase_re)
        float original_real = amp.x;
        float original_imag = amp.y;
        amp.x = original_real * phase_re - original_imag * phase_im;
        amp.y = original_real * phase_im + original_imag * phase_re;
        state_vector[global_id] = amp;
    }
}