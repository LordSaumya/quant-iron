__kernel void phase_shift_kernel(
    __global float2 *state_vector,
    const int num_qubits,
    const int target_qubit,
    __global const int *control_qubits,
    const int num_control_qubits,
    const float cos_angle,
    const float sin_angle
) {
    int global_id = get_global_id(0); // Corresponds to the basis state index i

    // Check if the target qubit is |1> for this basis state
    if (!((global_id >> target_qubit) & 1)) {
        return; // Apply phase only if target qubit is 1
    }

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
        // Apply phase: (a+bi) * (cos_angle + i*sin_angle)
        // = (a*cos_angle - b*sin_angle) + i*(a*sin_angle + b*cos_angle)
        float original_real = amp.x;
        float original_imag = amp.y;
        amp.x = original_real * cos_angle - original_imag * sin_angle;
        amp.y = original_real * sin_angle + original_imag * cos_angle;
        state_vector[global_id] = amp;
    }
}