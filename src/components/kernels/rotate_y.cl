__kernel void rotate_y_kernel(
    __global float2 *state_vector,
    const int num_qubits,
    const int target_qubit,
    __global const int *control_qubits,
    const int num_control_qubits,
    const float cos_half_angle,
    const float sin_half_angle
) {
    int k = get_global_id(0); // k from 0 to 2^(num_qubits-1) - 1

    // Reconstruct i0 and i1 from k
    int i0 = (k >> target_qubit << (target_qubit + 1)) | (k & ((1 << target_qubit) - 1));
    int i1 = i0 | (1 << target_qubit);

    // Check control qubits
    bool controls_met = true;
    if (num_control_qubits > 0) {
        for (int c_idx = 0; c_idx < num_control_qubits; ++c_idx) {
            if (!((i0 >> control_qubits[c_idx]) & 1)) {
                controls_met = false;
                break;
            }
        }
    }

    if (controls_met) {
        float2 amp0 = state_vector[i0];
        float2 amp1 = state_vector[i1];

        // RY transformation:
        // amp0' = cos_half_angle * amp0 - sin_half_angle * amp1
        // amp1' = sin_half_angle * amp0 + cos_half_angle * amp1

        float new_amp0_re = cos_half_angle * amp0.x - sin_half_angle * amp1.x;
        float new_amp0_im = cos_half_angle * amp0.y - sin_half_angle * amp1.y;

        float new_amp1_re = sin_half_angle * amp0.x + cos_half_angle * amp1.x;
        float new_amp1_im = sin_half_angle * amp0.y + cos_half_angle * amp1.y;

        state_vector[i0] = (float2)(new_amp0_re, new_amp0_im);
        state_vector[i1] = (float2)(new_amp1_re, new_amp1_im);
    }
}