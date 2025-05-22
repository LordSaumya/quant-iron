__kernel void rotate_x_kernel(
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
    // i0 has 0 at target_qubit, i1 has 1 at target_qubit
    int i0 = (k >> target_qubit << (target_qubit + 1)) | (k & ((1 << target_qubit) - 1));
    int i1 = i0 | (1 << target_qubit);

    // Check control qubits based on i0 (or i1, since control bits are the same for the pair)
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

        // RX transformation:
        // amp0' = cos_half_angle * amp0 - I * sin_half_angle * amp1
        // amp1' = -I * sin_half_angle * amp0 + cos_half_angle * amp1
        // where I is (0, 1)
        // amp0'.x = cos_half_angle * amp0.x - (0*sin_half_angle*amp1.x - 1*sin_half_angle*amp1.y)
        //         = cos_half_angle * amp0.x + sin_half_angle * amp1.y
        // amp0'.y = cos_half_angle * amp0.y - (0*sin_half_angle*amp1.y + 1*sin_half_angle*amp1.x)
        //         = cos_half_angle * amp0.y - sin_half_angle * amp1.x

        // amp1'.x = -(0*sin_half_angle*amp0.x - 1*sin_half_angle*amp0.y) + cos_half_angle * amp1.x
        //         = sin_half_angle * amp0.y + cos_half_angle * amp1.x
        // amp1'.y = -(0*sin_half_angle*amp0.y + 1*sin_half_angle*amp0.x) + cos_half_angle * amp1.y
        //         = -sin_half_angle * amp0.x + cos_half_angle * amp1.y

        float new_amp0_re = cos_half_angle * amp0.x + sin_half_angle * amp1.y;
        float new_amp0_im = cos_half_angle * amp0.y - sin_half_angle * amp1.x;

        float new_amp1_re = sin_half_angle * amp0.y + cos_half_angle * amp1.x;
        float new_amp1_im = -sin_half_angle * amp0.x + cos_half_angle * amp1.y;

        state_vector[i0] = (float2)(new_amp0_re, new_amp0_im);
        state_vector[i1] = (float2)(new_amp1_re, new_amp1_im);
    }
}