__kernel void pauli_y_kernel(
    __global float2 *state_vector,
    const int num_qubits,
    const int target_qubit,
    __global const int *control_qubits,
    const int num_control_qubits) {

    int global_id = get_global_id(0);
    int i0 = (global_id >> target_qubit << (target_qubit + 1)) | (global_id & ((1 << target_qubit) - 1));
    int i1 = i0 | (1 << target_qubit);

    bool controls_active = true;
    if (num_control_qubits > 0) {
        for (int k = 0; k < num_control_qubits; ++k) {
            if (!((i0 >> control_qubits[k]) & 1)) {
                controls_active = false;
                break;
            }
        }
    }

    if (controls_active) {
        float2 amp0 = state_vector[i0];
        float2 amp1 = state_vector[i1];
        // Pauli Y: |0> -> i|1>, |1> -> -i|0>
        // amp0' = -i * amp1
        // amp1' =  i * amp0
        state_vector[i0] = (float2)(amp1.y, -amp1.x); // -i * (re1 + i*im1) = im1 - i*re1
        state_vector[i1] = (float2)(-amp0.y, amp0.x); //  i * (re0 + i*im0) = -im0 + i*re0
    }
}
