__kernel void pauli_x_kernel(
    __global float2 *state_vector,
    const int num_qubits,
    const int target_qubit,
    __global const int *control_qubits,
    const int num_control_qubits) {

    int global_id = get_global_id(0);
    // Calculate i0 and i1, the indices of the two basis states that differ only at the target_qubit
    // global_id iterates from 0 to (2^(num_qubits-1) - 1)
    // i0 has 0 at target_qubit, i1 has 1 at target_qubit
    int i0 = (global_id >> target_qubit << (target_qubit + 1)) | (global_id & ((1 << target_qubit) - 1));
    int i1 = i0 | (1 << target_qubit);

    // Check control qubits
    bool controls_active = true;
    if (num_control_qubits > 0) {
        for (int k = 0; k < num_control_qubits; ++k) {
            if (!((i0 >> control_qubits[k]) & 1)) { // Check based on i0, as controls apply to the pair
                controls_active = false;
                break;
            }
        }
    }

    if (controls_active) {
        float2 amp0 = state_vector[i0];
        float2 amp1 = state_vector[i1];
        state_vector[i0] = amp1;
        state_vector[i1] = amp0;
    }
}
