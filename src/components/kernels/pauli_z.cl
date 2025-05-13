__kernel void pauli_z_kernel(
    __global float2 *state_vector,
    const int num_qubits,
    const int target_qubit,
    __global const int *control_qubits,
    const int num_control_qubits) {

    int global_id = get_global_id(0); // global_id iterates from 0 to (2^num_qubits - 1)

    // Check if the target qubit is 1 for this basis state
    bool target_is_one = (global_id >> target_qubit) & 1;

    if (target_is_one) {
        // Check control qubits
        bool controls_active = true;
        if (num_control_qubits > 0) {
            for (int k = 0; k < num_control_qubits; ++k) {
                if (!((global_id >> control_qubits[k]) & 1)) {
                    controls_active = false;
                    break;
                }
            }
        }

        if (controls_active) {
            // Apply phase -1
            state_vector[global_id].x = -state_vector[global_id].x;
            state_vector[global_id].y = -state_vector[global_id].y;
        }
    }
}
