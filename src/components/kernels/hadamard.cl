__kernel void hadamard_kernel(
    __global float2 *state_vector,      // The quantum state vector (array of complex numbers)
    const int num_qubits,           // Total number of qubits in the state
    const int target_qubit,         // The index of the qubit to apply Hadamard to
    __global const int *control_qubits, // Array of control qubit indices (can be empty)
    const int num_control_qubits    // Number of control qubits
) {
    const float sqrt_2_inv = 0.70710678118f; // 1.0f / sqrt(2.0f)

    // Each work-item is responsible for one pair of amplitudes (alpha, beta)
    // that differ only at the target_qubit.
    // get_global_id(0) gives k, which ranges from 0 to (2^(num_qubits-1) - 1).
    int k = get_global_id(0);

    // Calculate the indices i0 and i1 in the full state vector.
    // i0: basis state where target_qubit is 0.
    // i1: basis state where target_qubit is 1.
    // This bit manipulation reconstructs the full indices from k by inserting
    // a 0 (for i0) or 1 (for i1) at the target_qubit position.
    int i0 = (k >> target_qubit << (target_qubit + 1)) | (k & ((1 << target_qubit) - 1));
    int i1 = i0 | (1 << target_qubit);

    // Handle control qubits if any are specified.
    if (num_control_qubits > 0) {
        bool all_controls_are_one = true;
        for (int c_idx = 0; c_idx < num_control_qubits; ++c_idx) {
            int control_q = control_qubits[c_idx];
            // Check if the control_q bit is 0 in the i0 basis state.
            // (It will be the same for i1, as they only differ at target_qubit).
            if (!((i0 >> control_q) & 1)) {
                all_controls_are_one = false;
                break;
            }
        }
        // If controls are not met, this work-item does nothing.
        if (!all_controls_are_one) {
            return;
        }
    }

    // Load the amplitudes for the pair.
    float2 amp0 = state_vector[i0]; // Amplitude for |...0...>
    float2 amp1 = state_vector[i1]; // Amplitude for |...1...>

    // Apply the Hadamard transformation.
    // new_amp0 = (amp0 + amp1) / sqrt(2)
    state_vector[i0].x = sqrt_2_inv * (amp0.x + amp1.x);
    state_vector[i0].y = sqrt_2_inv * (amp0.y + amp1.y);

    // new_amp1 = (amp0 - amp1) / sqrt(2)
    state_vector[i1].x = sqrt_2_inv * (amp0.x - amp1.x);
    state_vector[i1].y = sqrt_2_inv * (amp0.y - amp1.y);
}
