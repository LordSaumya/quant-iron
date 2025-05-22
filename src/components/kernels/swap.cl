// OpenCL Kernel for SWAP gate
// Swaps the state of two qubits q_idx1 and q_idx2.
// state:      Pointer to the complex state vector.
// num_qubits: Total number of qubits in the system.
// q_idx1:     Index of the first qubit.
// q_idx2:     Index of the second qubit.

__kernel void swap_kernel(
    __global float2* state,
    const uint num_qubits,
    const uint q_idx1,                 // First target qubit for SWAP
    __global const int* control_qubits, // Array of control qubit indices
    const int num_control_qubits,    // Number of control qubits
    const uint q_idx2)                 // Second target qubit for SWAP
{
    // Each work-item is responsible for one pair of amplitudes (i, j)
    // that differ at q_idx1 and q_idx2.
    // get_global_id(0) gives k, which ranges from 0 to (2^(num_qubits-2) - 1)
    // if num_qubits >= 2. If num_qubits < 2, this kernel should not be called
    // or q_idx1/q_idx2 validation on host should prevent issues.
    uint k_gid = get_global_id(0);

    // Determine the lower and higher qubit indices for consistent bit manipulation
    uint q_low = min(q_idx1, q_idx2);
    uint q_high = max(q_idx1, q_idx2);

    // If q_idx1 and q_idx2 are the same, no operation is needed.
    // This check should ideally be done on the host side.
    if (q_low == q_high) {
        return;
    }

    // Reconstruct the first basis state index 'i'.
    // 'k_gid' provides the bits for qubits other than q_low and q_high.
    // For index 'i': bit at q_high is 0, bit at q_low is 1.
    // For index 'j': bit at q_high is 1, bit at q_low is 0.
    // (This convention is arbitrary but must be consistent for i and j construction)

    uint i = 0;
    uint current_gid_bit = 0; // Tracks which bit of k_gid we are using

    // Iterate through all qubit positions to construct 'i'
    for (uint bit_pos = 0; bit_pos < num_qubits; ++bit_pos) {
        if (bit_pos == q_low) {
            i |= (1u << q_low); // Set q_low bit to 1 for state 'i'
        } else if (bit_pos == q_high) {
            // Bit at q_high is 0 for state 'i' (implicitly, as i starts at 0)
        } else {
            // For other qubits, take the bit from k_gid
            if ((k_gid >> current_gid_bit) & 1u) {
                i |= (1u << bit_pos);
            }
            current_gid_bit++;
        }
    }

    // Construct the second basis state index 'j' by flipping bits at q_low and q_high in 'i'.
    // j = i XOR ( (1 << q_low) | (1 << q_high) )
    uint j = i ^ ((1u << q_low) | (1u << q_high));


    // Handle control qubits if any are specified.
    if (num_control_qubits > 0) {
        bool all_controls_are_one = true;
        for (int c_idx = 0; c_idx < num_control_qubits; ++c_idx) {
            int control_q = control_qubits[c_idx];
            // Check if the control_q bit is 0 in the 'i' basis state.
            // (It will be the same for 'j' if control_q is not q_low or q_high,
            // which should be ensured by host-side validation).
            if (!((i >> control_q) & 1u)) {
                all_controls_are_one = false;
                break;
            }
        }
        // If controls are not met, this work-item does nothing.
        if (!all_controls_are_one) {
            return;
        }
    }

    // Perform the swap of amplitudes for state[i] and state[j]
    float2 temp = state[i];
    state[i] = state[j];
    state[j] = temp;
}