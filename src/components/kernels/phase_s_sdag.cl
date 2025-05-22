__kernel void phase_s_sdag_kernel(
    __global float2 *state_vector,
    const int num_qubits,
    const int target_qubit,
    __global const int *control_qubits,
    const int num_control_qubits,
    const float sign // 1.0f for S, -1.0f for Sdag
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
        // Apply phase:
        // For S (sign = 1.0): (a+bi) * i = -b + ai
        // For Sdag (sign = -1.0): (a+bi) * (-i) = b - ai
        // General form: (a+bi) * (sign * i) = (sign * -b) + (sign * a)i
        float original_real = amp.x;
        float original_imag = amp.y;
        amp.x = -sign * original_imag;
        amp.y =  sign * original_real;
        state_vector[global_id] = amp;
    }
}