// Helper for complex multiplication: a * b
inline float2 cmul(float2 a, float2 b) {
    return (float2)(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// Helper for complex addition: a + b
inline float2 cadd(float2 a, float2 b) {
    return (float2)(a.x + b.x, a.y + b.y);
}

// Helper for complex subtraction: a - b
inline float2 csub(float2 a, float2 b) {
    return (float2)(a.x - b.x, a.y - b.y);
}

// Helper for scalar-complex multiplication: s * c
inline float2 scmul(float s, float2 c) {
    return (float2)(s * c.x, s * c.y);
}

__kernel void match_gate_kernel(
    __global float2* state,
    const uint num_qubits,
    const uint q1,
    __global const int* control_qubits,
    const int num_control_qubits,
    const uint q2,
    const float cos_theta_half,
    const float sin_theta_half,
    const float2 exp_i_phi1,
    const float2 exp_i_phi2)
{
    // Each work-item handles a unique combination of the other n-2 qubits
    uint k = get_global_id(0);

    uint q_low = min(q1, q2);
    uint q_high = max(q1, q2);

    // Reconstruct the base index for the |..00..> state
    uint base_idx = 0;
    uint k_mask = 0;
    for (uint i = 0; i < num_qubits; ++i) {
        if (i == q_low || i == q_high) continue;
        if ((k >> k_mask) & 1) {
            base_idx |= (1 << i);
        }
        k_mask++;
    }

    // Determine the four indices in the subspace
    uint i00 = base_idx;
    uint i01 = base_idx | (1 << q_low);
    uint i10 = base_idx | (1 << q_high);
    uint i11 = base_idx | (1 << q_low) | (1 << q_high);

    // Check control qubits. The check is against the base index.
    bool controls_met = true;
    if (num_control_qubits > 0) {
        for (int i = 0; i < num_control_qubits; ++i) {
            if (!((base_idx >> control_qubits[i]) & 1)) {
                controls_met = false;
                break;
            }
        }
    }

    if (controls_met) {
        // Load amplitudes
        float2 amp01 = state[i01];
        float2 amp10 = state[i10];
        float2 amp11 = state[i11];

        // Apply Matchgate logic
        // new_amp01 = cos_theta_half * amp01 - exp_i_phi1 * sin_theta_half * amp10;
        float2 term1 = scmul(cos_theta_half, amp01);
        float2 term2 = cmul(exp_i_phi1, scmul(sin_theta_half, amp10));
        float2 new_amp01 = csub(term1, term2);

        // new_amp10 = sin_theta_half * amp01 + exp_i_phi2 * cos_theta_half * amp10;
        float2 term3 = scmul(sin_theta_half, amp01);
        float2 term4 = cmul(exp_i_phi2, scmul(cos_theta_half, amp10));
        float2 new_amp10 = cadd(term3, term4);
        
        // new_amp11 = amp11 * exp_i_phi2
        float2 new_amp11 = cmul(amp11, exp_i_phi2);

        // Write back the updated amplitudes
        state[i01] = new_amp01;
        state[i10] = new_amp10;
        state[i11] = new_amp11;
        // amp00 is unchanged
    }
}