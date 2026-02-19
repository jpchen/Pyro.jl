# Copyright (c) 2024
# SPDX-License-Identifier: Apache-2.0

# =============================================================================
# NUTS (No-U-Turn Sampler) Kernel
#
# Implements the NUTS algorithm from Hoffman & Gelman (2014) with the
# multinomial sampling extension from Betancourt (2017). NUTS extends HMC
# by automatically tuning the trajectory length using a recursive doubling
# procedure that builds a binary tree of leapfrog states. The tree stops
# growing when a U-turn is detected (the trajectory starts doubling back).
#
# Key features:
# - **Adaptive trajectory length**: No need to manually tune num_steps
# - **Multinomial sampling**: Selects from the trajectory proportional to
#   the unnormalized probability (more efficient than slice sampling)
# - **Generalized U-turn criterion**: Uses summed momentum for robust
#   termination (Betancourt 2017, Section A.4.2)
# - **Divergence detection**: Flags transitions with large energy errors
#
# The tree building procedure works by:
# 1. Start with the current state
# 2. Randomly choose to extend the tree left or right
# 3. Double the tree depth by building a new subtree
# 4. Check for U-turns at each level
# 5. Stop when a U-turn is detected or max depth is reached
#
# Reference: Pyro's pyro/infer/mcmc/nuts.py
#            Hoffman & Gelman (2014) "The No-U-Turn Sampler"
#            Betancourt (2017) "A Conceptual Introduction to HMC"
# =============================================================================

export NUTSKernel, setup!, sample_kernel, end_warmup!, diagnostics

"""
    _TreeInfo

Internal data structure tracking the state of the NUTS binary tree during
the tree-building recursion. Each node in the tree maintains:

- Left/right boundary states (positions, momenta, gradients, energies)
- A proposal state selected from the subtree
- The summed momentum across all leaves (for U-turn check)
- Aggregated weight (log-sum-exp of negative energies for multinomial)
- Diagnostic counters (number of proposals, turning/diverging flags)
"""
struct _TreeInfo
    z_left::Vector{Float64}
    r_left::Vector{Float64}
    z_left_grads::Vector{Float64}
    z_left_pe::Float64
    z_right::Vector{Float64}
    r_right::Vector{Float64}
    z_right_grads::Vector{Float64}
    z_right_pe::Float64
    z_proposal::Vector{Float64}
    z_proposal_pe::Float64
    z_proposal_grads::Vector{Float64}
    r_sum::Vector{Float64}
    weight::Float64             # log(sum(exp(-energy))) for multinomial sampling
    sum_accept_probs::Float64   # sum of min(1, exp(H₀ - H_leaf)) across leaves
    num_proposals::Int
    turning::Bool
    diverging::Bool
end

"""
    NUTSKernel

No-U-Turn Sampler kernel with multinomial sampling.

Automatically adapts trajectory length by building a balanced binary tree
of leapfrog states and stopping when a U-turn is detected.

# Constructor Arguments
- `potential_fn`: Function mapping parameter vector to scalar potential energy
- `step_size::Float64`: Initial leapfrog step size (default: 1.0)
- `adapt_step_size::Bool`: Adapt step size during warmup (default: true)
- `adapt_mass_matrix::Bool`: Adapt mass matrix during warmup (default: true)
- `target_accept_prob::Float64`: Target acceptance probability (default: 0.8)
- `max_tree_depth::Int`: Maximum binary tree depth (default: 10, giving up to 2^10=1024 leapfrog steps)

# Diagnostics
After sampling, `diagnostics(kernel)` returns:
- `step_size`: Final step size
- `accept_prob`: Mean acceptance probability
- `max_tree_depth`: Configured max tree depth
- `mean_tree_depth`: Average tree depth during sampling
- `divergences`: Number of divergent transitions
"""
mutable struct NUTSKernel
    potential_fn::Function
    step_size::Float64
    adapt_step_size::Bool
    adapt_mass_matrix::Bool
    target_accept_prob::Float64
    max_tree_depth::Int
    adapter::Union{WarmupAdapter, Nothing}
    inv_mass_matrix::Vector{Float64}
    current_z::Union{Vector{Float64}, Nothing}
    current_pe::Float64
    current_grads::Union{Vector{Float64}, Nothing}
    warmup_steps::Int
    is_warmup::Bool
    # Diagnostics
    sum_accept_probs::Float64
    num_proposals_total::Int
    sample_count::Int
    divergence_count::Int
    total_tree_depth::Int

    function NUTSKernel(
        potential_fn::Function;
        step_size::Float64=1.0,
        adapt_step_size::Bool=true,
        adapt_mass_matrix::Bool=true,
        target_accept_prob::Float64=0.8,
        max_tree_depth::Int=10
    )
        new(potential_fn, step_size,
            adapt_step_size, adapt_mass_matrix, target_accept_prob,
            max_tree_depth,
            nothing, Float64[], nothing, 0.0, nothing, 0, true,
            0.0, 0, 0, 0, 0)
    end
end

"""
    setup!(kernel::NUTSKernel, warmup_steps::Int, initial_params::Vector{Float64})

Initialize the NUTS kernel before sampling begins.
"""
function setup!(kernel::NUTSKernel, warmup_steps::Int, initial_params::Vector{Float64})
    dim = length(initial_params)
    kernel.warmup_steps = warmup_steps
    kernel.is_warmup = true
    kernel.sum_accept_probs = 0.0
    kernel.num_proposals_total = 0
    kernel.sample_count = 0
    kernel.divergence_count = 0
    kernel.total_tree_depth = 0

    # Initialize position and compute initial potential energy + gradients
    kernel.current_z = copy(initial_params)
    kernel.current_grads, kernel.current_pe = potential_energy_grad(
        kernel.potential_fn, kernel.current_z
    )

    # Initialize inverse mass matrix to identity
    kernel.inv_mass_matrix = ones(dim)

    # Find reasonable initial step size
    if kernel.adapt_step_size
        kernel.step_size = _find_reasonable_step_size(
            kernel.potential_fn, kernel.current_z, kernel.current_grads,
            kernel.inv_mass_matrix
        )
    end

    # Set up adaptation
    kernel.adapter = WarmupAdapter(
        dim;
        adapt_step_size=kernel.adapt_step_size,
        adapt_mass_matrix=kernel.adapt_mass_matrix,
        target_accept_prob=kernel.target_accept_prob
    )
    configure_warmup!(kernel.adapter, warmup_steps, kernel.step_size)

    return nothing
end

"""
    _is_turning(r_left, r_right, r_sum, inv_mass_matrix) -> Bool

Check the generalized U-turn criterion (Betancourt 2017, Section A.4.2).

The criterion uses the summed momentum ρ across all leaves in the subtree:
    ρ_corrected = ρ - (r_left + r_right) / 2

Then checks:
    turning = (r_left · ρ_corrected ≤ 0) OR (r_right · ρ_corrected ≤ 0)

This is more robust than the original NUTS criterion which only checked
endpoints, as it accounts for the trajectory's overall direction.
"""
function _is_turning(
    r_left::Vector{Float64},
    r_right::Vector{Float64},
    r_sum::Vector{Float64},
    inv_mass_matrix::Vector{Float64}
)
    # Generalized U-turn criterion (Betancourt 2017, Section A.4.2).
    # r_sum is the sum of raw momenta across all leaves in the subtree.
    # The correction subtracts half the boundary momenta (all in raw space).
    rho = r_sum .- (r_left .+ r_right) ./ 2.0

    # The turning check uses M⁻¹ * r (velocity) dotted with rho.
    # For diagonal M: velocity = inv_mass_matrix .* r
    r_left_velocity = r_left .* inv_mass_matrix
    r_right_velocity = r_right .* inv_mass_matrix

    turning_left = dot(r_left_velocity, rho) <= 0
    turning_right = dot(r_right_velocity, rho) <= 0

    return turning_left || turning_right
end

"""
    _build_basetree(potential_fn, state, direction, step_size, inv_mass_matrix,
                    energy_current) -> _TreeInfo

Build a base tree (depth 0) by taking a single leapfrog step.

This is the leaf node of the NUTS binary tree. It takes one leapfrog step
in the given direction and evaluates whether the resulting state is valid.

# Arguments
- `direction`: +1 for forward, -1 for backward in time
- `energy_current`: Energy at the start of the overall trajectory (for weight computation)
"""
function _build_basetree(
    potential_fn::Function,
    z::Vector{Float64},
    r::Vector{Float64},
    z_grads::Vector{Float64},
    z_pe::Float64,
    direction::Int,
    step_size::Float64,
    inv_mass_matrix::Vector{Float64},
    energy_current::Float64
)
    # Take one leapfrog step in the given direction
    state = IntegratorState(z, r, z_pe, z_grads)
    new_state = velocity_verlet(
        potential_fn, state, direction * step_size, 1, inv_mass_matrix
    )

    z_new = new_state.z
    r_new = new_state.r
    pe_new = new_state.potential_energy
    grads_new = new_state.z_grads

    # Compute energy at the new state
    ke_new = _kinetic_energy(r_new, inv_mass_matrix)
    energy_new = pe_new + ke_new
    if isnan(energy_new)
        energy_new = Inf
    end

    # Divergence check: flag if energy error exceeds threshold.
    # 1000 nats is the standard threshold (following Stan/Pyro).
    delta_energy = energy_new - energy_current
    diverging = delta_energy > 1000.0

    # Multinomial weight: log probability proportional to exp(-energy).
    # The weight is the negative energy (log space), which will be
    # combined via log-sum-exp across the tree.
    weight = -energy_new

    # Per-leaf acceptance probability for step size adaptation.
    # This is min(1, exp(H_current - H_leaf)) = min(1, exp(-delta_energy)).
    leaf_accept_prob = isinf(delta_energy) ? 0.0 : min(1.0, exp(-delta_energy))

    return _TreeInfo(
        z_new, r_new, grads_new, pe_new,         # left boundary = right boundary (single leaf)
        z_new, r_new, grads_new, pe_new,
        z_new, pe_new, grads_new,                  # proposal = this leaf
        r_new,                                      # r_sum = just this momentum
        weight, leaf_accept_prob, 1, false, diverging
    )
end

"""
    _build_tree(potential_fn, z, r, z_grads, z_pe, direction, depth, step_size,
                inv_mass_matrix, energy_current) -> _TreeInfo

Recursively build a binary tree of depth `depth` by doubling.

At depth 0, builds a base tree (single leapfrog step).
At depth > 0:
1. Build the first half-tree of depth `depth-1`
2. If no U-turn or divergence, extend from the appropriate boundary
3. Merge the two half-trees: combine weights, check U-turn, select proposal
"""
function _build_tree(
    potential_fn::Function,
    z::Vector{Float64},
    r::Vector{Float64},
    z_grads::Vector{Float64},
    z_pe::Float64,
    direction::Int,
    depth::Int,
    step_size::Float64,
    inv_mass_matrix::Vector{Float64},
    energy_current::Float64
)
    if depth == 0
        return _build_basetree(
            potential_fn, z, r, z_grads, z_pe,
            direction, step_size, inv_mass_matrix, energy_current
        )
    end

    # Build the first half-tree
    half = _build_tree(
        potential_fn, z, r, z_grads, z_pe,
        direction, depth - 1, step_size, inv_mass_matrix, energy_current
    )

    # Early termination if the first half-tree is turning or diverging
    if half.turning || half.diverging
        return half
    end

    # Extend from the appropriate boundary depending on direction.
    # If direction > 0 (forward), extend from the right boundary.
    # If direction < 0 (backward), extend from the left boundary.
    if direction > 0
        other = _build_tree(
            potential_fn,
            half.z_right, half.r_right, half.z_right_grads, half.z_right_pe,
            direction, depth - 1, step_size, inv_mass_matrix, energy_current
        )
    else
        other = _build_tree(
            potential_fn,
            half.z_left, half.r_left, half.z_left_grads, half.z_left_pe,
            direction, depth - 1, step_size, inv_mass_matrix, energy_current
        )
    end

    # If the other half-tree is turning or diverging, propagate
    if other.turning || other.diverging
        # Keep the half-tree's proposal but mark as turning/diverging
        return _TreeInfo(
            direction > 0 ? half.z_left : other.z_left,
            direction > 0 ? half.r_left : other.r_left,
            direction > 0 ? half.z_left_grads : other.z_left_grads,
            direction > 0 ? half.z_left_pe : other.z_left_pe,
            direction > 0 ? other.z_right : half.z_right,
            direction > 0 ? other.r_right : half.r_right,
            direction > 0 ? other.z_right_grads : half.z_right_grads,
            direction > 0 ? other.z_right_pe : half.z_right_pe,
            half.z_proposal, half.z_proposal_pe, half.z_proposal_grads,
            half.r_sum .+ other.r_sum,
            _logaddexp(half.weight, other.weight),
            half.sum_accept_probs + other.sum_accept_probs,
            half.num_proposals + other.num_proposals,
            other.turning, other.diverging
        )
    end

    # Merge the two half-trees: select proposal via multinomial sampling.
    # P(select from other) = exp(other.weight - combined_weight)
    combined_weight = _logaddexp(half.weight, other.weight)
    accept_prob = exp(other.weight - combined_weight)

    if rand() < accept_prob
        z_proposal = other.z_proposal
        z_proposal_pe = other.z_proposal_pe
        z_proposal_grads = other.z_proposal_grads
    else
        z_proposal = half.z_proposal
        z_proposal_pe = half.z_proposal_pe
        z_proposal_grads = half.z_proposal_grads
    end

    # Merge boundaries: left comes from the leftward subtree, right from rightward
    z_left = direction > 0 ? half.z_left : other.z_left
    r_left = direction > 0 ? half.r_left : other.r_left
    z_left_grads = direction > 0 ? half.z_left_grads : other.z_left_grads
    z_left_pe = direction > 0 ? half.z_left_pe : other.z_left_pe
    z_right = direction > 0 ? other.z_right : half.z_right
    r_right = direction > 0 ? other.r_right : half.r_right
    z_right_grads = direction > 0 ? other.z_right_grads : half.z_right_grads
    z_right_pe = direction > 0 ? other.z_right_pe : half.z_right_pe

    # Combined momentum sum for U-turn check
    r_sum = half.r_sum .+ other.r_sum

    # Check U-turn on the full merged tree
    turning = _is_turning(r_left, r_right, r_sum, inv_mass_matrix)

    return _TreeInfo(
        z_left, r_left, z_left_grads, z_left_pe,
        z_right, r_right, z_right_grads, z_right_pe,
        z_proposal, z_proposal_pe, z_proposal_grads,
        r_sum, combined_weight,
        half.sum_accept_probs + other.sum_accept_probs,
        half.num_proposals + other.num_proposals,
        turning, false
    )
end

"""
    _logaddexp(a, b) -> Float64

Numerically stable computation of log(exp(a) + exp(b)).
"""
function _logaddexp(a::Float64, b::Float64)
    m = max(a, b)
    if isinf(m) && m < 0
        return -Inf
    end
    return m + log(exp(a - m) + exp(b - m))
end

"""
    sample_kernel(kernel::NUTSKernel, t::Int) -> Vector{Float64}

Draw one NUTS sample at iteration `t`.

# Algorithm (Hoffman & Gelman 2014, Algorithm 6 + multinomial extension)
1. Sample momentum r ~ N(0, M)
2. Compute initial energy H₀ = U(z) + K(r)
3. Initialize the tree with the current state
4. Repeat until U-turn or max_tree_depth:
   a. Choose random direction (forward or backward)
   b. Build a subtree of the current depth in that direction
   c. If no U-turn, accept the subtree's proposal with multinomial probability
   d. Check global U-turn across the full tree
5. Return the selected proposal
"""
function sample_kernel(kernel::NUTSKernel, t::Int)
    z = kernel.current_z
    pe = kernel.current_pe
    grads = kernel.current_grads
    inv_mass_matrix = kernel.inv_mass_matrix

    # Step 1: Sample momentum
    r = _sample_momentum(inv_mass_matrix)

    # Step 2: Compute initial Hamiltonian
    energy_current = pe + _kinetic_energy(r, inv_mass_matrix)

    # Step 3: Initialize the tree with the current state as a single leaf
    z_left = copy(z)
    r_left = copy(r)
    z_left_grads = copy(grads)
    z_left_pe = pe
    z_right = copy(z)
    r_right = copy(r)
    z_right_grads = copy(grads)
    z_right_pe = pe
    z_proposal = copy(z)
    z_proposal_pe = pe
    z_proposal_grads = copy(grads)
    r_sum = copy(r)
    tree_weight = -energy_current  # log weight of initial state
    depth = 0
    num_proposals = 1
    sum_accept_probs = 0.0

    # Step 4: Grow the tree until U-turn or max depth
    while depth < kernel.max_tree_depth
        # Random direction: extend tree forward or backward
        direction = rand([-1, 1])

        # Build a new subtree from the appropriate boundary
        if direction > 0
            new_tree = _build_tree(
                kernel.potential_fn,
                z_right, r_right, z_right_grads, z_right_pe,
                direction, depth, kernel.step_size, inv_mass_matrix,
                energy_current
            )
        else
            new_tree = _build_tree(
                kernel.potential_fn,
                z_left, r_left, z_left_grads, z_left_pe,
                direction, depth, kernel.step_size, inv_mass_matrix,
                energy_current
            )
        end

        # Check for divergence or turning in the new subtree
        if new_tree.diverging && !kernel.is_warmup
            kernel.divergence_count += 1
        end

        if new_tree.turning || new_tree.diverging
            break
        end

        # Accept the new subtree's proposal with multinomial probability
        new_tree_weight = _logaddexp(tree_weight, new_tree.weight)
        accept_prob_subtree = exp(new_tree.weight - new_tree_weight)

        if rand() < accept_prob_subtree
            z_proposal = new_tree.z_proposal
            z_proposal_pe = new_tree.z_proposal_pe
            z_proposal_grads = new_tree.z_proposal_grads
        end

        # Accumulate acceptance statistics for adaptation.
        # Each leaf in the tree computed min(1, exp(H₀ - H_leaf)) and stored
        # the sum in sum_accept_probs. We aggregate these across subtrees.
        sum_accept_probs += new_tree.sum_accept_probs
        num_proposals += new_tree.num_proposals

        # Update tree boundaries
        if direction > 0
            z_right = new_tree.z_right
            r_right = new_tree.r_right
            z_right_grads = new_tree.z_right_grads
            z_right_pe = new_tree.z_right_pe
        else
            z_left = new_tree.z_left
            r_left = new_tree.r_left
            z_left_grads = new_tree.z_left_grads
            z_left_pe = new_tree.z_left_pe
        end

        # Update combined momentum sum
        r_sum = r_sum .+ new_tree.r_sum

        # Check global U-turn
        tree_weight = new_tree_weight
        if _is_turning(r_left, r_right, r_sum, inv_mass_matrix)
            break
        end

        depth += 1
    end

    # Update cached state with the accepted proposal
    kernel.current_z = z_proposal
    kernel.current_pe = z_proposal_pe
    kernel.current_grads = z_proposal_grads

    # Update running diagnostics
    kernel.sample_count += 1
    kernel.total_tree_depth += depth
    effective_accept_prob = num_proposals > 0 ? sum_accept_probs / num_proposals : 0.0
    kernel.sum_accept_probs += effective_accept_prob
    kernel.num_proposals_total += num_proposals

    # Adaptation during warmup
    if kernel.is_warmup && kernel.adapter !== nothing
        step!(kernel.adapter, t, kernel.current_z, effective_accept_prob)

        if kernel.adapt_step_size
            kernel.step_size = get_step_size(kernel.adapter.dual_averaging)
        end
        if kernel.adapt_mass_matrix
            kernel.inv_mass_matrix = get_inv_mass_matrix(kernel.adapter)
        end
    end

    return copy(kernel.current_z)
end

"""
    end_warmup!(kernel::NUTSKernel)

Signal the end of warmup. Finalizes step size and mass matrix.
"""
function end_warmup!(kernel::NUTSKernel)
    kernel.is_warmup = false
    if kernel.adapter !== nothing
        step_size, inv_mass_matrix = finalize!(kernel.adapter)
        kernel.step_size = step_size
        kernel.inv_mass_matrix = inv_mass_matrix
    end
    return nothing
end

"""
    diagnostics(kernel::NUTSKernel) -> Dict

Return diagnostic information about the NUTS kernel's sampling performance.
"""
function diagnostics(kernel::NUTSKernel)
    return Dict(
        "step_size" => kernel.step_size,
        "accept_prob" => kernel.sample_count > 0 ?
            kernel.sum_accept_probs / kernel.sample_count : 0.0,
        "max_tree_depth" => kernel.max_tree_depth,
        "mean_tree_depth" => kernel.sample_count > 0 ?
            kernel.total_tree_depth / kernel.sample_count : 0.0,
        "divergences" => kernel.divergence_count
    )
end
