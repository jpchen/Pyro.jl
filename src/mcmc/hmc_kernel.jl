# Copyright (c) 2024
# SPDX-License-Identifier: Apache-2.0

# =============================================================================
# HMC (Hamiltonian Monte Carlo) Kernel
#
# Implements standard HMC with Metropolis correction as a standalone kernel.
# The kernel interface is designed so that different MCMC kernels (HMC, NUTS)
# can be plugged into the same MCMC orchestrator.
#
# HMC simulates Hamiltonian dynamics to propose distant states while
# maintaining high acceptance rates. The algorithm:
# 1. Sample momentum r ~ N(0, M) from the kinetic energy distribution
# 2. Simulate L leapfrog steps at step size ε
# 3. Accept/reject with the Metropolis criterion using total energy H = U + K
#
# Reference: Pyro's pyro/infer/mcmc/hmc.py
# =============================================================================

export HMCKernel, setup!, sample_kernel, end_warmup!, diagnostics

"""
    HMCKernel

Hamiltonian Monte Carlo kernel implementing the MCMCKernel interface.

# Constructor Arguments
- `potential_fn`: Function mapping parameter vector to scalar potential energy
- `step_size::Float64`: Leapfrog step size (default: 1.0)
- `num_steps::Int`: Number of leapfrog steps per proposal (default: 10)
- `adapt_step_size::Bool`: Adapt step size during warmup (default: true)
- `adapt_mass_matrix::Bool`: Adapt mass matrix during warmup (default: true)
- `target_accept_prob::Float64`: Target Metropolis acceptance rate (default: 0.8)

# Fields
- `potential_fn`: The potential energy function (negative log joint)
- `step_size::Float64`: Current step size
- `num_steps::Int`: Number of leapfrog steps
- `adapt_step_size::Bool`: Whether to adapt step size
- `adapt_mass_matrix::Bool`: Whether to adapt mass matrix
- `target_accept_prob::Float64`: Target acceptance probability
- `adapter::Union{WarmupAdapter, Nothing}`: Warmup adapter (initialized in setup!)
- `inv_mass_matrix::Vector{Float64}`: Current inverse mass matrix (diagonal)
- `current_z::Union{Vector{Float64}, Nothing}`: Cached current position
- `current_pe::Float64`: Cached current potential energy
- `current_grads::Union{Vector{Float64}, Nothing}`: Cached current gradients
- `warmup_steps::Int`: Number of warmup steps
- `is_warmup::Bool`: Whether currently in warmup phase
- `accept_count::Int`: Running acceptance count for diagnostics
- `sample_count::Int`: Total samples drawn
- `divergence_count::Int`: Number of divergent transitions detected
"""
mutable struct HMCKernel
    potential_fn::Function
    step_size::Float64
    num_steps::Int
    adapt_step_size::Bool
    adapt_mass_matrix::Bool
    target_accept_prob::Float64
    adapter::Union{WarmupAdapter, Nothing}
    inv_mass_matrix::Vector{Float64}
    current_z::Union{Vector{Float64}, Nothing}
    current_pe::Float64
    current_grads::Union{Vector{Float64}, Nothing}
    warmup_steps::Int
    is_warmup::Bool
    accept_count::Int
    sample_count::Int
    divergence_count::Int

    function HMCKernel(
        potential_fn::Function;
        step_size::Float64=1.0,
        num_steps::Int=10,
        adapt_step_size::Bool=true,
        adapt_mass_matrix::Bool=true,
        target_accept_prob::Float64=0.8
    )
        new(potential_fn, step_size, num_steps,
            adapt_step_size, adapt_mass_matrix, target_accept_prob,
            nothing, Float64[], nothing, 0.0, nothing, 0, true, 0, 0, 0)
    end
end

"""
    _kinetic_energy(r, inv_mass_matrix) -> Float64

Compute kinetic energy K(r) = 0.5 * r' * M⁻¹ * r for diagonal mass matrix.
This corresponds to the log-density of N(0, M) up to a constant.
"""
function _kinetic_energy(r::Vector{Float64}, inv_mass_matrix::Vector{Float64})
    return 0.5 * sum(r .* r .* inv_mass_matrix)
end

"""
    _sample_momentum(inv_mass_matrix) -> Vector{Float64}

Sample momentum from the kinetic energy distribution r ~ N(0, M).
For diagonal M with M⁻¹ = diag(inv_mass_matrix), the mass matrix
M = diag(1 ./ inv_mass_matrix), so we sample r ~ N(0, 1/inv_mass_matrix).
"""
function _sample_momentum(inv_mass_matrix::Vector{Float64})
    # Standard deviation = sqrt(mass) = sqrt(1/inv_mass) = 1/sqrt(inv_mass)
    scale = 1.0 ./ sqrt.(inv_mass_matrix)
    return randn(length(inv_mass_matrix)) .* scale
end

"""
    _find_reasonable_step_size(potential_fn, z, grads, inv_mass_matrix) -> Float64

Heuristic for finding a reasonable initial step size.

Takes one leapfrog step and adjusts the step size until the acceptance
probability is in a reasonable range. Following Stan/Pyro convention:
- If acceptance prob > 0.5, double step size (want larger steps)
- If acceptance prob < 0.5, halve step size (want smaller steps)
- Repeat until the acceptance probability crosses 0.5

This is only called once at initialization to seed the dual averaging.
"""
function _find_reasonable_step_size(
    potential_fn::Function,
    z::Vector{Float64},
    grads::Vector{Float64},
    inv_mass_matrix::Vector{Float64}
)
    step_size = 1.0

    # Sample momentum and compute initial energy
    r = _sample_momentum(inv_mass_matrix)
    pe = potential_fn(z)
    ke = _kinetic_energy(r, inv_mass_matrix)
    energy_current = pe + ke

    # Take one leapfrog step
    state = IntegratorState(z, r, pe, grads)
    new_state = velocity_verlet(potential_fn, state, step_size, 1, inv_mass_matrix)
    energy_new = new_state.potential_energy + _kinetic_energy(new_state.r, inv_mass_matrix)

    delta_energy = energy_new - energy_current
    # Handle NaN
    if isnan(delta_energy)
        delta_energy = Inf
    end

    direction = log(0.8) < -delta_energy ? 1 : -1

    for _ in 1:100  # Max iterations to prevent infinite loops
        step_size_new = direction == 1 ? step_size * 2.0 : step_size / 2.0

        # Clamp to reasonable range
        step_size_new = clamp(step_size_new, 1e-10, 1e10)
        if step_size_new == step_size
            break
        end
        step_size = step_size_new

        # Take one step with new step size
        new_state = velocity_verlet(potential_fn, state, step_size, 1, inv_mass_matrix)
        energy_new = new_state.potential_energy + _kinetic_energy(new_state.r, inv_mass_matrix)
        delta_energy = energy_new - energy_current
        if isnan(delta_energy)
            delta_energy = Inf
        end

        # Check if we've crossed the threshold
        new_direction = log(0.8) < -delta_energy ? 1 : -1
        if new_direction != direction
            break
        end
    end

    return clamp(step_size, 1e-10, 1e10)
end

"""
    setup!(kernel::HMCKernel, warmup_steps::Int, initial_params::Vector{Float64})

Initialize the HMC kernel before sampling begins. Computes initial
gradients, finds a reasonable step size, and sets up the warmup adapter.
"""
function setup!(kernel::HMCKernel, warmup_steps::Int, initial_params::Vector{Float64})
    dim = length(initial_params)
    kernel.warmup_steps = warmup_steps
    kernel.is_warmup = true
    kernel.accept_count = 0
    kernel.sample_count = 0
    kernel.divergence_count = 0

    # Initialize position and compute initial potential energy + gradients
    kernel.current_z = copy(initial_params)
    kernel.current_grads, kernel.current_pe = potential_energy_grad(
        kernel.potential_fn, kernel.current_z
    )

    # Initialize inverse mass matrix to identity
    kernel.inv_mass_matrix = ones(dim)

    # Find a reasonable initial step size using the heuristic
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
    sample_kernel(kernel::HMCKernel, t::Int) -> Vector{Float64}

Draw one HMC sample at iteration `t`.

# Algorithm
1. Sample momentum r ~ N(0, M)
2. Compute current Hamiltonian H = U(z) + K(r)
3. Run `num_steps` leapfrog steps
4. Compute proposed Hamiltonian H' = U(z') + K(r')
5. Accept with probability min(1, exp(H - H'))
6. During warmup, adapt step size and mass matrix
"""
function sample_kernel(kernel::HMCKernel, t::Int)
    z = kernel.current_z
    pe = kernel.current_pe
    grads = kernel.current_grads
    inv_mass_matrix = kernel.inv_mass_matrix

    # Step 1: Sample momentum
    r = _sample_momentum(inv_mass_matrix)

    # Step 2: Current Hamiltonian (potential + kinetic energy)
    energy_current = pe + _kinetic_energy(r, inv_mass_matrix)

    # Step 3: Leapfrog integration
    state = IntegratorState(z, r, pe, grads)
    new_state = velocity_verlet(
        kernel.potential_fn, state, kernel.step_size, kernel.num_steps,
        inv_mass_matrix
    )

    # Step 4: Proposed Hamiltonian
    energy_proposed = new_state.potential_energy + _kinetic_energy(new_state.r, inv_mass_matrix)
    delta_energy = energy_proposed - energy_current
    if isnan(delta_energy)
        delta_energy = Inf
    end

    # Step 5: Metropolis accept/reject
    accept_prob = min(1.0, exp(-delta_energy))
    if rand() < accept_prob
        # Accept
        kernel.current_z = new_state.z
        kernel.current_pe = new_state.potential_energy
        kernel.current_grads = new_state.z_grads
        kernel.accept_count += 1
    end
    # If rejected, keep current state (already cached)

    kernel.sample_count += 1

    # Detect divergences: energy difference > 1000 nats indicates
    # the integrator has gone off the rails
    if !kernel.is_warmup && delta_energy > 1000.0
        kernel.divergence_count += 1
    end

    # Step 6: Adaptation during warmup
    if kernel.is_warmup && kernel.adapter !== nothing
        step!(kernel.adapter, t, kernel.current_z, accept_prob)

        # Update step size from dual averaging
        if kernel.adapt_step_size
            kernel.step_size = get_step_size(kernel.adapter.dual_averaging)
        end

        # Update inverse mass matrix
        if kernel.adapt_mass_matrix
            kernel.inv_mass_matrix = get_inv_mass_matrix(kernel.adapter)
        end
    end

    return copy(kernel.current_z)
end

"""
    end_warmup!(kernel::HMCKernel)

Signal the end of warmup. Finalizes the step size and mass matrix
from the adapter, switching to the averaged values.
"""
function end_warmup!(kernel::HMCKernel)
    kernel.is_warmup = false
    if kernel.adapter !== nothing
        step_size, inv_mass_matrix = finalize!(kernel.adapter)
        kernel.step_size = step_size
        kernel.inv_mass_matrix = inv_mass_matrix
    end
    return nothing
end

"""
    diagnostics(kernel::HMCKernel) -> Dict

Return diagnostic information about the kernel's sampling performance.
"""
function diagnostics(kernel::HMCKernel)
    return Dict(
        "step_size" => kernel.step_size,
        "accept_rate" => kernel.sample_count > 0 ?
            kernel.accept_count / kernel.sample_count : 0.0,
        "num_steps" => kernel.num_steps,
        "divergences" => kernel.divergence_count
    )
end
