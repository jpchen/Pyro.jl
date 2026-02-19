# Copyright (c) 2024
# SPDX-License-Identifier: Apache-2.0

# =============================================================================
# Leapfrog / Velocity Verlet Integrator
#
# Implements the symplectic Stormer-Verlet integrator for Hamiltonian dynamics.
# Given a potential energy function U(q), the integrator evolves the state
# (q, p) through Hamilton's equations:
#
#   dq/dt =  ∂H/∂p = p / M        (M = mass matrix, identity by default)
#   dp/dt = -∂H/∂q = -∇U(q)
#
# The velocity Verlet scheme is:
#   p ← p - (ε/2) * ∇U(q)        # half-step momentum
#   q ← q + ε * M⁻¹ * p          # full-step position
#   p ← p - (ε/2) * ∇U(q)        # half-step momentum
#
# This is a second-order symplectic integrator that exactly preserves the
# symplectic structure, making it ideal for HMC where volume preservation
# and time-reversibility are required for detailed balance.
#
# Reference: Pyro's pyro/ops/integrator.py
# =============================================================================

export potential_energy_grad, velocity_verlet, IntegratorState

"""
    IntegratorState

Holds the full state of the leapfrog integrator at a single point.

# Fields
- `z::Vector{Float64}`: Position (unconstrained parameters)
- `r::Vector{Float64}`: Momentum
- `potential_energy::Float64`: U(z) = -log p(z, x)
- `z_grads::Vector{Float64}`: ∇U(z), gradient of potential energy
"""
struct IntegratorState
    z::Vector{Float64}
    r::Vector{Float64}
    potential_energy::Float64
    z_grads::Vector{Float64}
end

"""
    potential_energy_grad(potential_fn, z) -> (grad, pe)

Compute the gradient and value of the potential energy function at position `z`.
Uses ForwardDiff for automatic differentiation, which is efficient and
compatible with Julia's type system (no mutation restrictions like Zygote).

Returns `(z_grads, potential_energy)`. If the computation produces NaN
(e.g., from numerical instability), returns zero gradients and `Inf` energy
so the trajectory is safely rejected.
"""
function potential_energy_grad(potential_fn::Function, z::Vector{Float64})
    # Use ForwardDiff for gradient computation - it handles scalar-valued
    # functions of vectors naturally and doesn't have Zygote's mutation issues
    result = DiffResults.GradientResult(z)
    try
        ForwardDiff.gradient!(result, potential_fn, z)
        pe = DiffResults.value(result)
        grads = DiffResults.gradient(result)

        # Guard against NaN: if energy is NaN, return Inf energy and zero
        # gradients. This causes the Metropolis step to reject, which is
        # the safe behavior (equivalent to Pyro's exception handler pattern).
        if isnan(pe) || any(isnan, grads)
            return zeros(length(z)), Inf
        end
        return copy(grads), pe
    catch e
        # Numerical failure (singular Hessian, etc.) -> reject
        return zeros(length(z)), Inf
    end
end

"""
    velocity_verlet(potential_fn, state, step_size, num_steps, inv_mass_matrix)
    -> IntegratorState

Perform `num_steps` leapfrog integration steps using the velocity Verlet scheme.

# Arguments
- `potential_fn`: Maps position vector -> scalar potential energy U(q)
- `state`: Current `IntegratorState` (position, momentum, energy, grads)
- `step_size::Float64`: Integration step size ε
- `num_steps::Int`: Number of leapfrog steps L
- `inv_mass_matrix`: Inverse mass matrix M⁻¹ (Vector for diagonal, Matrix for dense)

# Returns
- New `IntegratorState` at the end of the trajectory

# Algorithm (per step)
```
r ← r - (ε/2) * ∇U(z)          # half-step momentum update
z ← z + ε * M⁻¹ * r            # full-step position update
∇U(z), U(z) ← compute_grads(z) # recompute potential at new position
r ← r - (ε/2) * ∇U(z)          # half-step momentum update
```
"""
function velocity_verlet(
    potential_fn::Function,
    state::IntegratorState,
    step_size::Float64,
    num_steps::Int,
    inv_mass_matrix::Union{Vector{Float64}, Matrix{Float64}}
)
    z = copy(state.z)
    r = copy(state.r)
    z_grads = copy(state.z_grads)
    potential_energy = state.potential_energy

    for _ in 1:num_steps
        # Half-step momentum update: r -= (ε/2) * ∇U(z)
        r .-= (step_size / 2.0) .* z_grads

        # Full-step position update: z += ε * M⁻¹ * r
        if inv_mass_matrix isa Vector
            z .+= step_size .* inv_mass_matrix .* r
        else
            z .+= step_size .* (inv_mass_matrix * r)
        end

        # Recompute potential energy and gradients at new position
        z_grads, potential_energy = potential_energy_grad(potential_fn, z)

        # Half-step momentum update: r -= (ε/2) * ∇U(z)
        r .-= (step_size / 2.0) .* z_grads
    end

    return IntegratorState(z, r, potential_energy, z_grads)
end
