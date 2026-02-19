# Copyright (c) 2024
# SPDX-License-Identifier: Apache-2.0

# =============================================================================
# Adaptation Machinery for HMC/NUTS
#
# Implements two adaptation strategies used during the warmup phase:
#
# 1. **Dual Averaging** for step size (Nesterov 2009, Algorithm 4):
#    Finds the step size ε that achieves a target acceptance probability.
#    Uses a primal-dual averaging scheme that converges to the optimal ε.
#
# 2. **Welford Online Variance** for mass matrix estimation:
#    Estimates the diagonal mass matrix from running statistics of the
#    sampled positions, using Welford's numerically stable algorithm.
#
# 3. **WarmupAdapter** coordinates both adaptations using a windowed
#    schedule following the Stan convention:
#    - Initial buffer (15%): Only step size adaptation
#    - Middle windows (doubling sizes): Both step size and mass matrix
#    - Final buffer (10%): Only step size adaptation
#
# Reference: Pyro's pyro/infer/mcmc/adaptation.py
# =============================================================================

export DualAveraging, WelfordCovariance, WarmupAdapter
export initialize!, step!, get_step_size, get_inv_mass_matrix
export configure_warmup!, finalize!, get_covariance

"""
    DualAveraging

Nesterov's dual averaging scheme for step size adaptation.

Maintains both an "online" step size (for exploration during warmup) and a
smoothed average (used after warmup). The target statistic is:

    H_t = target_accept_prob - current_accept_prob

When H_t > 0, the acceptance rate is too low → decrease step size.
When H_t < 0, the acceptance rate is too high → increase step size.

# Fields
- `target_accept_prob::Float64`: Target MH acceptance probability (default 0.8)
- `gamma::Float64`: Controls shrinkage toward prox_center (default 0.05)
- `t0::Float64`: Stabilization offset (default 10.0)
- `kappa::Float64`: Power for step size schedule (default 0.75)
- `log_step_size::Float64`: Current log step size (for exploration)
- `log_step_size_avg::Float64`: Averaged log step size (for final value)
- `h_avg::Float64`: Running average of the dual variable
- `mu::Float64`: Proximal center = log(10 * initial_step_size)
- `t::Int`: Iteration counter

Reference: Algorithm 4 in Hoffman & Gelman (2014) "The No-U-Turn Sampler"
"""
mutable struct DualAveraging
    target_accept_prob::Float64
    gamma::Float64
    t0::Float64
    kappa::Float64
    log_step_size::Float64
    log_step_size_avg::Float64
    h_avg::Float64
    mu::Float64
    t::Int

    function DualAveraging(; target_accept_prob::Float64=0.8,
                            gamma::Float64=0.05,
                            t0::Float64=10.0,
                            kappa::Float64=0.75)
        new(target_accept_prob, gamma, t0, kappa, 0.0, 0.0, 0.0, 0.0, 0)
    end
end

"""
    initialize!(da::DualAveraging, step_size::Float64)

Initialize dual averaging with an initial step size. Sets the proximal
center μ = log(10 * step_size), which biases early adaptation toward
larger step sizes since the initial heuristic tends to underestimate.
"""
function initialize!(da::DualAveraging, step_size::Float64)
    da.log_step_size = log(step_size)
    da.log_step_size_avg = 0.0
    da.h_avg = 0.0
    da.mu = log(10.0 * step_size)
    da.t = 0
    return nothing
end

"""
    step!(da::DualAveraging, accept_prob::Float64)

Update dual averaging with the current acceptance probability.

The update rule (Hoffman & Gelman 2014, Algorithm 4):
    H_t = target - accept_prob
    h̄_t = (1 - 1/(t + t₀)) * h̄_{t-1} + (1/(t + t₀)) * H_t
    log ε_t = μ - (√t / γ) * h̄_t
    log ε̄_t = t^{-κ} * log ε_t + (1 - t^{-κ}) * log ε̄_{t-1}
"""
function step!(da::DualAveraging, accept_prob::Float64)
    da.t += 1
    t = da.t

    # Clamp acceptance probability for numerical stability
    accept_prob = clamp(accept_prob, 0.0, 1.0)

    # Error signal: positive means we need a smaller step size
    eta1 = 1.0 / (t + da.t0)
    da.h_avg = (1.0 - eta1) * da.h_avg + eta1 * (da.target_accept_prob - accept_prob)

    # Update log step size using dual averaging
    da.log_step_size = da.mu - (sqrt(t) / da.gamma) * da.h_avg

    # Update averaged log step size with decaying weight
    eta2 = t^(-da.kappa)
    da.log_step_size_avg = eta2 * da.log_step_size + (1.0 - eta2) * da.log_step_size_avg

    return nothing
end

"""
    get_step_size(da::DualAveraging; averaged::Bool=false) -> Float64

Return the current step size. If `averaged=true`, returns the smoothed
average (used after warmup). Otherwise returns the online estimate
(used during warmup for exploration).
"""
function get_step_size(da::DualAveraging; averaged::Bool=false)
    if averaged
        return exp(da.log_step_size_avg)
    else
        return exp(da.log_step_size)
    end
end

"""
    WelfordCovariance

Online estimation of variance using Welford's algorithm.

Computes a running estimate of the diagonal of the covariance matrix,
which is used as the inverse mass matrix in HMC. Welford's algorithm is
numerically stable even for large sample sizes.

# Fields
- `n::Int`: Number of samples seen
- `mean::Vector{Float64}`: Running mean
- `m2::Vector{Float64}`: Running sum of squared deviations

Reference: Welford (1962) "Note on a method for calculating corrected
sums of squares and products"
"""
mutable struct WelfordCovariance
    n::Int
    mean::Vector{Float64}
    m2::Vector{Float64}

    WelfordCovariance(dim::Int) = new(0, zeros(dim), zeros(dim))
end

"""
    step!(wc::WelfordCovariance, z::Vector{Float64})

Incorporate a new sample into the running variance estimate.
Uses the Welford update equations:
    δ = z - mean
    mean ← mean + δ / n
    δ₂ = z - mean   (note: uses updated mean)
    M₂ ← M₂ + δ * δ₂
"""
function step!(wc::WelfordCovariance, z::Vector{Float64})
    wc.n += 1
    delta = z .- wc.mean
    wc.mean .+= delta ./ wc.n
    delta2 = z .- wc.mean
    wc.m2 .+= delta .* delta2
    return nothing
end

"""
    get_covariance(wc::WelfordCovariance; regularize::Bool=true) -> Vector{Float64}

Return the estimated diagonal covariance (variance per dimension).

If `regularize=true`, applies the Stan regularization:
    Var_reg = (n / (n + 5)) * Var + 1e-3 * (5 / (n + 5))

This shrinks the estimate toward a small positive value, preventing
zero or near-zero variance estimates that would cause numerical issues.
"""
function get_covariance(wc::WelfordCovariance; regularize::Bool=true)
    if wc.n < 2
        return ones(length(wc.mean))
    end

    var = wc.m2 ./ (wc.n - 1)

    if regularize
        # Stan regularization: shrink toward 1e-3
        n = wc.n
        var = (n / (n + 5)) .* var .+ 1e-3 * (5 / (n + 5))
    end

    return var
end

"""
    WarmupAdapter

Coordinates step size and mass matrix adaptation during the warmup phase.

Uses a windowed adaptation schedule following the Stan convention:
- **Initial buffer** (first 15% of warmup): Only step size adaptation.
  The mass matrix is frozen to identity, allowing the chain to move
  away from the initial position before estimating geometry.
- **Middle windows** (doubling sizes): Both step size AND mass matrix
  adaptation. At the end of each window, the mass matrix is finalized
  from accumulated statistics and the step size is re-initialized.
- **Final buffer** (last 10%): Only step size adaptation with the
  final mass matrix, allowing the step size to converge.

# Fields
- `adapt_step_size::Bool`: Whether to adapt the step size
- `adapt_mass_matrix::Bool`: Whether to adapt the mass matrix
- `dual_averaging::DualAveraging`: Step size adapter
- `welford::WelfordCovariance`: Mass matrix estimator
- `inv_mass_matrix::Union{Vector{Float64}, Nothing}`: Current inverse mass matrix
- `dim::Int`: Dimension of the parameter space
- `initial_buffer::Int`: Steps before mass matrix adaptation starts
- `final_buffer::Int`: Steps of final step-size-only adaptation
- `window_size::Int`: Initial adaptation window size (doubles each window)
- `warmup_steps::Int`: Total warmup steps
- `current_window_start::Int`: Start of current adaptation window
- `current_window_end::Int`: End of current adaptation window
"""
mutable struct WarmupAdapter
    adapt_step_size::Bool
    adapt_mass_matrix::Bool
    dual_averaging::DualAveraging
    welford::WelfordCovariance
    inv_mass_matrix::Union{Vector{Float64}, Nothing}
    dim::Int
    initial_buffer::Int
    final_buffer::Int
    window_size::Int
    warmup_steps::Int
    current_window_start::Int
    current_window_end::Int

    function WarmupAdapter(
        dim::Int;
        adapt_step_size::Bool=true,
        adapt_mass_matrix::Bool=true,
        target_accept_prob::Float64=0.8
    )
        da = DualAveraging(target_accept_prob=target_accept_prob)
        wc = WelfordCovariance(dim)
        new(adapt_step_size, adapt_mass_matrix, da, wc, nothing, dim,
            0, 0, 0, 0, 0, 0)
    end
end

"""
    configure_warmup!(adapter::WarmupAdapter, warmup_steps::Int, initial_step_size::Float64)

Configure the warmup schedule. Sets up the buffer sizes and initial
adaptation window following the Stan convention.
"""
function configure_warmup!(adapter::WarmupAdapter, warmup_steps::Int, initial_step_size::Float64)
    adapter.warmup_steps = warmup_steps

    if warmup_steps < 20
        # Too few steps for windowed adaptation - use entire warmup
        adapter.initial_buffer = 0
        adapter.final_buffer = 0
        adapter.window_size = warmup_steps
    else
        # Stan convention: 15% initial buffer, 10% final buffer
        adapter.initial_buffer = max(1, round(Int, 0.15 * warmup_steps))
        adapter.final_buffer = max(1, round(Int, 0.10 * warmup_steps))
        adapter.window_size = max(1, warmup_steps - adapter.initial_buffer - adapter.final_buffer)
    end

    adapter.current_window_start = adapter.initial_buffer + 1
    adapter.current_window_end = min(
        adapter.current_window_start + _next_window_size(adapter) - 1,
        warmup_steps - adapter.final_buffer
    )

    # Initialize dual averaging
    if adapter.adapt_step_size
        initialize!(adapter.dual_averaging, initial_step_size)
    end

    # Initialize inverse mass matrix to identity
    adapter.inv_mass_matrix = ones(adapter.dim)
    adapter.welford = WelfordCovariance(adapter.dim)

    return nothing
end

"""
    _next_window_size(adapter) -> Int

Compute the next adaptation window size. Windows double in size,
starting from an initial size of 25 (or the available remaining steps).
"""
function _next_window_size(adapter::WarmupAdapter)
    # Initial window ~ 25 steps, then double
    return max(1, min(25, adapter.window_size))
end

"""
    step!(adapter::WarmupAdapter, t::Int, z::Vector{Float64}, accept_prob::Float64)

Perform one adaptation step at iteration `t`.

During the initial buffer, only step size is adapted.
During middle windows, both step size and mass matrix are adapted.
At window boundaries, the mass matrix is finalized and step size is reset.
During the final buffer, only step size is adapted.
"""
function step!(adapter::WarmupAdapter, t::Int, z::Vector{Float64}, accept_prob::Float64)
    # Always adapt step size during warmup
    if adapter.adapt_step_size
        step!(adapter.dual_averaging, accept_prob)
    end

    # Adapt mass matrix only during the middle windows
    in_middle = t > adapter.initial_buffer &&
                t <= adapter.warmup_steps - adapter.final_buffer
    if adapter.adapt_mass_matrix && in_middle
        step!(adapter.welford, z)
    end

    # Check if we've reached the end of a middle adaptation window
    if adapter.adapt_mass_matrix && t == adapter.current_window_end && in_middle
        # Finalize mass matrix from accumulated statistics
        var = get_covariance(adapter.welford; regularize=true)
        adapter.inv_mass_matrix = 1.0 ./ var

        # Reset Welford for next window
        adapter.welford = WelfordCovariance(adapter.dim)

        # Re-initialize step size adaptation. The factor of 10 biases
        # toward larger step sizes since the new mass matrix typically
        # permits them (following Pyro/Stan convention).
        if adapter.adapt_step_size
            current_step_size = get_step_size(adapter.dual_averaging)
            initialize!(adapter.dual_averaging, current_step_size)
        end

        # Advance to next window (double the window size, capped by final buffer)
        adapter.current_window_start = t + 1
        next_size = min(
            2 * (adapter.current_window_end - adapter.current_window_start + 1 + 25),
            adapter.warmup_steps - adapter.final_buffer - t
        )
        next_size = max(next_size, 1)
        adapter.current_window_end = min(
            t + next_size,
            adapter.warmup_steps - adapter.final_buffer
        )
    end

    return nothing
end

"""
    finalize!(adapter::WarmupAdapter) -> (Float64, Vector{Float64})

Finalize adaptation at the end of warmup. Returns the averaged step size
and the final inverse mass matrix.
"""
function finalize!(adapter::WarmupAdapter)
    step_size = if adapter.adapt_step_size
        get_step_size(adapter.dual_averaging; averaged=true)
    else
        get_step_size(adapter.dual_averaging)
    end

    inv_mass_matrix = adapter.inv_mass_matrix !== nothing ?
        adapter.inv_mass_matrix : ones(adapter.dim)

    return step_size, inv_mass_matrix
end

"""
    get_inv_mass_matrix(adapter::WarmupAdapter) -> Vector{Float64}

Return the current inverse mass matrix estimate.
"""
function get_inv_mass_matrix(adapter::WarmupAdapter)
    return adapter.inv_mass_matrix !== nothing ?
        adapter.inv_mass_matrix : ones(adapter.dim)
end
