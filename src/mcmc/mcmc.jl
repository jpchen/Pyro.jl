# Copyright (c) 2024
# SPDX-License-Identifier: Apache-2.0

# =============================================================================
# MCMC Module - Public API and Orchestrator
#
# This is the main entry point for MCMC inference in MiniPyro. It provides:
#
# 1. A high-level `run_mcmc` function that orchestrates warmup and sampling
# 2. A `potential_energy_from_model` helper to bridge MiniPyro models to
#    the potential energy function interface expected by HMC/NUTS kernels
# 3. An `MCMCResult` struct that holds samples and diagnostics
#
# The module is designed so that different kernels (HMC, NUTS, or future
# custom kernels) can be plugged in as long as they implement:
#   - setup!(kernel, warmup_steps, initial_params)
#   - sample_kernel(kernel, t) -> Vector{Float64}
#   - end_warmup!(kernel)
#   - diagnostics(kernel) -> Dict
#
# Usage:
#   # With a potential energy function directly:
#   kernel = NUTSKernel(potential_fn)
#   result = run_mcmc(kernel, num_samples=1000, initial_params=zeros(2))
#
#   # With a MiniPyro model:
#   potential_fn = potential_energy_from_model(model, data)
#   kernel = NUTSKernel(potential_fn)
#   result = run_mcmc(kernel, num_samples=1000, initial_params=zeros(1))
#
# Reference: Pyro's pyro/infer/mcmc/api.py
# =============================================================================

module MCMC

using Distributions
using ForwardDiff
using LinearAlgebra
using DiffResults
using Random
using Statistics

# Include component files in dependency order
include("integrator.jl")
include("adaptation.jl")
include("hmc_kernel.jl")
include("nuts_kernel.jl")

export IntegratorState, potential_energy_grad, velocity_verlet
export DualAveraging, WelfordCovariance, WarmupAdapter
export initialize!, get_step_size, get_inv_mass_matrix
export configure_warmup!, finalize!, get_covariance
export HMCKernel, NUTSKernel
export setup!, sample_kernel, end_warmup!, diagnostics
export MCMCResult, run_mcmc, potential_energy_from_model

"""
    MCMCResult

Container for MCMC sampling results.

# Fields
- `samples::Matrix{Float64}`: Samples matrix of shape (num_samples, dim).
  Each row is one sample from the posterior.
- `warmup_samples::Matrix{Float64}`: Warmup samples (usually discarded).
- `diagnostics::Dict`: Kernel diagnostics (step size, acceptance rate, etc.)
- `elapsed_time::Float64`: Wall-clock time in seconds

# Accessors
- `result.samples[:, i]` - all samples of parameter i
- `mean(result)` - posterior mean per dimension
- `std(result)` - posterior std per dimension
"""
struct MCMCResult
    samples::Matrix{Float64}
    warmup_samples::Matrix{Float64}
    diagnostics::Dict
    elapsed_time::Float64
end

"""
    Base.show(io, result::MCMCResult)

Pretty-print MCMC results summary.
"""
function Base.show(io::IO, result::MCMCResult)
    n, d = size(result.samples)
    println(io, "MCMCResult")
    println(io, "  Samples: $n x $d (num_samples x dim)")
    println(io, "  Time: $(round(result.elapsed_time, digits=2))s")
    for (k, v) in result.diagnostics
        println(io, "  $k: $v")
    end
end

"""
    Statistics.mean(result::MCMCResult) -> Vector{Float64}

Compute the posterior mean for each parameter dimension.
"""
function Statistics.mean(result::MCMCResult)
    return vec(mean(result.samples, dims=1))
end

"""
    Statistics.std(result::MCMCResult) -> Vector{Float64}

Compute the posterior standard deviation for each parameter dimension.
"""
function Statistics.std(result::MCMCResult)
    return vec(std(result.samples, dims=1))
end

"""
    run_mcmc(kernel, num_samples, initial_params;
             warmup_steps=num_samples, progress=true) -> MCMCResult

Run MCMC inference using the given kernel.

This is the main entry point for MCMC sampling. It handles the warmup phase
(where step size and mass matrix are adapted) and the sampling phase.

# Arguments
- `kernel`: An MCMC kernel (HMCKernel or NUTSKernel)
- `num_samples::Int`: Number of posterior samples to collect
- `initial_params::Vector{Float64}`: Starting position in parameter space

# Keyword Arguments
- `warmup_steps::Int`: Number of warmup (burn-in) steps (default: same as num_samples)
- `progress::Bool`: Whether to print progress messages (default: true)

# Returns
- `MCMCResult` containing samples, diagnostics, and timing

# Example
```julia
# Define potential energy (negative log posterior)
potential_fn(z) = 0.5 * sum(z .^ 2)  # Standard normal

# Run NUTS
kernel = NUTSKernel(potential_fn)
result = run_mcmc(kernel, 1000, zeros(2), warmup_steps=500)

# Inspect results
println("Posterior mean: ", mean(result))
println("Posterior std:  ", std(result))
```
"""
function run_mcmc(
    kernel,
    num_samples::Int,
    initial_params::Vector{Float64};
    warmup_steps::Int=num_samples,
    progress::Bool=true
)
    dim = length(initial_params)
    start_time = time()

    # Phase 1: Setup
    if progress
        println("Running MCMC with $(typeof(kernel).name.name)...")
        println("  Warmup: $warmup_steps steps")
        println("  Samples: $num_samples")
        println("  Dimensions: $dim")
    end

    setup!(kernel, warmup_steps, initial_params)

    # Phase 2: Warmup (samples are collected but typically discarded)
    warmup_samples = Matrix{Float64}(undef, warmup_steps, dim)
    for t in 1:warmup_steps
        warmup_samples[t, :] = sample_kernel(kernel, t)
    end

    # Transition from warmup to sampling
    end_warmup!(kernel)

    if progress
        diag = diagnostics(kernel)
        step_size = get(diag, "step_size", "N/A")
        println("  Adapted step size: $(round(step_size, digits=6))")
    end

    # Phase 3: Sampling
    samples = Matrix{Float64}(undef, num_samples, dim)
    for t in 1:num_samples
        samples[t, :] = sample_kernel(kernel, warmup_steps + t)
    end

    elapsed = time() - start_time

    # Collect diagnostics
    diag = diagnostics(kernel)

    if progress
        println("  Elapsed: $(round(elapsed, digits=2))s")
        accept_key = haskey(diag, "accept_rate") ? "accept_rate" : "accept_prob"
        if haskey(diag, accept_key)
            println("  Accept rate: $(round(diag[accept_key], digits=3))")
        end
        if haskey(diag, "divergences") && diag["divergences"] > 0
            println("  WARNING: $(diag["divergences"]) divergent transitions detected!")
        end
    end

    return MCMCResult(samples, warmup_samples, diag, elapsed)
end

"""
    potential_energy_from_model(model::Function, data...;
                                 observed::Dict{String,Any}=Dict(),
                                 param_names::Vector{String}=String[],
                                 param_priors::Dict{String,Distribution}=Dict()) -> Function

Create a potential energy function from a MiniPyro-style model specification.

This is a convenience function for simple models. For complex models, you may
want to write the potential energy function directly.

The potential energy is the negative log joint density:
    U(z) = -log p(z) - log p(data | z)

# Arguments
- `model_logpdf`: A function `(params_dict) -> log_joint` that computes the
  log joint density given a dictionary of parameter values

# Returns
- A function `potential_fn(z::Vector{Float64}) -> Float64`

# Example
```julia
# For a Bayesian linear regression:
# y ~ Normal(w*x + b, sigma)
# w ~ Normal(0, 1), b ~ Normal(0, 1)
function my_potential(z)
    w, b = z[1], z[2]
    # Prior
    lp = logpdf(Normal(0, 1), w) + logpdf(Normal(0, 1), b)
    # Likelihood
    for i in 1:length(x_data)
        lp += logpdf(Normal(w * x_data[i] + b, 1.0), y_data[i])
    end
    return -lp  # potential = negative log joint
end
```
"""
function potential_energy_from_model(model_logpdf::Function)
    return z -> -model_logpdf(z)
end

end # module MCMC
