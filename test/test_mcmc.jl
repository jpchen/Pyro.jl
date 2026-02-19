# Copyright (c) 2024
# SPDX-License-Identifier: Apache-2.0

"""
Test Suite for MCMC Module (HMC and NUTS)

This file tests all components of the MCMC inference stack:
  1. Leapfrog integrator (velocity Verlet)
  2. Adaptation machinery (dual averaging, Welford covariance, warmup adapter)
  3. HMC kernel
  4. NUTS kernel
  5. End-to-end posterior inference on synthetic data

Run tests with:
    julia --project=. test/test_mcmc.jl

Or run the full suite:
    julia --project=. test/runtests.jl
"""

using Test
using Random
using Statistics
using Distributions
using LinearAlgebra

# Load MiniPyro and the MCMC submodule
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
include(joinpath(@__DIR__, "..", "src", "MiniPyro.jl"))
using .MiniPyro
using .MiniPyro.MCMC

println("=" ^ 60)
println("MCMC Module Test Suite")
println("=" ^ 60)
println()

# =============================================================================
# Test: Potential Energy Gradient (ForwardDiff)
# =============================================================================

@testset "Potential Energy Gradient" begin
    println("Testing Potential Energy Gradient...")

    # Simple quadratic potential: U(z) = 0.5 * z'z (standard normal)
    potential_fn(z) = 0.5 * sum(z .^ 2)

    z = [1.0, 2.0, 3.0]
    grads, pe = potential_energy_grad(potential_fn, z)

    # U(z) = 0.5 * (1 + 4 + 9) = 7.0
    @test pe ≈ 7.0

    # ∇U(z) = z = [1, 2, 3]
    @test grads ≈ z

    # Test at origin: gradient should be zero
    z0 = [0.0, 0.0]
    grads0, pe0 = potential_energy_grad(potential_fn, z0)
    @test pe0 ≈ 0.0
    @test grads0 ≈ [0.0, 0.0]

    # Test with non-trivial potential: Rosenbrock-like
    rosenbrock(z) = (1.0 - z[1])^2 + 100.0 * (z[2] - z[1]^2)^2
    z_rb = [1.0, 1.0]  # At the minimum
    grads_rb, pe_rb = potential_energy_grad(rosenbrock, z_rb)
    @test pe_rb ≈ 0.0 atol=1e-10
    @test grads_rb ≈ [0.0, 0.0] atol=1e-10

    # Test NaN handling: function that can produce NaN
    bad_potential(z) = log(z[1])  # Undefined for z[1] <= 0
    grads_bad, pe_bad = potential_energy_grad(bad_potential, [-1.0])
    @test pe_bad == Inf || isnan(pe_bad)  # Should return Inf for safety

    println("  ✓ Potential Energy Gradient tests passed")
end

# =============================================================================
# Test: Velocity Verlet Integrator
# =============================================================================

@testset "Velocity Verlet Integrator" begin
    println("Testing Velocity Verlet Integrator...")

    # Harmonic oscillator: U(z) = 0.5 * z'z
    # This has an exact solution: circular orbits in phase space.
    # After a full period (2π), z and r should return to their starting values.
    potential_fn(z) = 0.5 * sum(z .^ 2)

    z0 = [1.0, 0.0]
    r0 = [0.0, 1.0]
    grads0 = [1.0, 0.0]  # ∇U = z
    pe0 = 0.5
    inv_mass = [1.0, 1.0]

    state = IntegratorState(z0, r0, pe0, grads0)

    # Test energy conservation: total energy should be approximately preserved
    # E = U + K = 0.5*(1+0) + 0.5*(0+1) = 1.0
    initial_energy = pe0 + 0.5 * sum(r0 .^ 2 .* inv_mass)

    # Small step size, many steps for accuracy
    new_state = velocity_verlet(potential_fn, state, 0.01, 100, inv_mass)
    final_energy = new_state.potential_energy + 0.5 * sum(new_state.r .^ 2 .* inv_mass)

    # Symplectic integrator preserves energy to O(ε²)
    @test abs(final_energy - initial_energy) < 0.01

    # Test that position changes (not stuck)
    @test new_state.z != z0

    # Test with identity mass matrix (vector form)
    state2 = IntegratorState([0.5], [0.5], 0.125, [0.5])
    new_state2 = velocity_verlet(potential_fn, state2, 0.1, 10, [1.0])
    energy2_init = 0.125 + 0.5 * 0.25
    energy2_final = new_state2.potential_energy + 0.5 * sum(new_state2.r .^ 2)
    @test abs(energy2_final - energy2_init) < 0.01

    # Test time-reversibility: running forward then backward should return
    # (approximately) to the starting point
    state3 = IntegratorState([1.0], [0.5], 0.5, [1.0])
    fwd = velocity_verlet(potential_fn, state3, 0.05, 20, [1.0])
    # Negate momentum for time reversal
    rev_state = IntegratorState(fwd.z, -fwd.r, fwd.potential_energy, fwd.z_grads)
    rev = velocity_verlet(potential_fn, rev_state, 0.05, 20, [1.0])
    @test rev.z ≈ state3.z atol=1e-6

    println("  ✓ Velocity Verlet Integrator tests passed")
end

# =============================================================================
# Test: Dual Averaging (Step Size Adaptation)
# =============================================================================

@testset "Dual Averaging" begin
    println("Testing Dual Averaging...")

    da = DualAveraging(target_accept_prob=0.8)
    initialize!(da, 1.0)

    @test da.t == 0
    @test da.mu ≈ log(10.0)  # log(10 * step_size)

    # Simulate high acceptance → step size should increase
    for _ in 1:100
        step!(da, 0.95)  # Acceptance prob > target
    end
    step_size_high = get_step_size(da)
    @test step_size_high > 1.0  # Should grow

    # Reset and simulate low acceptance → step size should decrease
    initialize!(da, 1.0)
    for _ in 1:100
        step!(da, 0.3)  # Acceptance prob < target
    end
    step_size_low = get_step_size(da)
    @test step_size_low < 1.0  # Should shrink

    # Averaged step size should be defined
    avg = get_step_size(da; averaged=true)
    @test isfinite(avg) && avg > 0

    println("  ✓ Dual Averaging tests passed")
end

# =============================================================================
# Test: Welford Online Covariance
# =============================================================================

@testset "Welford Covariance" begin
    println("Testing Welford Covariance...")

    Random.seed!(42)

    # Generate samples from a known distribution
    true_var = [4.0, 1.0, 9.0]
    true_std = sqrt.(true_var)
    n_samples = 10000

    wc = WelfordCovariance(3)

    for _ in 1:n_samples
        z = randn(3) .* true_std
        step!(wc, z)
    end

    @test wc.n == n_samples

    # Estimated variance should be close to true variance
    est_var = get_covariance(wc; regularize=false)
    @test est_var ≈ true_var rtol=0.1  # 10% tolerance for 10k samples

    # Regularized variance should be slightly different but still close
    est_var_reg = get_covariance(wc; regularize=true)
    @test all(est_var_reg .> 0)

    # Edge case: fewer than 2 samples should return ones
    wc_small = WelfordCovariance(2)
    step!(wc_small, [1.0, 2.0])
    @test get_covariance(wc_small) == ones(2)

    println("  ✓ Welford Covariance tests passed")
end

# =============================================================================
# Test: Warmup Adapter
# =============================================================================

@testset "Warmup Adapter" begin
    println("Testing Warmup Adapter...")

    Random.seed!(42)
    dim = 2

    adapter = WarmupAdapter(dim; adapt_step_size=true, adapt_mass_matrix=true)
    configure_warmup!(adapter, 100, 1.0)

    @test adapter.warmup_steps == 100
    @test adapter.inv_mass_matrix ≈ ones(dim)
    @test adapter.initial_buffer > 0
    @test adapter.final_buffer > 0

    # Simulate warmup steps
    for t in 1:100
        z = randn(dim) .* [2.0, 0.5]  # Anisotropic samples
        step!(adapter, t, z, 0.7)
    end

    # After warmup, mass matrix should reflect the anisotropy
    step_size, inv_mass = finalize!(adapter)
    @test isfinite(step_size) && step_size > 0
    @test all(isfinite.(inv_mass))
    @test all(inv_mass .> 0)

    # Short warmup (< 20 steps) should still work
    adapter2 = WarmupAdapter(dim)
    configure_warmup!(adapter2, 10, 0.5)
    @test adapter2.initial_buffer == 0
    for t in 1:10
        step!(adapter2, t, randn(dim), 0.8)
    end
    s, m = finalize!(adapter2)
    @test isfinite(s) && s > 0

    println("  ✓ Warmup Adapter tests passed")
end

# =============================================================================
# Test: HMC Kernel
# =============================================================================

@testset "HMC Kernel" begin
    println("Testing HMC Kernel...")

    Random.seed!(123)

    # Standard normal posterior: U(z) = 0.5 * z'z
    potential_fn(z) = 0.5 * sum(z .^ 2)
    dim = 2

    kernel = HMCKernel(
        potential_fn;
        step_size=0.1,
        num_steps=10,
        adapt_step_size=true,
        adapt_mass_matrix=true,
        target_accept_prob=0.8
    )

    # Setup
    setup!(kernel, 50, zeros(dim))
    @test kernel.current_z !== nothing
    @test length(kernel.current_z) == dim
    @test kernel.step_size > 0

    # Warmup
    for t in 1:50
        z = sample_kernel(kernel, t)
        @test length(z) == dim
        @test all(isfinite.(z))
    end

    # End warmup
    end_warmup!(kernel)
    @test !kernel.is_warmup

    # Sampling - collect samples and check they come from N(0, I)
    samples = zeros(500, dim)
    for t in 1:500
        samples[t, :] = sample_kernel(kernel, 50 + t)
    end

    # Posterior mean should be near zero
    sample_mean = vec(mean(samples, dims=1))
    @test all(abs.(sample_mean) .< 0.3)

    # Posterior std should be near 1
    sample_std = vec(std(samples, dims=1))
    @test all(abs.(sample_std .- 1.0) .< 0.3)

    # Diagnostics should be populated
    diag = diagnostics(kernel)
    @test haskey(diag, "step_size")
    @test haskey(diag, "accept_rate")
    @test diag["accept_rate"] > 0.3  # Should have reasonable acceptance

    println("  ✓ HMC Kernel tests passed")
end

# =============================================================================
# Test: NUTS Kernel
# =============================================================================

@testset "NUTS Kernel" begin
    println("Testing NUTS Kernel...")

    Random.seed!(456)

    # Standard normal posterior: U(z) = 0.5 * z'z
    potential_fn(z) = 0.5 * sum(z .^ 2)
    dim = 2

    kernel = NUTSKernel(
        potential_fn;
        step_size=0.1,
        adapt_step_size=true,
        adapt_mass_matrix=true,
        target_accept_prob=0.8,
        max_tree_depth=10
    )

    # Setup
    setup!(kernel, 50, zeros(dim))
    @test kernel.current_z !== nothing
    @test kernel.step_size > 0

    # Warmup
    for t in 1:50
        z = sample_kernel(kernel, t)
        @test length(z) == dim
        @test all(isfinite.(z))
    end

    end_warmup!(kernel)

    # Sampling
    samples = zeros(500, dim)
    for t in 1:500
        samples[t, :] = sample_kernel(kernel, 50 + t)
    end

    # Posterior mean near zero
    sample_mean = vec(mean(samples, dims=1))
    @test all(abs.(sample_mean) .< 0.3)

    # Posterior std near 1
    sample_std = vec(std(samples, dims=1))
    @test all(abs.(sample_std .- 1.0) .< 0.3)

    # Diagnostics
    diag = diagnostics(kernel)
    @test haskey(diag, "step_size")
    @test haskey(diag, "accept_prob")
    @test haskey(diag, "mean_tree_depth")
    @test diag["mean_tree_depth"] > 0  # Tree should have grown

    println("  ✓ NUTS Kernel tests passed")
end

# =============================================================================
# Test: run_mcmc Orchestrator
# =============================================================================

@testset "run_mcmc Orchestrator" begin
    println("Testing run_mcmc Orchestrator...")

    Random.seed!(789)

    # Standard normal
    potential_fn(z) = 0.5 * sum(z .^ 2)

    # Test with NUTS
    kernel = NUTSKernel(potential_fn)
    result = run_mcmc(kernel, 200, zeros(2); warmup_steps=100, progress=false)

    @test size(result.samples) == (200, 2)
    @test size(result.warmup_samples) == (100, 2)
    @test result.elapsed_time > 0
    @test haskey(result.diagnostics, "step_size")

    # Mean and std accessors
    m = mean(result)
    s = std(result)
    @test length(m) == 2
    @test length(s) == 2

    # Test with HMC
    kernel_hmc = HMCKernel(potential_fn; num_steps=10)
    result_hmc = run_mcmc(kernel_hmc, 200, zeros(2); warmup_steps=100, progress=false)
    @test size(result_hmc.samples) == (200, 2)

    println("  ✓ run_mcmc Orchestrator tests passed")
end

# =============================================================================
# Test: Bayesian Linear Regression (Synthetic Data)
#
# This is the key integration test: we generate synthetic data from a known
# model, run NUTS to infer the posterior, and check that the posterior mean
# is close to the true parameters.
#
# Model:
#   w ~ Normal(0, 1)      (weight)
#   b ~ Normal(0, 1)      (bias)
#   y_i ~ Normal(w*x_i + b, σ)  for i = 1..N
#
# With enough data, the posterior should concentrate near the true (w, b).
# =============================================================================

@testset "Bayesian Linear Regression (NUTS)" begin
    println("Testing Bayesian Linear Regression with NUTS...")

    Random.seed!(42)

    # True parameters
    true_w = 2.5
    true_b = -1.0
    sigma = 0.5
    n_data = 100

    # Generate synthetic data
    x_data = randn(n_data)
    y_data = true_w .* x_data .+ true_b .+ sigma .* randn(n_data)

    # Potential energy = -log p(w, b, y | x)
    # z = [w, b]
    function potential_fn(z)
        w, b = z[1], z[2]
        # Prior: w ~ N(0, 1), b ~ N(0, 1)
        lp = -0.5 * w^2 - 0.5 * b^2
        # Likelihood: y_i ~ N(w*x_i + b, sigma)
        for i in 1:n_data
            residual = y_data[i] - (w * x_data[i] + b)
            lp += -0.5 * (residual / sigma)^2 - log(sigma)
        end
        return -lp
    end

    # Run NUTS
    kernel = NUTSKernel(potential_fn; max_tree_depth=8)
    result = run_mcmc(kernel, 500, [0.0, 0.0]; warmup_steps=500, progress=false)

    posterior_mean = mean(result)
    posterior_std = std(result)

    println("  True w=$(true_w), estimated w=$(round(posterior_mean[1], digits=3)) ± $(round(posterior_std[1], digits=3))")
    println("  True b=$(true_b), estimated b=$(round(posterior_mean[2], digits=3)) ± $(round(posterior_std[2], digits=3))")

    # Posterior mean should be close to true values
    @test abs(posterior_mean[1] - true_w) < 0.3
    @test abs(posterior_mean[2] - true_b) < 0.3

    # Posterior std should be small (concentrated posterior)
    @test posterior_std[1] < 0.2
    @test posterior_std[2] < 0.2

    # No divergences
    @test result.diagnostics["divergences"] == 0

    println("  ✓ Bayesian Linear Regression tests passed")
end

# =============================================================================
# Test: Bayesian Normal Mean Inference (NUTS)
#
# The simplest possible model to verify correctness:
#   mu ~ Normal(0, prior_std)
#   x_i ~ Normal(mu, likelihood_std) for i = 1..N
#
# The exact posterior is:
#   mu | x ~ Normal(posterior_mean, posterior_std)
# where:
#   posterior_precision = 1/prior_std^2 + N/likelihood_std^2
#   posterior_mean = (N * x_bar / likelihood_std^2) / posterior_precision
#   posterior_std = 1 / sqrt(posterior_precision)
# =============================================================================

@testset "Normal Mean Inference (NUTS, exact posterior)" begin
    println("Testing Normal Mean Inference with exact posterior check...")

    Random.seed!(2024)

    # Setup
    prior_std = 1.0
    likelihood_std = 1.0
    true_mu = 3.0
    n_data = 50

    # Generate data
    data = true_mu .+ likelihood_std .* randn(n_data)
    x_bar = mean(data)

    # Exact posterior
    post_precision = 1.0 / prior_std^2 + n_data / likelihood_std^2
    post_mean = (n_data * x_bar / likelihood_std^2) / post_precision
    post_std = 1.0 / sqrt(post_precision)

    # Potential energy (1D)
    function potential_fn(z)
        mu = z[1]
        # Prior
        lp = -0.5 * (mu / prior_std)^2
        # Likelihood
        for i in 1:n_data
            lp += -0.5 * ((data[i] - mu) / likelihood_std)^2
        end
        return -lp
    end

    # Run NUTS
    kernel = NUTSKernel(potential_fn)
    result = run_mcmc(kernel, 1000, [0.0]; warmup_steps=500, progress=false)

    mcmc_mean = mean(result)[1]
    mcmc_std = std(result)[1]

    println("  Exact posterior: mean=$(round(post_mean, digits=4)), std=$(round(post_std, digits=4))")
    println("  MCMC  posterior: mean=$(round(mcmc_mean, digits=4)), std=$(round(mcmc_std, digits=4))")

    # MCMC mean should match exact posterior mean within Monte Carlo error
    @test abs(mcmc_mean - post_mean) < 3 * post_std / sqrt(1000)  # ~3 SE

    # MCMC std should be within 30% of exact posterior std
    @test abs(mcmc_std - post_std) / post_std < 0.3

    println("  ✓ Normal Mean Inference tests passed")
end

# =============================================================================
# Test: Correlated 2D Normal (NUTS)
#
# Tests that NUTS handles correlated posteriors correctly by sampling from
# a 2D normal with known covariance structure.
# =============================================================================

@testset "Correlated 2D Normal (NUTS)" begin
    println("Testing Correlated 2D Normal with NUTS...")

    Random.seed!(321)

    # Target: N([1, -1], Σ) with correlation
    target_mean = [1.0, -1.0]
    target_cov = [1.0 0.8; 0.8 1.0]
    target_precision = inv(target_cov)

    # Potential energy: U(z) = 0.5 * (z - μ)' Σ⁻¹ (z - μ)
    function potential_fn(z)
        d = z .- target_mean
        return 0.5 * dot(d, target_precision * d)
    end

    kernel = NUTSKernel(potential_fn; max_tree_depth=8)
    result = run_mcmc(kernel, 1000, [0.0, 0.0]; warmup_steps=500, progress=false)

    mcmc_mean = mean(result)
    mcmc_cov = cov(result.samples)

    println("  Target mean: $target_mean")
    println("  MCMC mean:   $(round.(mcmc_mean, digits=3))")

    # Mean should be close
    @test all(abs.(mcmc_mean .- target_mean) .< 0.2)

    # Diagonal variances should be close
    @test abs(mcmc_cov[1, 1] - target_cov[1, 1]) < 0.3
    @test abs(mcmc_cov[2, 2] - target_cov[2, 2]) < 0.3

    # Correlation should be approximately preserved
    mcmc_corr = mcmc_cov[1, 2] / sqrt(mcmc_cov[1, 1] * mcmc_cov[2, 2])
    target_corr = target_cov[1, 2] / sqrt(target_cov[1, 1] * target_cov[2, 2])
    @test abs(mcmc_corr - target_corr) < 0.2

    println("  ✓ Correlated 2D Normal tests passed")
end

# =============================================================================
# Test: HMC vs NUTS Agreement
#
# Both samplers should converge to the same posterior for a simple problem.
# =============================================================================

@testset "HMC vs NUTS Agreement" begin
    println("Testing HMC vs NUTS agreement...")

    # Simple 1D normal: should be easy for both
    potential_fn(z) = 0.5 * z[1]^2

    # HMC
    Random.seed!(100)
    hmc = HMCKernel(potential_fn; num_steps=10)
    hmc_result = run_mcmc(hmc, 500, [0.0]; warmup_steps=200, progress=false)

    # NUTS
    Random.seed!(100)
    nuts = NUTSKernel(potential_fn)
    nuts_result = run_mcmc(nuts, 500, [0.0]; warmup_steps=200, progress=false)

    # Both should give mean ≈ 0 and std ≈ 1
    @test abs(mean(hmc_result)[1]) < 0.3
    @test abs(mean(nuts_result)[1]) < 0.3
    @test abs(std(hmc_result)[1] - 1.0) < 0.3
    @test abs(std(nuts_result)[1] - 1.0) < 0.3

    println("  ✓ HMC vs NUTS agreement tests passed")
end

# =============================================================================
# Test: potential_energy_from_model helper
# =============================================================================

@testset "potential_energy_from_model" begin
    println("Testing potential_energy_from_model helper...")

    # A log-pdf function for N(0, 1)
    logpdf_fn(z) = -0.5 * sum(z .^ 2) - length(z) * 0.5 * log(2π)

    potential_fn = potential_energy_from_model(logpdf_fn)

    z = [1.0, 0.0]
    pe = potential_fn(z)

    # Should be -logpdf_fn(z)
    expected = 0.5 * sum(z .^ 2) + length(z) * 0.5 * log(2π)
    @test pe ≈ expected

    println("  ✓ potential_energy_from_model tests passed")
end

# =============================================================================
# Summary
# =============================================================================

println()
println("=" ^ 60)
println("All MCMC tests completed!")
println("=" ^ 60)
