# Copyright (c) 2024
# SPDX-License-Identifier: Apache-2.0

"""
Test Suite for MiniPyro.jl

This file contains comprehensive tests for all components of the MiniPyro
probabilistic programming library.

Run tests with:
    julia --project=. test/runtests.jl

Or from the Julia REPL:
    using Pkg
    Pkg.test()
"""

using Test
using Random
using Statistics

# Add src to load path and import MiniPyro
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
include(joinpath(@__DIR__, "..", "src", "MiniPyro.jl"))
using .MiniPyro
using Distributions

println("=" ^ 60)
println("MiniPyro.jl Test Suite")
println("=" ^ 60)
println()

# =============================================================================
# Test: Parameter Store
# =============================================================================

@testset "Parameter Store" begin
    println("Testing Parameter Store...")

    # Clear store before testing
    clear_param_store!()
    @test isempty(get_param_store())

    # Test parameter creation
    x = param("test_param", 1.0)
    @test x == 1.0
    @test haskey(get_param_store(), "test_param")

    # Test parameter retrieval (should return same value)
    y = param("test_param")
    @test y == 1.0

    # Test parameter with constraint
    z = param("positive_param", 0.0; constraint=exp)
    @test z == 1.0  # exp(0) = 1

    # Test multiple parameters
    a = param("param_a", 2.0)
    b = param("param_b", 3.0)
    @test length(get_param_store()) == 4

    # Test clear
    clear_param_store!()
    @test isempty(get_param_store())

    println("  ✓ Parameter Store tests passed")
end

# =============================================================================
# Test: Sample Primitive
# =============================================================================

@testset "Sample Primitive" begin
    println("Testing Sample Primitive...")

    Random.seed!(42)

    # Test basic sampling (no handlers)
    x = sample("x", Normal(0.0, 1.0))
    @test isa(x, Float64)

    # Test observed sampling
    y = sample("y", Normal(0.0, 1.0); obs=5.0)
    @test y == 5.0

    # Test sampling from different distributions
    z_uniform = sample("z_uniform", Uniform(0.0, 1.0))
    @test 0.0 <= z_uniform <= 1.0

    z_bernoulli = sample("z_bernoulli", Bernoulli(0.7))
    @test z_bernoulli ∈ [0, 1]

    println("  ✓ Sample Primitive tests passed")
end

# =============================================================================
# Test: Trace Handler
# =============================================================================

@testset "Trace Handler" begin
    println("Testing Trace Handler...")

    clear_param_store!()
    Random.seed!(42)

    # Test tracing a simple function
    tr = trace() do
        x = sample("x", Normal(0.0, 1.0))
        y = sample("y", Normal(x, 1.0))
    end

    @test haskey(tr, "x")
    @test haskey(tr, "y")
    @test tr["x"]["type"] == "sample"
    @test tr["y"]["type"] == "sample"
    @test isa(tr["x"]["value"], Float64)
    @test isa(tr["y"]["value"], Float64)

    # Test that trace captures distribution
    @test haskey(tr["x"], "dist")
    @test isa(tr["x"]["dist"], Normal)

    # Test get_trace helper
    function test_model(data)
        loc = sample("loc", Normal(0.0, 1.0))
        sample("obs", Normal(loc, 1.0); obs=data)
    end

    tr2 = get_trace(test_model, 3.0)
    @test haskey(tr2, "loc")
    @test haskey(tr2, "obs")
    @test tr2["obs"]["value"] == 3.0  # Observed value

    println("  ✓ Trace Handler tests passed")
end

# =============================================================================
# Test: Replay Handler
# =============================================================================

@testset "Replay Handler" begin
    println("Testing Replay Handler...")

    Random.seed!(42)

    # First, get a trace
    original_trace = trace() do
        sample("x", Normal(0.0, 1.0))
        sample("y", Normal(0.0, 1.0))
    end

    original_x = original_trace["x"]["value"]
    original_y = original_trace["y"]["value"]

    # Now replay - should get same values
    replayed_trace = trace() do
        replay(original_trace) do
            sample("x", Normal(100.0, 1.0))  # Different dist, should still use trace value
            sample("y", Normal(100.0, 1.0))
        end
    end

    @test replayed_trace["x"]["value"] == original_x
    @test replayed_trace["y"]["value"] == original_y

    println("  ✓ Replay Handler tests passed")
end

# =============================================================================
# Test: Block Handler
# =============================================================================

@testset "Block Handler" begin
    println("Testing Block Handler...")

    clear_param_store!()
    Random.seed!(42)

    # Test blocking sample sites, only capturing params
    function model_with_params(data)
        loc = param("loc", 0.0)
        x = sample("x", Normal(loc, 1.0))
        return x
    end

    # Block samples, trace only params
    param_trace = trace() do
        block(msg -> msg["type"] == "sample") do
            model_with_params(nothing)
        end
    end

    @test haskey(param_trace, "loc")
    @test !haskey(param_trace, "x")  # Should be blocked

    println("  ✓ Block Handler tests passed")
end

# =============================================================================
# Test: ELBO Computation
# =============================================================================

@testset "ELBO Computation" begin
    println("Testing ELBO Computation...")

    clear_param_store!()
    Random.seed!(42)

    # Simple model and guide
    function simple_model(data)
        loc = sample("loc", Normal(0.0, 1.0))
        for i in 1:length(data)
            sample("obs_$i", Normal(loc, 1.0); obs=data[i])
        end
    end

    function simple_guide(data)
        guide_loc = param("guide_loc", 0.0)
        sample("loc", Normal(guide_loc, 1.0))
    end

    data = [1.0, 2.0, 3.0]

    # ELBO should be a finite number
    loss = elbo(simple_model, simple_guide, data)
    @test isfinite(loss)
    @test isa(loss, Float64)

    # ELBO should change when we change parameters
    clear_param_store!()
    loss1 = elbo(simple_model, simple_guide, data)

    # Manually change the parameter
    store = get_param_store()
    store["guide_loc"] = (5.0, identity)  # Move guide far from data

    loss2 = elbo(simple_model, simple_guide, data)

    # The losses should be different
    @test loss1 != loss2

    println("  ✓ ELBO Computation tests passed")
end

# =============================================================================
# Test: Adam Optimizer
# =============================================================================

@testset "Adam Optimizer" begin
    println("Testing Adam Optimizer...")

    # Test optimizer creation
    opt = Adam(0.01)
    @test opt.lr == 0.01
    @test opt.beta == (0.9, 0.999)

    # Test with custom beta
    opt2 = Adam(0.001; beta=(0.8, 0.99))
    @test opt2.lr == 0.001
    @test opt2.beta == (0.8, 0.99)

    # Test parameter update
    clear_param_store!()
    param("test", 1.0)

    # Manually trigger an update
    param_grads = Dict("test" => 0.1)
    step!(opt, param_grads)

    # Parameter should have changed
    new_val, _ = get_param_store()["test"]
    @test new_val != 1.0

    println("  ✓ Adam Optimizer tests passed")
end

# =============================================================================
# Test: SVI Training
# =============================================================================

@testset "SVI Training" begin
    println("Testing SVI Training...")

    clear_param_store!()
    Random.seed!(42)

    # Define model and guide
    function model(data)
        loc = sample("loc", Normal(0.0, 1.0))
        for i in 1:length(data)
            sample("obs_$i", Normal(loc, 1.0); obs=data[i])
        end
    end

    function guide(data)
        guide_loc = param("guide_loc", 0.0)
        guide_scale = exp(param("guide_scale_log", 0.0))
        sample("loc", Normal(guide_loc, guide_scale))
    end

    # Generate data centered at 3.0
    data = randn(50) .+ 3.0

    # Create SVI
    svi = SVI(model, guide, Adam(0.05), elbo)

    # Run a few training steps
    initial_loss = svi_step!(svi, data)
    @test isfinite(initial_loss)

    # Run more steps
    losses = Float64[]
    for _ in 1:100
        loss = svi_step!(svi, data)
        push!(losses, loss)
    end

    # Loss should generally decrease (not strictly, due to stochasticity)
    @test mean(losses[1:10]) > mean(losses[end-10:end])

    # Guide location should move toward data mean (3.0)
    guide_loc = get_param_store()["guide_loc"][1]
    @test abs(guide_loc - 3.0) < 1.0  # Within 1.0 of true mean

    println("  ✓ SVI Training tests passed")
end

# =============================================================================
# Test: Full Integration Test
# =============================================================================

@testset "Full Integration Test" begin
    println("Testing Full Integration...")

    clear_param_store!()
    Random.seed!(0)

    # This replicates the minipyro.py example
    function model(data)
        loc = sample("loc", Normal(0.0, 1.0))
        for i in 1:length(data)
            sample("obs_$i", Normal(loc, 1.0); obs=data[i])
        end
    end

    function guide(data)
        guide_loc = param("guide_loc", 0.0)
        guide_scale = exp(param("guide_scale_log", 0.0))
        sample("loc", Normal(guide_loc, guide_scale))
    end

    # Generate data
    data = randn(100) .+ 3.0

    # Train
    svi = SVI(model, guide, Adam(0.02), elbo)

    for step in 1:500
        svi_step!(svi, data)
    end

    # Check results
    store = get_param_store()
    guide_loc = store["guide_loc"][1]
    guide_scale = exp(store["guide_scale_log"][1])

    # Guide location should be close to 3.0
    @test abs(guide_loc - 3.0) < 0.2

    # Guide scale should be small (posterior is concentrated)
    # Theoretical posterior std ≈ 1/sqrt(101) ≈ 0.0995
    @test guide_scale < 0.5

    println("  ✓ Full Integration tests passed")
end

# =============================================================================
# Test: Plate Handler
# =============================================================================

@testset "Plate Handler" begin
    println("Testing Plate Handler...")

    # Test plate as iterator
    p = plate("test", 5)
    collected = collect(p)
    @test collected == [1, 2, 3, 4, 5]
    @test length(p) == 5

    # Test with_plate context
    Random.seed!(42)
    values = Float64[]

    with_plate("data", 3, -1) do
        x = sample("x", Normal(0.0, 1.0))
        push!(values, x)
    end

    @test length(values) == 1  # Only one sample in vectorized case

    println("  ✓ Plate Handler tests passed")
end

# =============================================================================
# Test: Seed Handler
# =============================================================================

@testset "Seed Handler" begin
    println("Testing Seed Handler...")

    # Same seed should give same results
    x1 = seed(42) do
        sample("x", Normal(0.0, 1.0))
    end

    x2 = seed(42) do
        sample("x", Normal(0.0, 1.0))
    end

    @test x1 == x2

    # Different seeds should (almost certainly) give different results
    x3 = seed(123) do
        sample("x", Normal(0.0, 1.0))
    end

    @test x1 != x3

    println("  ✓ Seed Handler tests passed")
end

# =============================================================================
# Test: Multiple Distributions
# =============================================================================

@testset "Multiple Distributions" begin
    println("Testing Multiple Distributions...")

    clear_param_store!()
    Random.seed!(42)

    # Test with various distributions
    function multi_dist_model()
        # Continuous distributions
        x_normal = sample("x_normal", Normal(0.0, 1.0))
        x_uniform = sample("x_uniform", Uniform(0.0, 1.0))
        x_exponential = sample("x_exponential", Exponential(1.0))

        # Discrete distributions
        x_bernoulli = sample("x_bernoulli", Bernoulli(0.5))
        x_poisson = sample("x_poisson", Poisson(5.0))

        return (x_normal, x_uniform, x_exponential, x_bernoulli, x_poisson)
    end

    tr = get_trace(multi_dist_model)

    @test haskey(tr, "x_normal")
    @test haskey(tr, "x_uniform")
    @test haskey(tr, "x_exponential")
    @test haskey(tr, "x_bernoulli")
    @test haskey(tr, "x_poisson")

    # Check value constraints
    @test 0.0 <= tr["x_uniform"]["value"] <= 1.0
    @test tr["x_exponential"]["value"] >= 0.0
    @test tr["x_bernoulli"]["value"] ∈ [0, 1]
    @test tr["x_poisson"]["value"] >= 0

    println("  ✓ Multiple Distributions tests passed")
end

# =============================================================================
# Summary
# =============================================================================

println()
println("=" ^ 60)
println("All tests completed!")
println("=" ^ 60)
