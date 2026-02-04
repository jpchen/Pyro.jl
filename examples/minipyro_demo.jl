# Copyright (c) 2024
# SPDX-License-Identifier: Apache-2.0

"""
MiniPyro Demo Script

This example demonstrates the functionality of MiniPyro.jl, which is a minimal
implementation of the Pyro Probabilistic Programming Language in Julia.

This script replicates the example from:
https://github.com/pyro-ppl/pyro/blob/dev/examples/minipyro.py

The example:
1. Defines a simple Bayesian model with a Normal prior and Normal likelihood
2. Defines a variational guide (approximate posterior)
3. Uses SVI to fit the guide to the true posterior
4. Demonstrates that the learned parameters converge to the correct values

Usage:
    julia --project=. examples/minipyro_demo.jl
    julia --project=. examples/minipyro_demo.jl --num-steps 2000 --lr 0.01
"""

# Add the src directory to the load path
push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

using Random
using Statistics
using Printf

# Import MiniPyro
include(joinpath(@__DIR__, "..", "src", "MiniPyro.jl"))
using .MiniPyro
using Distributions

# =============================================================================
# Configuration
# =============================================================================

"""
Configuration for the training run.
"""
Base.@kwdef struct Config
    num_steps::Int = 1001
    learning_rate::Float64 = 0.02
    seed::Int = 0
end

function parse_args()
    config = Config()

    i = 1
    while i <= length(ARGS)
        arg = ARGS[i]
        if arg == "-n" || arg == "--num-steps"
            config = Config(
                num_steps = parse(Int, ARGS[i+1]),
                learning_rate = config.learning_rate,
                seed = config.seed
            )
            i += 2
        elseif arg == "-lr" || arg == "--learning-rate"
            config = Config(
                num_steps = config.num_steps,
                learning_rate = parse(Float64, ARGS[i+1]),
                seed = config.seed
            )
            i += 2
        elseif arg == "--seed"
            config = Config(
                num_steps = config.num_steps,
                learning_rate = config.learning_rate,
                seed = parse(Int, ARGS[i+1])
            )
            i += 2
        elseif arg == "-h" || arg == "--help"
            println("""
            MiniPyro Demo

            Usage: julia examples/minipyro_demo.jl [options]

            Options:
              -n, --num-steps N      Number of training steps (default: 1001)
              -lr, --learning-rate R Learning rate (default: 0.02)
              --seed S               Random seed (default: 0)
              -h, --help             Show this help message
            """)
            exit(0)
        else
            i += 1
        end
    end

    return config
end

# =============================================================================
# Model and Guide Definitions
# =============================================================================

"""
    model(data)

A basic Bayesian model with:
- A single Normal latent random variable `loc` with prior N(0, 1)
- A batch of Normally distributed observations centered at `loc`

Generative process:
    loc ~ Normal(0, 1)
    for each observation i:
        obs[i] ~ Normal(loc, 1)

This is a simple model where the true posterior is analytically tractable:
    p(loc | data) ∝ N(loc; μ_posterior, σ_posterior)
where the posterior mean is approximately the sample mean for large N.
"""
function model(data::Vector{Float64})
    # Prior: loc ~ Normal(0, 1)
    loc = sample("loc", Normal(0.0, 1.0))

    # Likelihood: each observation ~ Normal(loc, 1)
    # Using a plate for efficient batched computation
    for i in 1:length(data)
        sample("obs_$i", Normal(loc, 1.0); obs=data[i])
    end

    return loc
end

"""
    guide(data)

Variational guide (approximate posterior) for the model.

The guide defines a Normal distribution over the latent variable `loc`
with learnable parameters:
- guide_loc: mean of the variational distribution
- guide_scale: standard deviation (parameterized as exp(guide_scale_log))

The goal of variational inference is to optimize these parameters so that
the guide approximates the true posterior p(loc | data).
"""
function guide(data::Vector{Float64})
    # Learnable parameters
    guide_loc = param("guide_loc", 0.0)
    # Use exp transform to ensure scale is positive
    guide_scale = exp(param("guide_scale_log", 0.0))

    # Sample from the variational distribution
    loc = sample("loc", Normal(guide_loc, guide_scale))

    return loc
end

# =============================================================================
# Main Training Loop
# =============================================================================

"""
    main(config::Config)

Run the MiniPyro demo with the given configuration.

This function:
1. Generates synthetic data from a Normal distribution centered at 3.0
2. Sets up SVI with the model, guide, Adam optimizer, and ELBO loss
3. Trains the guide parameters to approximate the posterior
4. Reports the final learned parameter values
5. Validates that the guide mean is close to 3.0
"""
function main(config::Config)
    println("=" ^ 60)
    println("MiniPyro.jl Demo")
    println("=" ^ 60)
    println()

    # Set random seed for reproducibility
    Random.seed!(config.seed)

    # Generate synthetic data: 100 samples from Normal(3.0, 1.0)
    # The true latent location is 3.0, so our guide should learn this
    data = randn(100) .+ 3.0

    println("Generated $(length(data)) data points")
    println("Sample mean: $(@sprintf("%.4f", mean(data)))")
    println("Sample std:  $(@sprintf("%.4f", std(data)))")
    println()

    # Clear any existing parameters
    clear_param_store!()

    # Create SVI optimizer
    # - model: our probabilistic model
    # - guide: our variational approximation
    # - Adam: optimizer with specified learning rate
    # - elbo: Evidence Lower Bound loss function
    adam = Adam(config.learning_rate)
    svi = SVI(model, guide, adam, elbo)

    println("Training Configuration:")
    println("  - Learning rate: $(config.learning_rate)")
    println("  - Num steps: $(config.num_steps)")
    println()

    # Training loop
    println("Training...")
    println("-" ^ 40)

    for step in 1:config.num_steps
        # Take one SVI step (forward + backward + optimizer step)
        loss = svi_step!(svi, data)

        # Print progress every 100 steps
        if step == 1 || (step - 1) % 100 == 0
            @printf("Step %4d: loss = %.4f\n", step - 1, loss)
        end
    end

    println("-" ^ 40)
    println()

    # Report final parameter values
    println("Final parameter values:")
    println("-" ^ 40)

    store = get_param_store()
    for (name, (unconstrained_value, constraint)) in store
        constrained_value = constraint(unconstrained_value)
        @printf("  %s = %.4f", name, constrained_value)
        if constraint !== identity
            @printf(" (unconstrained: %.4f)", unconstrained_value)
        end
        println()
    end

    println()

    # Validate results
    guide_loc = store["guide_loc"][1]  # Already in constrained space (identity)
    guide_scale = exp(store["guide_scale_log"][1])

    println("Validation:")
    println("-" ^ 40)
    println("  Expected guide_loc ≈ 3.0 (true mean of data)")
    @printf("  Actual guide_loc   = %.4f\n", guide_loc)
    @printf("  Error              = %.4f\n", abs(guide_loc - 3.0))
    println()

    # The posterior standard deviation for this conjugate model is:
    # σ_posterior = 1 / sqrt(1/σ_prior² + n/σ_likelihood²) = 1/sqrt(1 + 100) ≈ 0.0995
    expected_scale = 1.0 / sqrt(1.0 + length(data))
    println("  Expected guide_scale ≈ $(@sprintf("%.4f", expected_scale)) (posterior std)")
    @printf("  Actual guide_scale   = %.4f\n", guide_scale)
    println()

    # Check that we've learned approximately the right value
    if abs(guide_loc - 3.0) < 0.1
        println("✓ Success! Guide location is within 0.1 of true mean (3.0)")
    else
        println("✗ Warning: Guide location is not close enough to true mean")
        println("  This might indicate a bug or insufficient training")
    end

    println()
    println("=" ^ 60)
    println("Demo completed successfully!")
    println("=" ^ 60)
end

# =============================================================================
# Entry Point
# =============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    config = parse_args()
    main(config)
end
