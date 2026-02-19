# MiniPyro.jl

A minimal implementation of the [Pyro](http://pyro.ai) Probabilistic Programming Language in Julia.

MiniPyro.jl is a didactic implementation that demonstrates the core concepts of probabilistic programming using effect handlers. It provides a subset of Pyro's functionality sufficient for understanding and experimenting with variational inference.

## Features

- **Probabilistic Primitives**: `sample()` and `param()` for defining probabilistic models
- **Effect Handlers**: Composable handlers for tracing, replaying, and blocking
- **Variational Inference**: ELBO loss function and SVI optimizer
- **MCMC Inference**: HMC and NUTS (No-U-Turn Sampler) with automatic adaptation
- **Automatic Differentiation**: Built on Zygote.jl (SVI) and ForwardDiff.jl (MCMC)
- **Distribution Support**: Uses Distributions.jl for probability distributions

## Installation

### Prerequisites

- Julia 1.6 or later (tested with Julia 1.9+)

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/Pyro.jl.git
   cd Pyro.jl
   ```

2. **Activate and instantiate the project**:
   ```julia
   # Start Julia in the project directory
   julia --project=.

   # In the Julia REPL:
   using Pkg
   Pkg.instantiate()
   ```

   This will install all required dependencies:
   - `Distributions.jl` - Probability distributions
   - `Zygote.jl` - Automatic differentiation (SVI)
   - `ForwardDiff.jl` - Automatic differentiation (MCMC)
   - `Optimisers.jl` - Optimization algorithms
   - `Random` - Random number generation (stdlib)
   - `Statistics` - Statistical functions (stdlib)
   - `LinearAlgebra` - Linear algebra operations (stdlib)

### Quick Start

```julia
# Start Julia with the project
julia --project=.

# Include MiniPyro
include("src/MiniPyro.jl")
using .MiniPyro
using Distributions
using Random

# Set seed for reproducibility
Random.seed!(42)

# Define a simple model
function model(data)
    loc = sample("loc", Normal(0.0, 1.0))
    for i in 1:length(data)
        sample("obs_$i", Normal(loc, 1.0); obs=data[i])
    end
end

# Define a guide (variational distribution)
function guide(data)
    guide_loc = param("guide_loc", 0.0)
    guide_scale = exp(param("guide_scale_log", 0.0))
    sample("loc", Normal(guide_loc, guide_scale))
end

# Generate data
data = randn(100) .+ 3.0  # 100 samples from N(3, 1)

# Clear parameter store and run SVI
clear_param_store!()
svi = SVI(model, guide, Adam(0.02), elbo)

for step in 1:1000
    loss = svi_step!(svi, data)
    if step % 100 == 0
        println("Step $step: loss = $loss")
    end
end

# Check learned parameters
store = get_param_store()
println("Learned location: ", store["guide_loc"][1])  # Should be ≈ 3.0
```

## Running the Demo

The repository includes a demo script that replicates the minipyro.py example from Pyro:

```bash
# Run with default settings
julia --project=. examples/minipyro_demo.jl

# Run with custom settings
julia --project=. examples/minipyro_demo.jl --num-steps 2000 --learning-rate 0.01

# Show help
julia --project=. examples/minipyro_demo.jl --help
```

Expected output:
```
============================================================
MiniPyro.jl Demo
============================================================

Generated 100 data points
Sample mean: 2.9876
Sample std:  0.9823

Training Configuration:
  - Learning rate: 0.02
  - Num steps: 1001

Training...
----------------------------------------
Step    0: loss = 156.7234
Step  100: loss = 147.2341
Step  200: loss = 146.8912
...
Step 1000: loss = 146.7123
----------------------------------------

Final parameter values:
----------------------------------------
  guide_loc = 2.9876
  guide_scale_log = -2.3456 (unconstrained: -2.3456)

Validation:
----------------------------------------
  Expected guide_loc ≈ 3.0 (true mean of data)
  Actual guide_loc   = 2.9876
  Error              = 0.0124

✓ Success! Guide location is within 0.1 of true mean (3.0)
```

## MCMC Inference (HMC / NUTS)

MiniPyro includes a full MCMC inference stack with HMC (Hamiltonian Monte Carlo) and NUTS (No-U-Turn Sampler), modeled after Pyro's implementation.

### Quick Start: NUTS Sampling

```julia
include("src/MiniPyro.jl")
using .MiniPyro
using .MiniPyro.MCMC
using Random

Random.seed!(42)

# Step 1: Define your model as a potential energy function.
# The potential energy is the negative log joint density: U(z) = -log p(z, data)
# z is the vector of unconstrained parameters to infer.

# Example: Bayesian linear regression y = w*x + b + noise
x_data = randn(50)
y_data = 2.5 .* x_data .+ (-1.0) .+ 0.5 .* randn(50)

function potential_fn(z)
    w, b = z[1], z[2]
    # Prior: w ~ N(0, 1), b ~ N(0, 1)
    lp = -0.5 * w^2 - 0.5 * b^2
    # Likelihood: y_i ~ N(w*x_i + b, 0.5)
    for i in 1:length(x_data)
        residual = y_data[i] - (w * x_data[i] + b)
        lp += -0.5 * (residual / 0.5)^2 - log(0.5)
    end
    return -lp  # potential = negative log joint
end

# Step 2: Create a kernel (NUTS or HMC)
kernel = NUTSKernel(potential_fn)

# Step 3: Run MCMC
result = run_mcmc(kernel, 1000, [0.0, 0.0]; warmup_steps=500)

# Step 4: Inspect results
println("Posterior mean: ", mean(result))  # Should be ≈ [2.5, -1.0]
println("Posterior std:  ", std(result))

# Access raw samples (Matrix: num_samples x dim)
samples = result.samples
w_samples = samples[:, 1]
b_samples = samples[:, 2]
```

### Using HMC Instead of NUTS

```julia
# HMC requires specifying the number of leapfrog steps manually
kernel = HMCKernel(potential_fn; step_size=0.1, num_steps=10)
result = run_mcmc(kernel, 1000, [0.0, 0.0]; warmup_steps=500)
```

### Kernel Options

**NUTSKernel** (recommended for most problems):
```julia
NUTSKernel(
    potential_fn;
    step_size=1.0,              # Initial step size (auto-tuned)
    adapt_step_size=true,       # Adapt step size during warmup
    adapt_mass_matrix=true,     # Adapt mass matrix during warmup
    target_accept_prob=0.8,     # Target Metropolis acceptance rate
    max_tree_depth=10           # Max binary tree depth (up to 2^10 leapfrog steps)
)
```

**HMCKernel**:
```julia
HMCKernel(
    potential_fn;
    step_size=1.0,              # Leapfrog step size (auto-tuned)
    num_steps=10,               # Number of leapfrog steps per proposal
    adapt_step_size=true,       # Adapt step size during warmup
    adapt_mass_matrix=true,     # Adapt mass matrix during warmup
    target_accept_prob=0.8      # Target Metropolis acceptance rate
)
```

### Diagnostics

```julia
result = run_mcmc(kernel, 1000, initial_params; warmup_steps=500)

# Check diagnostics
diag = result.diagnostics
println("Step size: ", diag["step_size"])
println("Divergences: ", diag["divergences"])    # Should be 0
println("Mean tree depth: ", diag["mean_tree_depth"])  # NUTS only
println("Accept prob: ", diag["accept_prob"])     # NUTS
# or diag["accept_rate"] for HMC
```

### Running MCMC Tests Only

```bash
# Run MCMC-specific tests
julia --project=. test/test_mcmc.jl

# Run the full test suite (includes both SVI and MCMC tests)
julia --project=. test/runtests.jl
```

### MCMC Architecture

```
┌──────────────────────────────────────────────┐
│                  run_mcmc()                    │
│  Orchestrator: warmup → adapt → sample        │
└─────────────────┬────────────────────────────┘
                  │
    ┌─────────────┴─────────────┐
    │                           │
┌───┴──────────┐     ┌─────────┴──────┐
│  HMCKernel   │     │  NUTSKernel    │
│  Fixed L     │     │  Adaptive L    │
│  steps       │     │  (tree build)  │
└───┬──────────┘     └─────────┬──────┘
    │                           │
    └─────────────┬─────────────┘
                  │
    ┌─────────────┴─────────────┐
    │   velocity_verlet()       │
    │   Leapfrog integrator     │
    └─────────────┬─────────────┘
                  │
    ┌─────────────┴─────────────┐
    │  potential_energy_grad()   │
    │  ForwardDiff.jl autodiff   │
    └───────────────────────────┘

    ┌───────────────────────────┐
    │     WarmupAdapter         │
    │  ┌─────────────────────┐  │
    │  │   DualAveraging     │  │  ← Step size adaptation
    │  └─────────────────────┘  │
    │  ┌─────────────────────┐  │
    │  │ WelfordCovariance   │  │  ← Mass matrix estimation
    │  └─────────────────────┘  │
    └───────────────────────────┘
```

## Running Tests

```bash
# Run the full test suite (SVI + MCMC)
julia --project=. test/runtests.jl

# Run only the core SVI tests
julia --project=. test/runtests.jl

# Run only MCMC tests
julia --project=. test/test_mcmc.jl
```

Or from the Julia REPL:
```julia
using Pkg
Pkg.test()
```

## API Reference

### Core Primitives

#### `sample(name, dist; obs=nothing)`

Sample from a probability distribution or observe data.

```julia
# Sample from a prior
loc = sample("loc", Normal(0.0, 1.0))

# Observe data (condition on observations)
sample("obs", Normal(loc, 1.0); obs=data)
```

#### `param(name, init_value; constraint=identity)`

Register or retrieve a learnable parameter.

```julia
# Unconstrained parameter
loc = param("loc", 0.0)

# Positive parameter using exp constraint
scale = param("scale_log", 0.0; constraint=exp)
```

### Effect Handlers

#### `trace(fn)`

Execute a function and record all sample/param sites.

```julia
tr = trace() do
    x = sample("x", Normal(0, 1))
    y = sample("y", Normal(x, 1))
end
# tr["x"]["value"], tr["y"]["value"] contain sampled values
```

#### `replay(guide_trace, fn)`

Execute a function, substituting values from a previous trace.

```julia
guide_trace = get_trace(guide, data)
model_trace = trace() do
    replay(guide_trace) do
        model(data)
    end
end
```

#### `block(fn; hide_fn)`

Block certain sites from handlers below.

```julia
# Only trace parameters, not samples
param_trace = trace() do
    block(msg -> msg["type"] == "sample") do
        model(data)
    end
end
```

### Inference

#### `elbo(model, guide, args...)`

Compute the Evidence Lower Bound (returns negative ELBO for minimization).

```julia
loss = elbo(model, guide, data)
```

#### `SVI(model, guide, optimizer, loss)`

Stochastic Variational Inference optimizer.

```julia
svi = SVI(model, guide, Adam(0.02), elbo)
for step in 1:1000
    loss = svi_step!(svi, data)
end
```

### Parameter Store

```julia
# Get the parameter store
store = get_param_store()

# Clear all parameters (do this before training)
clear_param_store!()

# Access a parameter value
value, constraint = store["param_name"]
constrained_value = constraint(value)
```

### MCMC Inference

#### `NUTSKernel(potential_fn; kwargs...)`

Create a NUTS kernel for MCMC sampling.

```julia
using .MiniPyro.MCMC

potential_fn(z) = 0.5 * sum(z .^ 2)  # Standard normal target
kernel = NUTSKernel(potential_fn; max_tree_depth=10)
result = run_mcmc(kernel, 1000, zeros(2); warmup_steps=500)
```

#### `HMCKernel(potential_fn; kwargs...)`

Create an HMC kernel for MCMC sampling.

```julia
kernel = HMCKernel(potential_fn; step_size=0.1, num_steps=10)
result = run_mcmc(kernel, 1000, zeros(2); warmup_steps=500)
```

#### `run_mcmc(kernel, num_samples, initial_params; kwargs...)`

Run MCMC inference and return an `MCMCResult`.

```julia
result = run_mcmc(kernel, 1000, zeros(2); warmup_steps=500, progress=true)
samples = result.samples          # Matrix (num_samples x dim)
posterior_mean = mean(result)     # Vector of posterior means
posterior_std = std(result)       # Vector of posterior stds
```

#### `potential_energy_from_model(logpdf_fn)`

Convert a log-pdf function to a potential energy function.

```julia
logpdf_fn(z) = -0.5 * sum(z .^ 2)  # log N(0, I)
potential_fn = potential_energy_from_model(logpdf_fn)
# potential_fn(z) = -logpdf_fn(z) = 0.5 * sum(z .^ 2)
```

## Architecture

MiniPyro uses the **effect handler** pattern for implementing probabilistic programming primitives. This design allows:

1. **Composability**: Multiple handlers can be stacked
2. **Separation of Concerns**: Model definition is separate from inference
3. **Flexibility**: New inference algorithms can be added without changing models

### Handler Stack

```
┌─────────────────────────────────────┐
│            User Code                │
│   sample("x", Normal(0, 1))         │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│         PYRO_STACK                  │
│  ┌─────────────────────────────┐    │
│  │    Trace Handler            │    │  ← Records all sites
│  └─────────────────────────────┘    │
│  ┌─────────────────────────────┐    │
│  │    Replay Handler           │    │  ← Substitutes values
│  └─────────────────────────────┘    │
│  ┌─────────────────────────────┐    │
│  │    Block Handler            │    │  ← Filters sites
│  └─────────────────────────────┘    │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│      Distribution Sampling          │
│      rand(Normal(0, 1))             │
└─────────────────────────────────────┘
```

## Comparison with Python Pyro

| Feature | Python Pyro | MiniPyro.jl |
|---------|-------------|-------------|
| Core primitives | ✓ | ✓ |
| Effect handlers | ✓ | ✓ |
| SVI | ✓ | ✓ |
| ELBO | ✓ | ✓ |
| Plates | Partial | Basic |
| HMC | ✓ | ✓ |
| NUTS | ✓ | ✓ |
| Step size adaptation | ✓ | ✓ (Dual averaging) |
| Mass matrix adaptation | ✓ | ✓ (Welford diagonal) |
| Neural Networks | PyTorch | Flux.jl |
| JIT compilation | ✓ | Julia native |

## Dependencies

- [Distributions.jl](https://github.com/JuliaStats/Distributions.jl) - Probability distributions
- [Zygote.jl](https://github.com/FluxML/Zygote.jl) - Automatic differentiation (SVI)
- [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl) - Automatic differentiation (MCMC)
- [DiffResults.jl](https://github.com/JuliaDiff/DiffResults.jl) - Efficient gradient+value computation
- [Optimisers.jl](https://github.com/FluxML/Optimisers.jl) - Optimization algorithms
- [LinearAlgebra](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/) - Linear algebra (stdlib)

## References

- [Pyro Documentation](http://docs.pyro.ai)
- [Original minipyro.py](https://github.com/pyro-ppl/pyro/blob/dev/pyro/contrib/minipyro.py)
- [Effect Handlers Tutorial](http://pyro.ai/examples/effect_handlers.html)
- [SVI Tutorial](http://pyro.ai/examples/svi_part_i.html)
- Hoffman & Gelman (2014) "The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo"
- Betancourt (2017) "A Conceptual Introduction to Hamiltonian Monte Carlo"

## License

Apache-2.0

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.
