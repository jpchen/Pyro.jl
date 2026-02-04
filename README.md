# MiniPyro.jl

A minimal implementation of the [Pyro](http://pyro.ai) Probabilistic Programming Language in Julia.

MiniPyro.jl is a didactic implementation that demonstrates the core concepts of probabilistic programming using effect handlers. It provides a subset of Pyro's functionality sufficient for understanding and experimenting with variational inference.

## Features

- **Probabilistic Primitives**: `sample()` and `param()` for defining probabilistic models
- **Effect Handlers**: Composable handlers for tracing, replaying, and blocking
- **Variational Inference**: ELBO loss function and SVI optimizer
- **Automatic Differentiation**: Built on Zygote.jl for gradient computation
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
   - `Zygote.jl` - Automatic differentiation
   - `Optimisers.jl` - Optimization algorithms
   - `Random` - Random number generation (stdlib)
   - `Statistics` - Statistical functions (stdlib)

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

## Running Tests

```bash
# Run the full test suite
julia --project=. test/runtests.jl
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
| MCMC/HMC | ✓ | Separate |
| Neural Networks | PyTorch | Flux.jl |
| JIT compilation | ✓ | Julia native |

## Dependencies

- [Distributions.jl](https://github.com/JuliaStats/Distributions.jl) - Probability distributions
- [Zygote.jl](https://github.com/FluxML/Zygote.jl) - Automatic differentiation
- [Optimisers.jl](https://github.com/FluxML/Optimisers.jl) - Optimization algorithms

## References

- [Pyro Documentation](http://docs.pyro.ai)
- [Original minipyro.py](https://github.com/pyro-ppl/pyro/blob/dev/pyro/contrib/minipyro.py)
- [Effect Handlers Tutorial](http://pyro.ai/examples/effect_handlers.html)
- [SVI Tutorial](http://pyro.ai/examples/svi_part_i.html)

## License

Apache-2.0

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.
