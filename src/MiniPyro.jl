# Copyright (c) 2024
# SPDX-License-Identifier: Apache-2.0

"""
    MiniPyro

A minimal implementation of the Pyro Probabilistic Programming Language in Julia.

This module provides a minimal but functional implementation of Pyro's core
probabilistic programming primitives. It is designed for educational purposes
and demonstrates how effect handlers can be used to implement probabilistic
programming concepts.

# Key Concepts

- **Effect Handlers (Messengers)**: Non-standard interpretation of probabilistic
  primitives like `sample()` and `param()`. Handlers can modify how these
  primitives behave (e.g., trace execution, replay values, block sites).

- **Parameter Store**: Global storage for learnable parameters that persist
  across function calls during optimization.

- **ELBO (Evidence Lower Bound)**: The fundamental objective in variational
  inference, used to optimize the guide to approximate the posterior.

# Main Components

- `sample(name, dist; obs=nothing)`: Sample from a distribution or observe data
- `param(name, init_value; constraint=identity)`: Register a learnable parameter
- `plate(name, size; dim=-1)`: Declare a batch dimension for vectorized sampling
- `Trace`: Effect handler that records execution traces
- `Replay`: Effect handler that replays values from a trace
- `Block`: Effect handler that blocks certain sites
- `SVI`: Stochastic Variational Inference optimizer
- `elbo`: Evidence Lower Bound loss function

# Example Usage

```julia
using MiniPyro
using Distributions

# Define a model
function model(data)
    loc = sample("loc", Normal(0.0, 1.0))
    for i in 1:length(data)
        sample("obs_\$i", Normal(loc, 1.0); obs=data[i])
    end
end

# Define a guide (variational distribution)
function guide(data)
    guide_loc = param("guide_loc", 0.0)
    guide_scale = exp(param("guide_scale_log", 0.0))
    sample("loc", Normal(guide_loc, guide_scale))
end

# Run variational inference
svi = SVI(model, guide, Adam(0.02), elbo)
for step in 1:1000
    loss = svi_step!(svi, data)
end
```

See also: [`sample`](@ref), [`param`](@ref), [`SVI`](@ref), [`elbo`](@ref)
"""
module MiniPyro

using Distributions
using Optimisers
using Random
using Statistics
using Zygote

export sample, param, plate, get_param_store, clear_param_store!
export Messenger, Trace, Replay, Block, Seed
export trace, replay, block, seed
export SVI, svi_step!, elbo, Trace_ELBO
export Adam

# =============================================================================
# Global State
# =============================================================================

"""
    PYRO_STACK

Global stack of effect handlers (Messengers). Handlers are applied in reverse
order (handlers later in the stack are applied first). This enables composition
of multiple effects like tracing and replaying.

Effect handlers push themselves onto this stack when entering their scope and
pop themselves when exiting.
"""
const PYRO_STACK = Vector{Any}()

"""
    PARAM_STORE

Global parameter store that maps parameter names to tuples of
(unconstrained_value, constraint). Parameters are stored in unconstrained space
for optimization, then transformed to constrained space when accessed.

Use `get_param_store()` to access and `clear_param_store!()` to reset.
"""
const PARAM_STORE = Dict{String, Tuple{Any, Function}}()

"""
    get_param_store()

Return a reference to the global parameter store.

The parameter store is a dictionary mapping parameter names to tuples of
(unconstrained_value, constraint_function).

# Returns
- `Dict{String, Tuple{Any, Function}}`: The global parameter store

# Example
```julia
store = get_param_store()
for (name, (value, constraint)) in store
    println("\$name = \$(constraint(value))")
end
```
"""
function get_param_store()
    return PARAM_STORE
end

"""
    clear_param_store!()

Clear all parameters from the global parameter store.

This should be called before starting a new training run to ensure
parameters are reinitialized.

# Example
```julia
clear_param_store!()
# Now PARAM_STORE is empty and ready for new parameters
```
"""
function clear_param_store!()
    empty!(PARAM_STORE)
    return nothing
end

# =============================================================================
# Messenger Base Type and Interface
# =============================================================================

"""
    Messenger

Abstract base type for effect handlers in MiniPyro.

Effect handlers (called Messengers for consistency with Pyro) enable non-standard
interpretations of probabilistic primitives. They work by intercepting messages
created by `sample()` and `param()` calls and modifying their behavior.

# Interface

Subtypes should implement:
- `process_message(handler, msg)`: Called before the primitive executes
- `postprocess_message(handler, msg)`: Called after the primitive executes

Handlers push themselves onto `PYRO_STACK` when entering scope and pop when exiting.

# Example

```julia
struct MyHandler <: Messenger
    fn::Union{Function, Nothing}
end

function process_message(h::MyHandler, msg)
    println("Processing: \$(msg["name"])")
end
```

See also: [`Trace`](@ref), [`Replay`](@ref), [`Block`](@ref)
"""
abstract type Messenger end

"""
    process_message(handler::Messenger, msg::Dict)

Process a message before the primitive executes.

This method is called for each handler on the stack (in reverse order) before
a `sample` or `param` call executes. Handlers can modify the message to change
behavior (e.g., set values, mark sites to stop processing).

# Arguments
- `handler`: The messenger/handler processing the message
- `msg`: Dictionary containing message data (type, name, fn, args, value, etc.)

Default implementation does nothing.
"""
function process_message(handler::Messenger, msg::Dict)
    return nothing
end

"""
    postprocess_message(handler::Messenger, msg::Dict)

Post-process a message after the primitive executes.

This method is called for each handler (in forward order, opposite of process)
after the primitive has executed and produced a value. Useful for recording
results (e.g., in Trace).

# Arguments
- `handler`: The messenger/handler processing the message
- `msg`: Dictionary containing message data including the computed value

Default implementation does nothing.
"""
function postprocess_message(handler::Messenger, msg::Dict)
    return nothing
end

# =============================================================================
# Trace Handler
# =============================================================================

"""
    Trace <: Messenger

Effect handler that records the inputs and outputs of all primitive sites.

Trace is fundamental to probabilistic programming - it captures the full
execution trace including all sampled values and their distributions, which
is necessary for computing probabilities and gradients.

# Fields
- `fn::Union{Function, Nothing}`: The function to trace (optional)
- `trace::Dict{String, Dict}`: Dictionary mapping site names to site info

# Usage

```julia
# Method 1: Using trace() function with do-block
tr = trace() do
    sample("x", Normal(0, 1))
    sample("y", Normal(0, 1))
end
# tr now contains info about sites "x" and "y"

# Method 2: Using get_trace for model functions
tr = get_trace(model, data)
```

# Trace Contents

Each site in the trace contains:
- `"type"`: Either "sample" or "param"
- `"name"`: The site name
- `"fn"`: The distribution (for samples) or parameter function
- `"value"`: The sampled/computed value
- `"args"`: Additional arguments
- `"kwargs"`: Keyword arguments including `obs` for observations

See also: [`trace`](@ref), [`get_trace`](@ref)
"""
mutable struct Trace <: Messenger
    fn::Union{Function, Nothing}
    trace::Dict{String, Dict}

    Trace(fn::Union{Function, Nothing}=nothing) = new(fn, Dict{String, Dict}())
end

"""
    enter!(handler::Trace)

Push the trace handler onto the effect stack and reset its trace dictionary.
Returns the trace dictionary for capturing results.
"""
function enter!(handler::Trace)
    push!(PYRO_STACK, handler)
    empty!(handler.trace)
    return handler.trace
end

"""
    exit!(handler::Trace)

Pop the trace handler from the effect stack.
Ensures the handler being removed is the most recently added one.
"""
function exit!(handler::Trace)
    @assert !isempty(PYRO_STACK) && PYRO_STACK[end] === handler
    pop!(PYRO_STACK)
    return nothing
end

function postprocess_message(handler::Trace, msg::Dict)
    # Only record sample sites, and ensure names are unique
    if msg["type"] == "sample"
        @assert !haskey(handler.trace, msg["name"]) "Sample sites must have unique names: $(msg["name"])"
    end
    handler.trace[msg["name"]] = copy(msg)
    return nothing
end

"""
    trace(fn::Function)

Execute a function while tracing all probabilistic primitive calls.

This is a convenience function that wraps the function execution in a Trace
handler and returns the resulting trace dictionary.

# Arguments
- `fn`: A zero-argument function to execute (use closures to pass arguments)

# Returns
- `Dict{String, Dict}`: Trace dictionary mapping site names to site info

# Example
```julia
tr = trace() do
    x = sample("x", Normal(0, 1))
    y = sample("y", Normal(x, 1))
end
println(tr["x"]["value"])  # The sampled value of x
println(tr["y"]["fn"])     # Normal(x, 1)
```

See also: [`Trace`](@ref), [`get_trace`](@ref)
"""
function trace(fn::Function)
    handler = Trace(fn)
    tr = enter!(handler)
    try
        fn()
    finally
        exit!(handler)
    end
    return tr
end

"""
    get_trace(fn::Function, args...; kwargs...)

Execute a function with arguments and return its execution trace.

This is a convenience method that wraps the function call in a trace handler
and returns the recorded trace.

# Arguments
- `fn`: The function to trace (typically a model or guide)
- `args...`: Positional arguments to pass to the function
- `kwargs...`: Keyword arguments to pass to the function

# Returns
- `Dict{String, Dict}`: The execution trace

# Example
```julia
function model(data)
    loc = sample("loc", Normal(0, 1))
    sample("obs", Normal(loc, 1); obs=data)
end

tr = get_trace(model, 3.0)
println(tr["loc"]["value"])
```
"""
function get_trace(fn::Function, args...; kwargs...)
    return trace(() -> fn(args...; kwargs...))
end

# =============================================================================
# Replay Handler
# =============================================================================

"""
    Replay <: Messenger

Effect handler that replays values from a previous trace.

When a `sample` site is encountered that exists in the guide trace, the
Replay handler substitutes the previously recorded value instead of sampling
a new one. This is essential for computing joint probabilities in the ELBO.

# Fields
- `fn::Union{Function, Nothing}`: The function to replay (optional)
- `guide_trace::Dict{String, Dict}`: Trace to replay values from

# Usage

```julia
# First, get a trace from the guide
guide_trace = get_trace(guide, data)

# Then, run the model while replaying guide values
model_trace = replay(guide_trace) do
    model(data)
end
```

This allows computing `log p(z, x)` where `z` comes from the guide.

See also: [`replay`](@ref), [`Trace`](@ref)
"""
mutable struct Replay <: Messenger
    fn::Union{Function, Nothing}
    guide_trace::Dict{String, Dict}

    Replay(guide_trace::Dict, fn::Union{Function, Nothing}=nothing) = new(fn, guide_trace)
end

function process_message(handler::Replay, msg::Dict)
    if haskey(handler.guide_trace, msg["name"])
        msg["value"] = handler.guide_trace[msg["name"]]["value"]
    end
    return nothing
end

"""
    replay(guide_trace::Dict, fn::Function)
    replay(fn::Function, guide_trace::Dict)

Execute a function while replaying values from a previous trace.

At each sample site, if the site name exists in the guide trace, the previously
recorded value is used instead of sampling a new one.

# Arguments
- `guide_trace`: A trace dictionary from a previous execution
- `fn`: The function to execute with replayed values

# Returns
- The return value of the function

# Example
```julia
guide_trace = get_trace(guide, data)

# Values from guide_trace will be used where site names match
result = replay(guide_trace) do
    model(data)
end
```
"""
function replay(guide_trace::Dict, fn::Function)
    handler = Replay(guide_trace, fn)
    push!(PYRO_STACK, handler)
    try
        return fn()
    finally
        @assert PYRO_STACK[end] === handler
        pop!(PYRO_STACK)
    end
end

replay(fn::Function, guide_trace::Dict) = replay(guide_trace, fn)

"""
    replay(guide_trace::Dict)

Create a replay handler that can be used with do-block syntax.

# Example
```julia
result = replay(guide_trace) do
    model(data)
end
```
"""
function replay(guide_trace::Dict)
    return fn -> replay(guide_trace, fn)
end

# =============================================================================
# Block Handler
# =============================================================================

"""
    Block <: Messenger

Effect handler that blocks certain sites from handlers below it on the stack.

Block enables selective application of effect handlers. Sites that match the
`hide_fn` predicate will only have handlers above Block applied, not those below.

# Fields
- `fn::Union{Function, Nothing}`: The function to wrap (optional)
- `hide_fn::Function`: Predicate function that returns true for sites to hide

# Usage

```julia
# Block all sample sites, allowing only param sites through
block(msg -> msg["type"] == "sample") do
    # Only param sites will be processed by handlers below block
    model(data)
end
```

This is used in SVI to capture parameters while ignoring sample sites.

See also: [`block`](@ref)
"""
mutable struct Block <: Messenger
    fn::Union{Function, Nothing}
    hide_fn::Function

    Block(hide_fn::Function=msg -> true, fn::Union{Function, Nothing}=nothing) = new(fn, hide_fn)
end

function process_message(handler::Block, msg::Dict)
    if handler.hide_fn(msg)
        msg["stop"] = true
    end
    return nothing
end

"""
    block(fn::Function; hide_fn=msg -> true)
    block(hide_fn::Function, fn::Function)

Execute a function while blocking certain sites from lower handlers.

Sites matching the `hide_fn` predicate will have their "stop" flag set,
preventing handlers below Block from processing them.

# Arguments
- `fn`: The function to execute
- `hide_fn`: Predicate returning true for sites to block (default: block all)

# Example
```julia
# Block sample sites, only trace params
param_trace = trace() do
    block(msg -> msg["type"] == "sample") do
        model(data)
    end
end
# param_trace only contains param sites
```
"""
function block(fn::Function; hide_fn::Function=msg -> true)
    handler = Block(hide_fn, fn)
    push!(PYRO_STACK, handler)
    try
        return fn()
    finally
        @assert PYRO_STACK[end] === handler
        pop!(PYRO_STACK)
    end
end

function block(hide_fn::Function, fn::Function)
    return block(fn; hide_fn=hide_fn)
end

# =============================================================================
# Seed Handler
# =============================================================================

"""
    Seed <: Messenger

Effect handler that sets the random seed for reproducible execution.

# Fields
- `fn::Union{Function, Nothing}`: The function to wrap
- `rng_seed::Int`: The random seed to use

# Usage
```julia
seed(42) do
    sample("x", Normal(0, 1))  # Always produces same value
end
```
"""
mutable struct Seed <: Messenger
    fn::Union{Function, Nothing}
    rng_seed::Int
    old_state::Union{Nothing, Random.AbstractRNG}

    Seed(rng_seed::Int, fn::Union{Function, Nothing}=nothing) = new(fn, rng_seed, nothing)
end

"""
    seed(rng_seed::Int, fn::Function)
    seed(rng_seed::Int)

Execute a function with a fixed random seed.

# Arguments
- `rng_seed`: The random seed to use
- `fn`: The function to execute

# Example
```julia
x = seed(42) do
    sample("x", Normal(0, 1))
end
# x will be the same every time this code runs
```
"""
function seed(rng_seed::Int, fn::Function)
    old_state = copy(Random.default_rng())
    Random.seed!(rng_seed)
    try
        return fn()
    finally
        copy!(Random.default_rng(), old_state)
    end
end

seed(rng_seed::Int) = fn -> seed(rng_seed, fn)

# =============================================================================
# Plate (Batch Dimension Handler)
# =============================================================================

"""
    PlateMessenger <: Messenger

Effect handler for declaring batch dimensions in vectorized models.

Plate allows declaring that a set of random variables are conditionally
independent given their parents. This enables efficient vectorized computation
by expanding distributions along the plate dimension.

# Fields
- `name::String`: Name of the plate (for debugging)
- `size::Int`: Number of elements in the plate
- `dim::Int`: The dimension to expand along (must be negative, counting from right)

# Usage

```julia
# Declare 100 i.i.d. observations
with_plate("data", 100, -1) do
    sample("obs", Normal(loc, 1.0); obs=data)
end
```

# Note

This implementation only supports broadcasting/expansion, not sequential iteration.
The `dim` parameter must be negative (PyTorch convention for batch dimensions).

See also: [`plate`](@ref)
"""
mutable struct PlateMessenger <: Messenger
    name::String
    size::Int
    dim::Int

    function PlateMessenger(name::String, size::Int, dim::Int)
        @assert dim < 0 "Plate dim must be negative (counting from right)"
        new(name, size, dim)
    end
end

function process_message(handler::PlateMessenger, msg::Dict)
    if msg["type"] == "sample"
        dist = msg["fn"]
        batch_shape = size(dist)

        # Calculate the target shape with the plate dimension
        ndims_needed = -handler.dim
        current_ndims = length(batch_shape)

        if current_ndims < ndims_needed || batch_shape[end + handler.dim + 1] != handler.size
            # Need to expand the distribution
            # This is a simplified version - in practice you'd need proper broadcasting
            # For now, we just mark that this site is inside a plate
            msg["plate_size"] = handler.size
            msg["plate_dim"] = handler.dim
        end
    end
    return nothing
end

# Make PlateMessenger iterable for sequential plate usage
Base.iterate(p::PlateMessenger, state=1) = state > p.size ? nothing : (state, state + 1)
Base.length(p::PlateMessenger) = p.size
Base.eltype(::Type{PlateMessenger}) = Int

"""
    plate(name::String, size::Int; dim::Int=-1)

Create a plate context for declaring conditionally independent random variables.

# Arguments
- `name`: Name of the plate (for debugging and tracing)
- `size`: Number of elements in the plate
- `dim`: Batch dimension (must be negative, default: -1)

# Returns
- `PlateMessenger`: A plate handler that can be used as a context manager

# Example
```julia
# Using with do-block
with_plate("data", length(data), -1) do
    sample("obs", Normal(loc, scale); obs=data)
end

# Using as an iterator (sequential)
for i in plate("data", length(data))
    sample("obs_\$i", Normal(loc, scale); obs=data[i])
end
```

See also: [`PlateMessenger`](@ref), [`with_plate`](@ref)
"""
function plate(name::String, size::Int; dim::Int=-1)
    return PlateMessenger(name, size, dim)
end

"""
    with_plate(fn::Function, name::String, size::Int, dim::Int=-1)

Execute a function within a plate context.

# Arguments
- `fn`: The function to execute
- `name`: Name of the plate
- `size`: Number of elements
- `dim`: Batch dimension (default: -1)

# Example
```julia
with_plate("data", 100, -1) do
    sample("obs", MvNormal(loc * ones(100), I); obs=data)
end
```
"""
function with_plate(fn::Function, name::String, size::Int, dim::Int=-1)
    handler = PlateMessenger(name, size, dim)
    push!(PYRO_STACK, handler)
    try
        return fn()
    finally
        @assert PYRO_STACK[end] === handler
        pop!(PYRO_STACK)
    end
end

# =============================================================================
# Apply Stack (Core Message Dispatch)
# =============================================================================

"""
    apply_stack(msg::Dict)

Apply all handlers on the PYRO_STACK to a message.

This is the core dispatch mechanism for effect handlers. It processes the
message through all handlers in reverse order (most recently added first),
then post-processes in forward order.

# Process
1. Iterate through handlers in reverse, calling `process_message`
2. If any handler sets `msg["stop"] = true`, stop processing
3. If no handler set a value, compute it using `msg["fn"]`
4. Iterate through handlers and call `postprocess_message`

# Arguments
- `msg`: A dictionary containing:
  - `"type"`: "sample" or "param"
  - `"name"`: Site name
  - `"fn"`: Function to call for value
  - `"args"`: Arguments for fn
  - `"value"`: Pre-set value (for observations or replay)

# Returns
- The modified message dictionary with computed `"value"`
"""
function apply_stack(msg::Dict)
    # Process in reverse order (most recent handler first)
    pointer = 0
    for handler in reverse(PYRO_STACK)
        pointer += 1
        process_message(handler, msg)
        if get(msg, "stop", false)
            break
        end
    end

    # If no handler set the value, compute it
    if msg["value"] === nothing
        msg["value"] = msg["fn"](msg["args"]...)
    end

    # Post-process in forward order (from where we stopped)
    start_idx = length(PYRO_STACK) - pointer + 1
    for handler in PYRO_STACK[start_idx:end]
        postprocess_message(handler, msg)
    end

    return msg
end

# =============================================================================
# Sample Primitive
# =============================================================================

"""
    sample(name::String, dist::Distribution; obs=nothing)

Sample from a probability distribution or observe data.

This is the core primitive for probabilistic programming. When called:
- If `obs` is provided, the value is observed (clamped to `obs`)
- If effect handlers are active, they can modify the sampling behavior
- Otherwise, a value is sampled from the distribution

# Arguments
- `name`: Unique name for this sample site
- `dist`: A `Distribution` from Distributions.jl
- `obs`: Optional observed value (for conditioning)

# Returns
- The sampled or observed value

# Example
```julia
# Sample from a prior
loc = sample("loc", Normal(0.0, 1.0))

# Observe data (likelihood)
sample("obs", Normal(loc, 1.0); obs=data)
```

# Note
Sample site names must be unique within a single execution trace.

See also: [`param`](@ref), [`Trace`](@ref)
"""
function sample(name::String, dist::Distribution; obs=nothing)
    # If no active handlers, just sample directly
    if isempty(PYRO_STACK)
        if obs !== nothing
            return obs
        else
            return rand(dist)
        end
    end

    # Create the initial message
    initial_msg = Dict{String, Any}(
        "type" => "sample",
        "name" => name,
        "fn" => d -> rand(d),
        "args" => (dist,),
        "kwargs" => Dict("obs" => obs),
        "value" => obs,  # Pre-set value if observed
        "dist" => dist,  # Store distribution for log_prob computation
    )

    # Apply the effect handler stack
    msg = apply_stack(initial_msg)
    return msg["value"]
end

# =============================================================================
# Param Primitive
# =============================================================================

"""
    param(name::String, init_value=nothing; constraint=identity)

Register or retrieve a learnable parameter.

Parameters are stored in the global PARAM_STORE and persist across function
calls. They are stored in unconstrained space for optimization and transformed
to constrained space when accessed.

# Arguments
- `name`: Unique name for this parameter
- `init_value`: Initial value (required on first access)
- `constraint`: Transform function from unconstrained to constrained space

# Returns
- The constrained parameter value

# Common Constraints
- `identity`: No constraint (default)
- `exp`: Positive values (e.g., for scale parameters)
- `x -> 1 / (1 + exp(-x))`: Values in (0, 1)
- Custom functions

# Example
```julia
# Unconstrained location parameter
loc = param("loc", 0.0)

# Positive scale parameter using exp constraint
scale = param("scale_log", 0.0; constraint=exp)
# Equivalent to: scale = exp(param("scale_log", 0.0))

# Access parameter value after training
store = get_param_store()
unconstrained_val, constraint_fn = store["scale_log"]
constrained_val = constraint_fn(unconstrained_val)
```

See also: [`get_param_store`](@ref), [`clear_param_store!`](@ref)
"""
function param(name::String, init_value=nothing; constraint::Function=identity)
    function param_fn(init_value, constraint)
        if haskey(PARAM_STORE, name)
            unconstrained_value, stored_constraint = PARAM_STORE[name]
        else
            # Initialize with the provided value
            @assert init_value !== nothing "Initial value required for new parameter '$name'"
            # Store in unconstrained space
            # For simplicity, we assume init_value is already in unconstrained space
            unconstrained_value = Float64(init_value)
            PARAM_STORE[name] = (unconstrained_value, constraint)
        end

        # Return constrained value
        return constraint(unconstrained_value)
    end

    # If no active handlers, just call the function
    if isempty(PYRO_STACK)
        return param_fn(init_value, constraint)
    end

    # Create message and apply handlers
    initial_msg = Dict{String, Any}(
        "type" => "param",
        "name" => name,
        "fn" => param_fn,
        "args" => (init_value, constraint),
        "value" => nothing,
    )

    msg = apply_stack(initial_msg)
    return msg["value"]
end

# =============================================================================
# ELBO Loss Function
# =============================================================================

"""
    elbo(model::Function, guide::Function, args...; kwargs...)

Compute the Evidence Lower Bound (ELBO) for variational inference.

The ELBO is defined as:
    ELBO = E_q[log p(z, x) - log q(z)]

where:
- `p(z, x)` is the joint probability under the model
- `q(z)` is the variational distribution (guide)
- The expectation is over samples from q

This implementation returns the negative ELBO (a loss to minimize).

# Arguments
- `model`: The probabilistic model function
- `guide`: The variational guide function
- `args...`: Arguments passed to both model and guide
- `kwargs...`: Keyword arguments passed to both model and guide

# Returns
- `-ELBO`: The negative evidence lower bound (to be minimized)

# Algorithm
1. Run guide and trace all sample sites → get `q(z)` samples and densities
2. Run model with replayed values → get `p(z, x)` densities
3. Compute `ELBO = Σ log p(z) + Σ log p(x|z) - Σ log q(z)`
4. Return `-ELBO`

# Example
```julia
loss = elbo(model, guide, data)
# Minimize loss to fit the guide to the posterior
```

# Note
This implementation assumes all random variables use reparameterized gradients
(the "reparameterization trick"). It does not support score function estimators.

See also: [`SVI`](@ref), [`Trace_ELBO`](@ref)
"""
function elbo(model::Function, guide::Function, args...; kwargs...)
    # Run guide and trace execution
    guide_trace = trace() do
        guide(args...; kwargs...)
    end

    # Run model with replayed values from guide
    model_trace = trace() do
        replay(guide_trace) do
            model(args...; kwargs...)
        end
    end

    # Compute ELBO
    elbo_val = 0.0

    # Add log p(z, x) terms from model
    for (name, site) in model_trace
        if site["type"] == "sample"
            dist = site["dist"]
            value = site["value"]
            # Handle both scalar and array values
            lp = logpdf(dist, value)
            elbo_val += isa(lp, Number) ? lp : sum(lp)
        end
    end

    # Subtract log q(z) terms from guide
    for (name, site) in guide_trace
        if site["type"] == "sample"
            dist = site["dist"]
            value = site["value"]
            lp = logpdf(dist, value)
            elbo_val -= isa(lp, Number) ? lp : sum(lp)
        end
    end

    # Return negative ELBO (we minimize loss, maximize ELBO)
    return -elbo_val
end

"""
    Trace_ELBO(; kwargs...)

Wrapper for compatibility with Pyro API. Returns the elbo function.

# Example
```julia
loss_fn = Trace_ELBO()
svi = SVI(model, guide, optimizer, loss_fn)
```
"""
function Trace_ELBO(; kwargs...)
    return elbo
end

# =============================================================================
# Optimizer Wrapper
# =============================================================================

"""
    Adam

Adaptive moment estimation optimizer wrapper for MiniPyro.

This wraps Optimisers.jl's Adam optimizer and manages per-parameter optimizer
states, allowing for dynamic parameter creation during training.

# Fields
- `lr::Float64`: Learning rate
- `beta::Tuple{Float64, Float64}`: Exponential decay rates for moment estimates
- `states::Dict`: Per-parameter optimizer states

# Example
```julia
optimizer = Adam(0.02)
# or with custom betas:
optimizer = Adam(0.01; beta=(0.9, 0.999))
```

See also: [`SVI`](@ref)
"""
mutable struct Adam
    lr::Float64
    beta::Tuple{Float64, Float64}
    states::Dict{String, Any}

    Adam(lr::Float64=0.001; beta::Tuple{Float64, Float64}=(0.9, 0.999)) =
        new(lr, beta, Dict{String, Any}())
end

"""
    step!(optimizer::Adam, params::Dict)

Take an optimization step for all parameters.

# Arguments
- `optimizer`: The Adam optimizer
- `params`: Dictionary mapping parameter names to (value, gradient) tuples

Updates parameters in-place in the PARAM_STORE.
"""
function step!(optimizer::Adam, param_grads::Dict)
    opt = Optimisers.Adam(optimizer.lr, optimizer.beta)

    for (name, grad) in param_grads
        if !haskey(PARAM_STORE, name)
            continue
        end

        unconstrained_val, constraint = PARAM_STORE[name]

        # Initialize optimizer state if needed
        if !haskey(optimizer.states, name)
            optimizer.states[name] = Optimisers.setup(opt, unconstrained_val)
        end

        state = optimizer.states[name]

        # Compute update
        state, new_val = Optimisers.update(state, unconstrained_val, grad)
        optimizer.states[name] = state

        # Update parameter store
        PARAM_STORE[name] = (new_val, constraint)
    end
end

# =============================================================================
# SVI (Stochastic Variational Inference)
# =============================================================================

"""
    SVI

Stochastic Variational Inference optimizer.

SVI is the unified interface for variational inference in MiniPyro. It combines
a model, guide, optimizer, and loss function to perform gradient-based optimization
of the guide parameters.

# Fields
- `model::Function`: The probabilistic model
- `guide::Function`: The variational guide (approximation to posterior)
- `optim::Adam`: The optimizer
- `loss::Function`: The loss function (typically `elbo`)

# Example
```julia
model(data) = ...
guide(data) = ...

svi = SVI(model, guide, Adam(0.02), elbo)

for step in 1:1000
    loss = svi_step!(svi, data)
    if step % 100 == 0
        println("Step \$step: loss = \$loss")
    end
end
```

See also: [`svi_step!`](@ref), [`elbo`](@ref), [`Adam`](@ref)
"""
struct SVI
    model::Function
    guide::Function
    optim::Adam
    loss::Function
end

"""
    svi_step!(svi::SVI, args...; kwargs...)

Take a single SVI optimization step.

This method:
1. Computes the loss (ELBO) for the current parameters
2. Computes gradients with respect to all parameters
3. Updates parameters using the optimizer

# Arguments
- `svi`: The SVI object
- `args...`: Arguments passed to model and guide
- `kwargs...`: Keyword arguments passed to model and guide

# Returns
- `Float64`: The loss value for this step

# Example
```julia
for step in 1:num_steps
    loss = svi_step!(svi, data)
end
```
"""
function svi_step!(svi::SVI, args...; kwargs...)
    # Get current parameter values for gradient computation
    param_names = collect(keys(PARAM_STORE))

    if isempty(param_names)
        # First call - need to initialize parameters by running once
        svi.loss(svi.model, svi.guide, args...; kwargs...)
        param_names = collect(keys(PARAM_STORE))
    end

    # Create a function that takes parameter values and returns loss
    function loss_fn(param_values)
        # Temporarily update PARAM_STORE with new values
        for (i, name) in enumerate(param_names)
            _, constraint = PARAM_STORE[name]
            PARAM_STORE[name] = (param_values[i], constraint)
        end
        return svi.loss(svi.model, svi.guide, args...; kwargs...)
    end

    # Get current parameter values
    current_values = [PARAM_STORE[name][1] for name in param_names]

    # Compute loss and gradients using Zygote
    loss_val, grads = Zygote.withgradient(loss_fn, current_values)

    # Convert gradients to dictionary
    param_grads = Dict{String, Any}()
    if grads[1] !== nothing
        for (i, name) in enumerate(param_names)
            param_grads[name] = grads[1][i]
        end
    end

    # Take optimizer step
    step!(svi.optim, param_grads)

    return loss_val
end

end # module MiniPyro
