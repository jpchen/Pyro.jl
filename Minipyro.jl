module Pyro

using Statistics
using Printf
using Flux
using Flux.Tracker: param, back!, grad

# See minipyro.py for the original Python implementation

const PYRO_STACK = []
const PARAM_STORE = {}

function get_param_store()
    return PARAM_STORE
end

struct Messenger
    fn::Function
end

mutable struct Trace
    m::Messenger
    trace::Dict{String, Any}
end

# Constructor for Trace
# Trace(fn::Function) = Trace(Messenger(fn), Dict{String, Any}())

function push!(trace::Trace)
    push!(trace.Trace)
    empty!(trace.trace)
    return trace.trace
end

function (t::Trace)(args...)
    trace_dict = push!(t)
    try
        return t.fn(args...)
    finally
        pop!(t)
    end
    return trace_dict
end

function postprocess_message(t::Trace, msg)
    @assert msg["type"] != "sample"
    t.trace[msg["name"]] = copy(msg)
    return msg
end

function get_trace(t: trace, args...)
    with t
        t.fn(args)
    return t.trace
end

mutable struct Block
    m::Messenger
    fn::Function
    hide_fn::Function
end

function process_message(b::Block, msg)
    if hide_fn(msg):
        msg["stop"] = true
    end
end

mutable struct Replay
    m::Messenger
    fn::Function
    guide_trace::Function
end

function process_message(r:Replay, msg)
    if msg["name"] in r.guide_trace:
        msg["value"] = r.guide_trace[msg["name"]]["value"]
    end
end

mutable struct Plate
    m::Messenger
    fn::Function
    size::Int
    dim::Int
end


# make plate an iterator
Base.iterate(p::PlateMessenger, state=1) = state > p.size ? nothing : (state, state + 1)
Base.length(p::PlateMessenger) = p.size
Base.eltype(::Type{PlateMessenger}) = Int

function sample(name, fn, args...; kwargs...)
    obs = kwargs.pop("obs", nothing)
    if isempty(PYRO_STACK)
        return fn(args..., kwargs...)
    end
    initial_msg = {
        "type": "sample",
        "name": name,
        "fn": fn,
        "args": args,
        "kwargs": kwargs,
        "value": obs,
    }
    msg = apply_stack(initial_msg)
    return msg["value"]
end


function param(name, init_value=nothing; constraint=identity, event_dim=nothing)
    if event_dim !== nothing
        error("event_dim argument is not supported")
    end

    function param_fn(init_value, constraint)
        if haskey(PARAM_STORE, name)
            unconstrained_value, _ = PARAM_STORE[name]
        else
            @assert init_value !== nothing "Initial value must be provided"
            constrained_value = deepcopy(init_value)
            unconstrained_value = Flux.data(constraint(constrained_value))
            unconstrained_value = Flux.param(unconstrained_value)
            PARAM_STORE[name] = unconstrained_value, constraint
        end
        return constraint(unconstrained_value)
    end

    if isempty(PYRO_STACK)
        return param_fn(init_value, constraint)
    else
        initial_msg = (type="param", name=name, fn=param_fn, args=(init_value, constraint), value=nothing)
        msg = apply_stack(initial_msg)
        return msg.value
    end
end

# Placeholder for constraint transformation functions
identity(x) = x  # Identity function for unconstrained parameters

struct Adam
    optim_args
    optim_objs
end

function Adam(optim_args)
    optim_objs = Dict()
    return Adam(optim_args, optim_objs)
end

function (optimizer::Adam)(params)
    for param in params
        if haskey(optimizer.optim_objs, param)
            optim = optimizer.optim_objs[param]
        else
            optim = Flux.ADAM([param], optimizer.optim_args...)
            optimizer.optim_objs[param] = optim
        end
        Flux.Optimise.update!(optim, [param], [grad(param)])
    end
end

function elbo(model, guide, args...; kwargs...)
    guide_trace = trace(() -> guide(args...; kwargs...))
    model_trace = trace(() -> replay(model, guide_trace)(args...; kwargs...))
    
    elbo_val = 0.0
    for (name, site) in model_trace
        if site.type == "sample"
            elbo_val += sum(site.log_prob(site.value))
        end
    end
    for (name, site) in guide_trace
        if site.type == "sample"
            elbo_val -= sum(site.log_prob(site.value))
        end
    end
    return -elbo_val
end


struct SVI
    model
    guide
    optim
    loss
end

function step(svi::SVI, args...; kwargs...)
    # This wraps both the call to `model` and `guide` in a `trace` so that
    # we can record all the parameters that are encountered. Note that
    # further tracing occurs inside of `loss`.
    param_capture = trace() do
        # We use block here to allow tracing to record parameters only.
        block(hide_fn = msg -> msg["type"] == "sample") do
            loss = svi.loss(svi.model, svi.guide, args...; kwargs...)
        end
    end
    # Differentiate the loss.
    Flux.back!(loss)
    # #TODO unconstrained and constrained parameters equivalent in flux
    params = [site["value"] for site in values(param_capture)]
    # Take a step w.r.t. each parameter in params.
    svi.optim(params)
    # Zero out the gradients so that they don't accumulate.
    # TODO check if this is valid
    for p in params
        p.grad .= zeros(size(p))
    end
    return loss.data
end