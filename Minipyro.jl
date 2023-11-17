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


mutable struct Trace
    fn::Function
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
    fn::Function
    hide_fn::Function
end

function process_message(b::Block, msg)
    if hide_fn(msg):
        msg["stop"] = true
    end
end

mutable struct Replay
    fn::Function
    guide_trace::Function
end

function process_message(r:Replay, msg)
    if msg["name"] in r.guide_trace:
        msg["value"] = r.guide_trace[msg["name"]]["value"]
    end
end


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