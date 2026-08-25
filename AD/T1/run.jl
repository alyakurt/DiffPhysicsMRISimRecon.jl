using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))
Pkg.instantiate()

include(joinpath(@__DIR__, "..", "relaxometry.jl"))

isempty(ARGS) || error("Usage: julia --project=. AD/T1/run.jl")
run_relaxometry(Val(:T1))
