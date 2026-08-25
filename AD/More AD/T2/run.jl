using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))
Pkg.instantiate()

include(joinpath(@__DIR__, "..", "relaxometry.jl"))

isempty(ARGS) || error("Usage: julia --project=. AD/T2/run.jl")
run_relaxometry(Val(:T2))
