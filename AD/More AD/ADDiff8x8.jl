# Reverse-mode AD counterpart of FiniteDiff/Diff8x8.jl.

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

include("reconstruct.jl")
run_ad_reconstruction((8, 8, 1))
