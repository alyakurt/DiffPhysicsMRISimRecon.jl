# Reverse-mode AD counterpart of FiniteDiff/Diff64x64.jl.

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

include("reconstruct.jl")
run_ad_reconstruction((64, 64, 1))
