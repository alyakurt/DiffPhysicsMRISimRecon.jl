# Reverse-mode AD counterpart of FiniteDiff/Diff32x32.jl.

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

include("reconstruct.jl")
run_ad_reconstruction((32, 32, 1))
