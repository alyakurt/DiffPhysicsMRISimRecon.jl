# Reverse-mode AD counterpart of FiniteDiff/Diff16x16.jl.

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

include("reconstruct.jl")
run_ad_reconstruction((16, 16, 1))
