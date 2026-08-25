# Reverse-mode AD counterpart of FiniteDiff/Diff128x128.jl.

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

include("reconstruct.jl")
run_ad_reconstruction((128, 128, 1))
