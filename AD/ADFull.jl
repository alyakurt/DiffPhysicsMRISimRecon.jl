# Reverse-mode AD reconstruction of the fully sampled 8 x 8 in-vivo brain.

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

include("reconstruct_full.jl")
run_ad_full_reconstruction((8, 8, 1))
