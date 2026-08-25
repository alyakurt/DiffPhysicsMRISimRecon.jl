# Reverse-mode AD reconstruction of the fully sampled 128 x 128 in-vivo brain.

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

include("reconstruct.jl")
run_ad_reconstruction((128, 128, 1); accelerated=false)
