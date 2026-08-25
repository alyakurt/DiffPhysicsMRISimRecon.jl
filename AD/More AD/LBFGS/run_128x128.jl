using Pkg
Pkg.activate(joinpath(@__DIR__, "..", ".."))
Pkg.instantiate()

include(joinpath(@__DIR__, "reconstruct.jl"))
run_lbfgs_reconstruction((128, 128, 1))
