# Finite-difference 8 x 8 in-vivo brain reconstruction.

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

include(joinpath(@__DIR__, "..", "reconstruct_8x8.jl"))
MeasuredSense8x8.run(:FiniteDiff, joinpath(@__DIR__, "Diff8x8"))
