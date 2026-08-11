module MeasuredSense8x8

using Enzyme
using FiniteDiff
using Interpolations: Flat, linear_interpolation
using KomaMRI
using LinearAlgebra: dot, norm, opnorm
using MRICoilSensitivities: espirit

include("coil_intensity_correction.jl")

const RECON_SIZE = (128, 128)
const VOXEL_GRID = (8, 8)
const DISPLAY_RANGE = (0f0, 2f-5)
const ITERATIONS = 20
const FINITE_DIFFERENCE_STEP = 1f-7

centered_axis(width, count) = range(-width / 2 + width / (2count); step=width / count, length=count)
centered_fft(data) = KomaMRI.fftshift(KomaMRI.fft(KomaMRI.ifftshift(data, (1, 2)), (1, 2)), (1, 2))

function build_encoding(maps, samples, fov)
    map_x = collect(LinRange(-fov[1] / 2, fov[1] / 2, RECON_SIZE[1]))
    map_y = collect(LinRange(-fov[2] / 2, fov[2] / 2, RECON_SIZE[2]))
    voxel_x, voxel_y = centered_axis.(fov[1:2], VOXEL_GRID)
    encoding = Matrix{ComplexF32}(undef, length(samples) * size(maps, 3), prod(VOXEL_GRID))
    basis = zeros(Float32, prod(VOXEL_GRID))
    for voxel in eachindex(basis)
        fill!(basis, 0)
        basis[voxel] = 1
        interpolation = linear_interpolation((voxel_x, voxel_y), reshape(basis, VOXEL_GRID); extrapolation_bc=Flat())
        density = Float32[interpolation(x, y) for x in map_x, y in map_y]
        spectrum = centered_fft(reshape(density, RECON_SIZE..., 1) .* maps)
        encoding[:, voxel] .= vec(reshape(spectrum, :, size(maps, 3))[samples, :])
    end
    encoding
end

function build_problem(output_directory)
    archive_directory = isempty(ARGS) ? joinpath(homedir(), "Desktop/Archive (1)") : first(ARGS)
    reference_file = joinpath(archive_directory, "mrd_hdf5/meas_MID01094_FID34194_hard_epi_20interleaves_5avg_fatsat.mrd")
    measured_file = joinpath(archive_directory, "mrd_hdf5/meas_MID01109_FID34203_hard_epi_2x_20interleaves_5avg_fatsat.mrd")
    adjustment_file = joinpath(archive_directory, "mrd_hdf5/meas_MID01050_FID34150_AdjCoilSens.mrd")
    sequence_file = joinpath(archive_directory, "seq/hard_epi_2x_20interleaves_5avg_fatsat.seq")
    navigator_count = 3

    seq = read_seq(sequence_file)
    shots = Int(seq.DEF["EpiShots"])
    acquired_shots = parse.(Int, split(seq.DEF["EpiAcquiredShots"], ',')) .- 1
    acquired_lines = [line for shot in acquired_shots for line in shot:shots:(RECON_SIZE[2] - 1)]

    raw_reference = RawAcquisitionData(ISMRMRDFile(reference_file))
    raw_reference.profiles = raw_reference.profiles[(navigator_count + 1):(navigator_count + RECON_SIZE[2])]
    acq_reference = AcquisitionData(raw_reference)
    acq_reference.traj[1].circular = false
    maps = espirit(acq_reference, (6, 6), 30, RECON_SIZE; eigThresh_1=0.02, eigThresh_2=0.0)
    maps ./= max.(sqrt.(sum(abs2, maps; dims=4)), eps(Float32))
    phase = CoilIntensityCorrection.calibration_phase(acq_reference, maps, RECON_SIZE)

    fov = Float32.(raw_reference.params["reconFOV"]) .* 1f-3
    map_x = collect(LinRange(-fov[1] / 2, fov[1] / 2, RECON_SIZE[1]))
    map_y = collect(LinRange(-fov[2] / 2, fov[2] / 2, RECON_SIZE[2]))
    correction = CoilIntensityCorrection.intensity_correction_map(adjustment_file, raw_reference, map_x, map_y)
    sensitivity_scale = correction.map ./ maximum(correction.map)
    maps .*= reshape(phase .* sensitivity_scale, RECON_SIZE..., 1, 1)
    maps = Array(@view maps[:, :, 1, :])

    raw_measured = RawAcquisitionData(ISMRMRDFile(measured_file))
    raw_measured.profiles = raw_measured.profiles[(navigator_count + 1):(navigator_count + length(acquired_lines))]
    for (profile, line) in zip(raw_measured.profiles, acquired_lines)
        profile.head.idx.kspace_encode_step_1 = UInt16(line)
    end
    acq_measured = AcquisitionData(raw_measured)
    acq_measured.traj[1].circular = false
    samples = acq_measured.subsampleIndices[1]
    data = ComplexF32.(acq_measured.kdata[1])

    encoding = build_encoding(maps, samples, fov)
    real_encoding = vcat(real.(encoding), imag.(encoding))
    real_data = vcat(real.(vec(data)), imag.(vec(data)))
    gram = real_encoding' * real_encoding
    right_hand_side = real_encoding' * real_data
    data_norm = sum(abs2, real_data)

    mkpath(output_directory)
    savefig(plot_image(sensitivity_scale; title="SCC sensitivity-scale map", zmin=0, zmax=1), joinpath(output_directory, "intensity_correction.png"))
    println("SCC correction: extrema = $(extrema(sensitivity_scale)), CG iterations = $(correction.iterations), residual = $(correction.residual)")
    (; gram, right_hand_side, data_norm, output_directory)
end

objective(x, problem)::Float32 = dot(x, problem.gram * x) - 2f0 * dot(problem.right_hand_side, x) + problem.data_norm

function objective_gradient(::Val{:AD}, x, problem)
    result = Enzyme.gradient(Enzyme.ReverseWithPrimal, objective, x, Enzyme.Const(problem))
    result.val, result.derivs[1]
end

function objective_gradient(::Val{:FiniteDiff}, x, problem)
    loss(values) = objective(values, problem)
    loss(x), FiniteDiff.finite_difference_gradient(loss, x; absstep=FINITE_DIFFERENCE_STEP)
end

function save_density(x, name, title, problem)
    density = clamp.(reshape(x, VOXEL_GRID), DISPLAY_RANGE...)
    savefig(plot_image(density; title, zmin=first(DISPLAY_RANGE), zmax=last(DISPLAY_RANGE)), joinpath(problem.output_directory, name))
end

function run(method::Symbol, output_directory)
    problem = build_problem(output_directory)
    x = zeros(Float32, prod(VOXEL_GRID))
    loss, gradient = objective_gradient(Val(method), x, problem)
    analytic_gradient = -2f0 .* problem.right_hand_side
    gradient_error = norm(gradient - analytic_gradient) / norm(analytic_gradient)
    gradient_error < 2f-3 || error("$method gradient check failed: $gradient_error")
    step = 0.45f0 / opnorm(problem.gram)
    for iteration in 1:ITERATIONS
        x .= max.(x .- step .* gradient, 0f0)
        loss, gradient = objective_gradient(Val(method), x, problem)
        save_density(x, "iteration_$(lpad(iteration, 2, '0')).png", "Iteration $iteration", problem)
        iteration in (1, ITERATIONS) && println("$method iteration $iteration: loss = $loss")
    end

    save_density(x, "reconstructed_density.png", "$method reconstructed brain density", problem)
    density = reshape(x, VOXEL_GRID)
    relative_data_reduction = 1 - objective(x, problem) / problem.data_norm
    center_mean = sum(density[4:5, 4:5]) / 4
    metrics = """
    method = $method
    final_loss = $(objective(x, problem))
    relative_data_reduction = $relative_data_reduction
    gradient_relative_error = $gradient_error
    density_extrema = $(extrema(density))
    center_mean = $center_mean
    display_range = $DISPLAY_RANGE
    """
    write(joinpath(output_directory, "metrics.txt"), metrics)
    println(metrics)
    density
end

end
