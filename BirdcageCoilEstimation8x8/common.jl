module BirdcageCoilEstimation8x8

using Enzyme
using FiniteDiff
using Interpolations: Flat, linear_interpolation
using KomaMRI
using KomaMRIBase: BirdcageCoilSens, get_sens
using LinearAlgebra: norm, opnorm
using MRICoilSensitivities: espirit

const RECON_SIZE = 8
const CALIBRATION_SIZE = 64
const CALIBRATION_WIDTH = 30
const FOV = 0.24f0
const NCOILS = 8
const DISPLAY_RANGE = (0f0, 1f0)
const ITERATIONS = 100
const CENTER_RADIUS = 0.02f0
const EDGE_RADIUS = 0.075f0

centered_axis(width, count) = range(-width / 2 + width / (2count); step=width / count, length=count)

function birdcage_maps(receiver, axis)
    positions = collect(Iterators.product(axis, axis))
    x, y = first.(positions), last.(positions)
    reshape(get_sens(receiver, x, y, zeros(eltype(x), length(x))), length(axis), length(axis), NCOILS)
end

function normalize_coils(maps)
    coil_norm = sqrt.(sum(abs2, maps; dims=3))
    maps ./ max.(coil_norm, eps(Float32)), dropdims(coil_norm; dims=3)
end

centered_fft(data) = KomaMRI.fftshift(KomaMRI.fft(KomaMRI.ifftshift(data, (1, 2)), (1, 2)), (1, 2)) ./ sqrt(prod(size(data)[1:2]))

function calibration_density(axis)
    Float32[
        (x / 0.10f0)^2 + (y / 0.09f0)^2 <= 1 ?
        0.7f0 + 0.3f0 * exp(-((x / 0.045f0)^2 + (y / 0.05f0)^2)) : 0f0
        for x in axis, y in axis
    ]
end

function estimate_maps(true_maps, density)
    kspace = centered_fft(reshape(density, size(density)..., 1) .* true_maps)
    first_calibration = fld(CALIBRATION_SIZE - CALIBRATION_WIDTH, 2) + 1
    calibration = kspace[
        first_calibration:(first_calibration + CALIBRATION_WIDTH - 1),
        first_calibration:(first_calibration + CALIBRATION_WIDTH - 1),
        :,
    ]
    maps = espirit(
        Array(calibration),
        (CALIBRATION_SIZE, CALIBRATION_SIZE),
        (6, 6);
        eigThresh_1=0.02,
        eigThresh_2=0.0,
        use_poweriterations=false,
    )
    maps = ndims(maps) == 4 ? dropdims(maps; dims=4) : maps
    first(normalize_coils(ComplexF32.(maps)))
end

function projection_residual(coil_images, estimated_maps, support)
    combined = sum(conj.(estimated_maps) .* coil_images; dims=3)
    residual = estimated_maps .* combined .- coil_images
    support3 = reshape(support, size(support)..., 1)
    norm(residual .* support3) / norm(coil_images .* support3)
end

function interpolate_maps(maps, source_axis, target_axis)
    result = Array{ComplexF32}(undef, length(target_axis), length(target_axis), size(maps, 3))
    for coil in axes(maps, 3)
        interpolation = linear_interpolation((source_axis, source_axis), maps[:, :, coil]; extrapolation_bc=Flat())
        result[:, :, coil] .= [interpolation(x, y) for x in target_axis, y in target_axis]
    end
    first(normalize_coils(result))
end

function sense_matrix(maps)
    size_x, size_y, coil_count = size(maps)
    frequencies_x = collect(-fld(size_x, 2):(cld(size_x, 2) - 1))
    frequencies_y = collect(-fld(size_y, 2):(cld(size_y, 2) - 1))
    fourier_x = ComplexF32[
        exp(-2f0π * im * k * position / size_x) / sqrt(size_x)
        for k in frequencies_x, position in frequencies_x
    ]
    fourier_y = ComplexF32[
        exp(-2f0π * im * k * position / size_y) / sqrt(size_y)
        for k in frequencies_y, position in frequencies_y
    ]
    acquired_y = 1:2:size_y
    encoding = Matrix{ComplexF32}(undef, size_x * length(acquired_y) * coil_count, size_x * size_y)
    row = 1
    for coil in 1:coil_count, ky in acquired_y, kx in 1:size_x
        phase = reshape(fourier_x[kx, :], size_x, 1) .* reshape(fourier_y[ky, :], 1, size_y)
        encoding[row, :] .= vec(maps[:, :, coil] .* phase)
        row += 1
    end
    vcat(real.(encoding), imag.(encoding))
end

function true_density(axis)
    Float32[(x / 0.095f0)^2 + (y / 0.10f0)^2 <= 1 for x in axis, y in axis]
end

function build_problem()
    calibration_axis = collect(centered_axis(FOV, CALIBRATION_SIZE))
    reconstruction_axis = collect(centered_axis(FOV, RECON_SIZE))
    receiver = BirdcageCoilSens(; ncoils=NCOILS, radius=0.20, L=0.30)
    true_calibration_maps = birdcage_maps(receiver, calibration_axis)
    physical_coil_norm = dropdims(sqrt.(sum(abs2, true_calibration_maps; dims=3)); dims=3)
    calibration_object = calibration_density(calibration_axis)
    estimated_maps = estimate_maps(true_calibration_maps, calibration_object)
    coil_images = reshape(calibration_object, CALIBRATION_SIZE, CALIBRATION_SIZE, 1) .* true_calibration_maps
    support = calibration_object .> 0
    radius = [hypot(x, y) for x in calibration_axis, y in calibration_axis]
    center = radius .< CENTER_RADIUS
    edge = support .& (radius .> EDGE_RADIUS)
    physical_center_edge_ratio = sum(physical_coil_norm[center]) / count(center) / (sum(physical_coil_norm[edge]) / count(edge))
    direct_residual = projection_residual(coil_images, estimated_maps, support)
    conjugate_residual = projection_residual(coil_images, conj.(estimated_maps), support)
    estimated_maps_8 = interpolate_maps(estimated_maps, calibration_axis, reconstruction_axis)
    density = true_density(reconstruction_axis)
    encoding = sense_matrix(estimated_maps_8)
    data = encoding * vec(density)
    (; density, encoding, data, direct_residual, conjugate_residual, physical_center_edge_ratio)
end

loss(x, encoding, data)::Float32 = sum(abs2, encoding * x - data)

function loss_gradient(::Val{:AD}, x, encoding, data)
    result = Enzyme.gradient(Enzyme.ReverseWithPrimal, loss, x, Enzyme.Const(encoding), Enzyme.Const(data))
    result.val, result.derivs[1]
end

function loss_gradient(::Val{:FiniteDiff}, x, encoding, data)
    objective(values) = loss(values, encoding, data)
    objective(x), FiniteDiff.finite_difference_gradient(objective, x; relstep=cbrt(eps(Float32)))
end

function save_density(density, path, title)
    savefig(
        plot_image(
            clamp.(density, DISPLAY_RANGE...);
            title,
            zmin=first(DISPLAY_RANGE),
            zmax=last(DISPLAY_RANGE),
        ),
        path,
    )
end

function run_reconstruction(method::Symbol, problem=build_problem())
    output_directory = joinpath(@__DIR__, method == :AD ? "AD8x8" : "FiniteDiff8x8")
    mkpath(output_directory)
    save_density(problem.density, joinpath(output_directory, "truth.png"), "Known density")
    x = zeros(Float32, length(problem.density))
    step = 0.45f0 / opnorm(problem.encoding)^2
    for iteration in 1:ITERATIONS
        objective, gradient = loss_gradient(Val(method), x, problem.encoding, problem.data)
        x .= max.(x .- step .* gradient, 0f0)
        iteration in (1, 20, ITERATIONS) && println("$method iteration $iteration: loss = $objective")
    end
    reconstruction = reshape(x, size(problem.density))
    save_density(reconstruction, joinpath(output_directory, "reconstructed_density.png"), "$method birdcage reconstruction")
    relative_error = norm(reconstruction - problem.density) / norm(problem.density)
    center = reconstruction[div(RECON_SIZE, 2):(div(RECON_SIZE, 2) + 1), div(RECON_SIZE, 2):(div(RECON_SIZE, 2) + 1)]
    support_mean = sum(reconstruction .* problem.density) / sum(problem.density)
    center_ratio = sum(center) / length(center) / support_mean
    metrics = """
    method = $method
    final_loss = $(loss(x, problem.encoding, problem.data))
    relative_density_error = $relative_error
    center_to_support_ratio = $center_ratio
    direct_projection_residual = $(problem.direct_residual)
    conjugate_projection_residual = $(problem.conjugate_residual)
    physical_center_to_edge_coil_norm = $(problem.physical_center_edge_ratio)
    """
    write(joinpath(output_directory, "metrics.txt"), metrics)
    println(metrics)
    (; reconstruction, relative_error, center_ratio)
end

end
