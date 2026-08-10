module CoilIntensityCorrection

using Interpolations: Flat, linear_interpolation
using KomaMRI
using LinearAlgebra: dot, norm

const SCC_REGULARIZATION = 5f-2
const SCC_TOLERANCE = 1f-4
const SCC_MAXIMUM_ITERATIONS = 500

centered_axis(width, count) = range(-width / 2 + width / (2count); step=width / count, length=count)
centered_fft(data) = KomaMRI.fftshift(KomaMRI.fft(KomaMRI.ifftshift(data, (1, 2, 3)), (1, 2, 3)), (1, 2, 3))
centered_ifft(data) = KomaMRI.fftshift(KomaMRI.ifft(KomaMRI.ifftshift(data, (1, 2, 3)), (1, 2, 3)), (1, 2, 3))

function tukey_window(count, alpha=0.5f0)
    alpha == 0 && return ones(Float32, count)
    position = range(0f0, 1f0; length=count)
    Float32[
        x < alpha / 2 ? (1 + cospi(2x / alpha - 1)) / 2 :
        x > 1 - alpha / 2 ? (1 + cospi(2x / alpha - 2 / alpha + 1)) / 2 : 1
        for x in position
    ]
end

function prescan_images(profiles)
    readout = Int(first(profiles).head.number_of_samples)
    phase_count = maximum(Int(profile.head.idx.kspace_encode_step_1) for profile in profiles) + 1
    partition_count = maximum(Int(profile.head.idx.kspace_encode_step_2) for profile in profiles) + 1
    coil_count = size(first(profiles).data, 2)
    kspace = zeros(ComplexF32, readout, phase_count, partition_count, coil_count)
    for profile in profiles
        phase = Int(profile.head.idx.kspace_encode_step_1) + 1
        partition = Int(profile.head.idx.kspace_encode_step_2) + 1
        kspace[:, phase, partition, :] .= profile.data
    end
    centered_ifft(kspace)
end

function root_sum_of_squares(images, channels)
    dropdims(sqrt.(sum(abs2, images[:, :, :, channels]; dims=4)); dims=4)
end

function isotropic_prescan(image)
    source_size = size(image)
    target_size = (source_size[1], 2 * source_size[2], 2 * source_size[3])
    source_kspace = centered_fft(image)
    target_kspace = zeros(ComplexF32, target_size)
    phase = (fld(target_size[2] - source_size[2], 2) + 1):(fld(target_size[2] - source_size[2], 2) + source_size[2])
    partition = (fld(target_size[3] - source_size[3], 2) + 1):(fld(target_size[3] - source_size[3], 2) + source_size[3])
    window = reshape(tukey_window(source_size[2]), 1, :, 1) .* reshape(tukey_window(source_size[3]), 1, 1, :)
    target_kspace[:, phase, partition] .= source_kspace .* window
    oversampled = centered_ifft(target_kspace)
    readout = (fld(target_size[1], 4) + 1):(target_size[1] - fld(target_size[1], 4))
    abs.(Array(@view oversampled[readout, :, :]))
end

periodic_laplacian(values) =
    6values .- circshift(values, (1, 0, 0)) .- circshift(values, (-1, 0, 0)) .-
    circshift(values, (0, 1, 0)) .- circshift(values, (0, -1, 0)) .-
    circshift(values, (0, 0, 1)) .- circshift(values, (0, 0, -1))

function smooth_sensitivity_scale(surface, body)
    normalization = maximum(body)
    surface = Float32.(surface ./ normalization)
    body = Float32.(body ./ normalization)
    body_squared = body .^ 2
    right_hand_side = body .* surface
    apply_normal(values) = body_squared .* values .+ SCC_REGULARIZATION .* periodic_laplacian(values)
    correction = zeros(Float32, size(body))
    residual = copy(right_hand_side)
    direction = copy(residual)
    residual_norm_squared = dot(residual, residual)
    iteration = 0
    while iteration < SCC_MAXIMUM_ITERATIONS && sqrt(residual_norm_squared) > SCC_TOLERANCE
        iteration += 1
        normal_direction = apply_normal(direction)
        step = residual_norm_squared / dot(direction, normal_direction)
        correction .+= step .* direction
        residual .-= step .* normal_direction
        next_residual_norm_squared = dot(residual, residual)
        direction .= residual .+ (next_residual_norm_squared / residual_norm_squared) .* direction
        residual_norm_squared = next_residual_norm_squared
    end
    correction, iteration, sqrt(residual_norm_squared)
end

function imaging_plane(correction, adjustment_profile, reference_profile, adjustment_fov, map_x, map_y)
    volume_axes = Tuple(map(centered_axis, adjustment_fov, size(correction)))
    interpolation = linear_interpolation(volume_axes, correction; extrapolation_bc=Flat())
    adjustment_position = Float32.(collect(adjustment_profile.head.position))
    adjustment_directions = hcat(
        Float32.(collect(adjustment_profile.head.read_dir)),
        Float32.(collect(adjustment_profile.head.phase_dir)),
        Float32.(collect(adjustment_profile.head.slice_dir)),
    )
    reference_position = Float32.(collect(reference_profile.head.position))
    reference_read = Float32.(collect(reference_profile.head.read_dir))
    reference_phase = Float32.(collect(reference_profile.head.phase_dir))
    result = Matrix{Float32}(undef, length(map_x), length(map_y))
    for (column, y) in enumerate(map_y), (row, x) in enumerate(map_x)
        patient_position = reference_position .+ 1000x .* reference_read .+ 1000y .* reference_phase
        coordinates = adjustment_directions' * (patient_position - adjustment_position)
        result[row, column] = interpolation(coordinates...)
    end
    result
end

"""Estimate the SCC paper's common real sensitivity-scale map from a Siemens adjustment prescan."""
function intensity_correction_map(adjustment_file, raw_reference, map_x, map_y)
    adjustment = RawAcquisitionData(ISMRMRDFile(adjustment_file))
    surface_profiles = filter(profile -> profile.head.active_channels > 2 && profile.head.number_of_samples == 128, adjustment.profiles)
    body_profiles = filter(profile -> profile.head.active_channels == 2 && profile.head.number_of_samples == 128, adjustment.profiles)
    isempty(surface_profiles) && error("Surface-coil prescan profiles are missing")
    isempty(body_profiles) && error("Body-coil prescan profiles are missing")

    adjustment_names = getproperty.(adjustment.params["coilLabel"], :name)
    reference_names = getproperty.(raw_reference.params["coilLabel"], :name)
    surface_channels = map(reference_names) do name
        channel = findfirst(==(name), adjustment_names)
        isnothing(channel) && error("Prescan does not contain EPI coil $name")
        channel
    end

    surface_images = prescan_images(surface_profiles)
    body_images = prescan_images(body_profiles)
    surface = isotropic_prescan(root_sum_of_squares(surface_images, surface_channels))
    body = isotropic_prescan(root_sum_of_squares(body_images, axes(body_images, 4)))
    correction, iterations, residual = smooth_sensitivity_scale(surface, body)
    adjustment_fov = Float32.(adjustment.params["reconFOV"])
    correction_map = imaging_plane(correction, first(surface_profiles), first(raw_reference.profiles), adjustment_fov, map_x, map_y)
    (; map=correction_map, iterations, residual)
end

"""Absorb the fully sampled calibration image phase into ESPIRiT maps for a real density unknown."""
function calibration_phase(acq_reference, sensitivity_maps, recon_size)
    direct_parameters = Dict{Symbol,Any}(:reco => "direct", :reconSize => recon_size)
    coil_images = ComplexF32.(Array(reconstruction(acq_reference, direct_parameters)[:, :, 1, 1, :, 1]))
    maps = @view sensitivity_maps[:, :, 1, :]
    combined = dropdims(sum(conj.(maps) .* coil_images; dims=3); dims=3)
    combined ./ max.(abs.(combined), eps(Float32))
end

end
