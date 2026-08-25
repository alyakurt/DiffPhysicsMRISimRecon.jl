using Enzyme
using KomaMRI
using LinearAlgebra: dot, norm
using MRICoilSensitivities: espirit
using Reactant
@eval using $(Sys.isapple() ? :Metal : :CUDA)

isdefined(@__MODULE__, :CoilIntensityCorrection) || include(joinpath(@__DIR__, "..", "coil_intensity_correction.jl"))

Reactant.set_default_backend(get(ENV, "REACTANT_BACKEND", Sys.isapple() ? "cpu" : "cuda"))
Reactant.allowscalar(false)

const RECON_SIZE = (128, 128)
const DISPLAY_RANGE = (0f0, 2f-5)
const ITERATIONS = 20

centered_axis(width, count) = range(-width / 2 + width / (2count); step=width / count, length=count)

function interpolation_matrix(source, target)
    matrix = zeros(Float32, length(target), length(source))
    for (row, position) in enumerate(target)
        if position <= first(source)
            matrix[row, 1] = 1
        elseif position >= last(source)
            matrix[row, end] = 1
        else
            left = searchsortedlast(source, position)
            weight = Float32((position - source[left]) / (source[left + 1] - source[left]))
            matrix[row, left] = 1 - weight
            matrix[row, left + 1] = weight
        end
    end
    matrix
end

function sense_signal(x, model)
    density = model.interpolation_x * reshape(x, model.voxel_grid) * transpose(model.interpolation_y)
    coil_images = reshape(density, :, 1) .* model.maps
    shifted_images = reshape(coil_images[model.spatial_indices, :], RECON_SIZE..., size(model.maps, 2))
    spectrum = KomaMRI.fft(shifted_images, (1, 2))
    reshape(spectrum, :, size(model.maps, 2))[model.sample_indices, :]
end

objective(x, model) = sum(abs2, sense_signal(x, model) - model.data)

function objective_gradient(x, model)
    result = Enzyme.gradient(Enzyme.ReverseWithPrimal, objective, x, Enzyme.Const(model))
    result.val, result.derivs[1]
end

function sense_adjoint(data, model)
    spectrum = zeros(ComplexF32, prod(RECON_SIZE), size(model.maps, 2))
    for sample in axes(data, 1)
        spectrum[model.sample_indices[sample], :] .+= @view data[sample, :]
    end
    shifted_images = prod(RECON_SIZE) .* KomaMRI.ifft(reshape(spectrum, RECON_SIZE..., size(model.maps, 2)), (1, 2))
    coil_images = zeros(ComplexF32, prod(RECON_SIZE), size(model.maps, 2))
    coil_images[model.spatial_indices, :] .= reshape(shifted_images, :, size(model.maps, 2))
    image = reshape(vec(sum(conj.(model.maps) .* coil_images; dims=2)), RECON_SIZE)
    vec(real.(transpose(model.interpolation_x) * image * model.interpolation_y))
end

function gram_norm(model)
    vector = fill(inv(sqrt(Float32(prod(model.voxel_grid)))), prod(model.voxel_grid))
    for _ in 1:30
        product = sense_adjoint(sense_signal(vector, model), model)
        vector .= product ./ norm(product)
    end
    dot(vector, sense_adjoint(sense_signal(vector, model), model))
end

function save_density(x, name, title, controls)
    density = clamp.(reshape(x, controls.voxel_grid), DISPLAY_RANGE...)
    savefig(plot_image(density; title, zmin=first(DISPLAY_RANGE), zmax=last(DISPLAY_RANGE)), joinpath(controls.output_directory, name))
end

function build_problem(voxel_grid, output_directory; accelerated=true)
    archive_directory = isempty(ARGS) ? joinpath(homedir(), "Desktop/Archive (1)") : first(ARGS)
    reference_file = joinpath(archive_directory, "mrd_hdf5/meas_MID01094_FID34194_hard_epi_20interleaves_5avg_fatsat.mrd")
    measured_file = accelerated ?
        joinpath(archive_directory, "mrd_hdf5/meas_MID01109_FID34203_hard_epi_2x_20interleaves_5avg_fatsat.mrd") :
        reference_file
    adjustment_file = joinpath(archive_directory, "mrd_hdf5/meas_MID01050_FID34150_AdjCoilSens.mrd")
    sequence_file = joinpath(archive_directory, "seq", accelerated ?
        "hard_epi_2x_20interleaves_5avg_fatsat.seq" :
        "hard_epi_20interleaves_5avg_fatsat.seq")
    navigator_count = 3

    seq = read_seq(sequence_file)
    acquired_lines = if accelerated
        shots = Int(seq.DEF["EpiShots"])
        acquired_shots = parse.(Int, split(seq.DEF["EpiAcquiredShots"], ',')) .- 1
        [line for shot in acquired_shots for line in shot:shots:(RECON_SIZE[2] - 1)]
    else
        collect(0:(RECON_SIZE[2] - 1))
    end

    raw_reference = RawAcquisitionData(ISMRMRDFile(reference_file))
    raw_reference.profiles = raw_reference.profiles[(navigator_count + 1):(navigator_count + RECON_SIZE[2])]
    acq_reference = AcquisitionData(raw_reference)
    acq_reference.traj[1].circular = false
    sensitivity_maps = espirit(acq_reference, (6, 6), 30, RECON_SIZE; eigThresh_1=0.02, eigThresh_2=0.0)
    sensitivity_maps ./= max.(sqrt.(sum(abs2, sensitivity_maps; dims=4)), eps(Float32))
    phase = CoilIntensityCorrection.calibration_phase(acq_reference, sensitivity_maps, RECON_SIZE)

    fov = Float32.(raw_reference.params["reconFOV"]) .* 1f-3
    map_x = collect(LinRange(-fov[1] / 2, fov[1] / 2, RECON_SIZE[1]))
    map_y = collect(LinRange(-fov[2] / 2, fov[2] / 2, RECON_SIZE[2]))
    correction = CoilIntensityCorrection.intensity_correction_map(adjustment_file, raw_reference, map_x, map_y)
    sensitivity_scale = correction.map ./ maximum(correction.map)
    sensitivity_maps .*= reshape(phase .* sensitivity_scale, RECON_SIZE..., 1, 1)
    maps = reshape(Array(@view sensitivity_maps[:, :, 1, :]), prod(RECON_SIZE), :)

    raw_measured = RawAcquisitionData(ISMRMRDFile(measured_file))
    raw_measured.profiles = raw_measured.profiles[(navigator_count + 1):(navigator_count + length(acquired_lines))]
    if accelerated
        for (profile, line) in zip(raw_measured.profiles, acquired_lines)
            profile.head.idx.kspace_encode_step_1 = UInt16(line)
        end
    end
    acq_measured = AcquisitionData(raw_measured)
    acq_measured.traj[1].circular = false
    centered_samples = acq_measured.subsampleIndices[1]
    data = ComplexF32.(acq_measured.kdata[1])

    voxel_x, voxel_y = centered_axis.(fov[1:2], voxel_grid)
    interpolation_x = interpolation_matrix(voxel_x, map_x)
    interpolation_y = interpolation_matrix(voxel_y, map_y)
    spatial_indices = vec(KomaMRI.ifftshift(reshape(1:prod(RECON_SIZE), RECON_SIZE), (1, 2)))
    frequency_indices = vec(KomaMRI.fftshift(reshape(1:prod(RECON_SIZE), RECON_SIZE), (1, 2)))
    sample_indices = frequency_indices[centered_samples]
    model = (; interpolation_x, interpolation_y, maps, spatial_indices, sample_indices, data, voxel_grid)

    mkpath(output_directory)
    savefig(plot_image(sensitivity_scale; title="SCC sensitivity-scale map", zmin=0, zmax=1), joinpath(output_directory, "intensity_correction.png"))
    println("SCC correction: extrema = $(extrema(sensitivity_scale)), CG iterations = $(correction.iterations), residual = $(correction.residual)")
    model
end

function run_ad_reconstruction(voxel_grid; accelerated=true)
    resolution = first(voxel_grid)
    output_prefix = accelerated ? "ADDiff" : "ADFull"
    output_directory = joinpath(@__DIR__, "$(output_prefix)$(resolution)x$(resolution)")
    model = build_problem(voxel_grid[1:2], output_directory; accelerated)
    controls = (; voxel_grid=voxel_grid[1:2], output_directory)
    step = 0.45f0 / gram_norm(model)

    device_model = map(Reactant.to_rarray, model)
    x = Reactant.to_rarray(zeros(Float32, prod(controls.voxel_grid)))
    compiled_gradient = Reactant.@compile sync=true objective_gradient(x, device_model)
    loss, gradient = compiled_gradient(x, device_model)
    analytic_gradient = -2f0 .* sense_adjoint(model.data, model)
    gradient_error = norm(Array(gradient) - analytic_gradient) / norm(analytic_gradient)
    gradient_error < 2f-3 || error("AD gradient check failed: $gradient_error")
    data_norm = sum(abs2, model.data)

    for iteration in 1:ITERATIONS
        x = max.(x .- step .* gradient, 0f0)
        loss, gradient = compiled_gradient(x, device_model)
        save_density(Array(x), "iteration_$(lpad(iteration, 2, '0')).png", "Iteration $iteration", controls)
        iteration in (1, ITERATIONS) && println("AD $(resolution)x$(resolution) iteration $iteration: loss = $(Reactant.to_number(loss))")
    end

    density = reshape(Array(x), controls.voxel_grid)
    final_loss = Reactant.to_number(loss)
    relative_data_reduction = 1 - final_loss / data_norm
    center = div(resolution, 2):(div(resolution, 2) + 1)
    center_mean = sum(@view density[center, center]) / 4
    save_density(density, "reconstructed_density.png", "AD $(resolution)x$(resolution) reconstructed brain density", controls)
    metrics = """
    method = AD
    resolution = $(resolution)x$(resolution)
    final_loss = $final_loss
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
