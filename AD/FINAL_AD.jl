using Enzyme
using Interpolations: Flat, linear_interpolation
using KomaMRI
using KomaMRIPlots
using Reactant

const IMAGE_SIZE = (128, 128)
const ITERATIONS = 80
const STEP_SIZE = 2f-5
const RELAXATION_STEP_SIZE = 1f3
const NAVIGATORS = 0
const DATA_DIRECTORY = isempty(ARGS) ?
    joinpath(homedir(), "Desktop/Data/cleaner brain data acq 21 august") : first(ARGS)
const OUTPUT_DIRECTORY = joinpath(@__DIR__, "FINAL_AD")

Reactant.set_default_backend("cpu")
Reactant.allowscalar(false)

centered_axis(width, count) = Float32.(range(-width / 2 + width / (2count); step=width / count, length=count))
centered_fft(data) = KomaMRI.fftshift(KomaMRI.fft(KomaMRI.ifftshift(data, (1, 2, 3)), (1, 2, 3)), (1, 2, 3))
centered_ifft(data) = KomaMRI.fftshift(KomaMRI.ifft(KomaMRI.ifftshift(data, (1, 2, 3)), (1, 2, 3)), (1, 2, 3))

function prescan_images(profiles)
    readout = Int(first(profiles).head.number_of_samples)
    phase_count = maximum(Int(profile.head.idx.kspace_encode_step_1) for profile in profiles) + 1
    partition_count = maximum(Int(profile.head.idx.kspace_encode_step_2) for profile in profiles) + 1
    kspace = zeros(ComplexF32, readout, phase_count, partition_count, size(first(profiles).data, 2))
    for profile in profiles
        phase = Int(profile.head.idx.kspace_encode_step_1) + 1
        partition = Int(profile.head.idx.kspace_encode_step_2) + 1
        kspace[:, phase, partition, :] .= profile.data
    end
    centered_ifft(kspace)
end

function isotropic_prescan(image)
    source_size = size(image)
    target_size = (source_size[1], 2source_size[2], 2source_size[3])
    target = zeros(ComplexF32, target_size)
    phase = (fld(source_size[2], 2) + 1):(fld(source_size[2], 2) + source_size[2])
    partition = (fld(source_size[3], 2) + 1):(fld(source_size[3], 2) + source_size[3])
    position = range(0f0, 1f0; length=source_size[2])
    window = Float32[x < 0.25f0 || x > 0.75f0 ? 0.5f0 * (1 - cospi(4x)) : 1 for x in position]
    target[:, phase, partition] .= centered_fft(image) .* reshape(window, 1, :, 1) .* reshape(window, 1, 1, :)
    oversampled = centered_ifft(target)
    readout = (fld(target_size[1], 4) + 1):(target_size[1] - fld(target_size[1], 4))
    Array(@view oversampled[readout, :, :])
end

function adjustment_sensitivity_maps(path, imaging, map_x, map_y; sensitivity_floor=1f-2)
    adjustment = RawAcquisitionData(ISMRMRDFile(path))
    adjustment_names = getproperty.(adjustment.params["coilLabel"], :name)
    imaging_names = getproperty.(imaging.params["coilLabel"], :name)
    channels = [only(findall(==(name), adjustment_names)) for name in imaging_names]
    profiles = filter(
        profile -> profile.head.active_channels == length(adjustment_names) && profile.head.number_of_samples == 128,
        adjustment.profiles,
    )
    images = prescan_images(profiles)
    volumes = cat((isotropic_prescan(@view images[:, :, :, channel]) for channel in channels)...; dims=4)
    rss = sqrt.(sum(abs2, volumes; dims=4))
    volumes ./= max.(rss, sensitivity_floor * maximum(rss))

    volume_axes = Tuple(map(centered_axis, Float32.(adjustment.params["reconFOV"]), size(volumes)[1:3]))
    adjustment_position = Float32[first(profiles).head.position...]
    adjustment_directions = hcat(
        Float32[first(profiles).head.read_dir...],
        Float32[first(profiles).head.phase_dir...],
        Float32[first(profiles).head.slice_dir...],
    )
    imaging_position = Float32[first(imaging.profiles).head.position...]
    imaging_read = Float32[first(imaging.profiles).head.read_dir...]
    imaging_phase = Float32[first(imaging.profiles).head.phase_dir...]
    maps = cat((
        let interpolation = linear_interpolation(volume_axes, @view(volumes[:, :, :, channel]); extrapolation_bc=Flat())
            [interpolation((adjustment_directions' * (imaging_position .+ 1000x .* imaging_read .+ 1000y .* imaging_phase .- adjustment_position))...) for x in map_x, y in map_y]
        end for channel in axes(volumes, 4)
    )...; dims=3)
    maps = reverse(reshape(ComplexF32.(maps), length(map_x), length(map_y), 1, length(channels)); dims=1)
    rss = sqrt.(sum(abs2, maps; dims=4))
    maps ./ max.(rss, sensitivity_floor * maximum(rss))
end

function load_problem()
    measured_path = joinpath(DATA_DIRECTORY, "brain_gre_3t_acc/meas_MID00492_FID42184_bssfp_optimized_2x.mrd")
    adjustment_path = joinpath(DATA_DIRECTORY, "brain_gre_3t_acc/meas_MID00478_FID42170_AdjCoilSens.mrd")
    sequence_path = joinpath(DATA_DIRECTORY, "bssfp_slice_all_adc_optimized_R2.seq")

    sequence = resolve_triggers(read_seq(sequence_path), CardiacSignal(; heart_rate=1))
    adc_blocks = findall(block -> is_ADC_on(sequence[block]), eachindex(sequence.DUR))
    image_profiles = length(adc_blocks) - NAVIGATORS

    measured = RawAcquisitionData(ISMRMRDFile(measured_path))
    fov = Float32.(measured.params["reconFOV"]) .* 1f-3
    map_x = collect(LinRange(-fov[1] / 2, fov[1] / 2, IMAGE_SIZE[1]))
    map_y = collect(LinRange(-fov[2] / 2, fov[2] / 2, IMAGE_SIZE[2]))
    maps = adjustment_sensitivity_maps(adjustment_path, measured, map_x, map_y)
    measured.profiles = measured.profiles[(NAVIGATORS + 1):(NAVIGATORS + image_profiles)]
    b = reduce(vcat, ComplexF32.(profile.data) for profile in measured.profiles)

    sequence = sequence[1:adc_blocks[NAVIGATORS + image_profiles]]
    samples_per_profile = size(first(measured.profiles).data, 1)
    first_image_sample = NAVIGATORS * samples_per_profile + 1
    image_samples = first_image_sample:(first_image_sample + size(b, 1) - 1)

    x_axis, y_axis = centered_axis.(fov[1:2], IMAGE_SIZE)
    spin_x = repeat(x_axis, IMAGE_SIZE[2])
    spin_y = repeat(y_axis; inner=IMAGE_SIZE[1])
    spin_z = zeros(Float32, prod(IMAGE_SIZE))

    map_z = Float32[-fov[3] / 2, 0, fov[3] / 2]
    receiver = ArbitraryCoilSens(map_x, map_y, map_z, repeat(maps, 1, 1, length(map_z), 1))
    coil_values = ComplexF32.(get_sens(receiver, spin_x, spin_y, spin_z))
    spin_x = vcat(spin_x, spin_x)
    spin_y = vcat(spin_y, spin_y)
    spin_z = vcat(spin_z, spin_z)
    coil_values = vcat(coil_values, ComplexF32(0, 1) .* coil_values)
    count = length(spin_x)
    relaxation = fill(1f0, count)
    object = Phantom(; x=spin_x, y=spin_y, z=spin_z, ρ=ones(Float32, count), T1=relaxation, T2=relaxation, T2s=relaxation)
    scanner = Scanner(; receiver=KomaMRICore.CoilSensitivities(Reactant.to_rarray(coil_values), nothing))
    sim_params = Dict{String,Any}("sim_method" => Bloch(), "gpu" => true, "Nthreads" => 1, "return_type" => "mat", "precision" => "f32")

    (; object=Reactant.to_rarray(object), sequence, scanner, sim_params, image_samples, b=Reactant.to_rarray(b))
end

function A(x, T1, T2, params)
    object = copy(params.object)
    object.T1 .= T1
    object.T2 .= T2
    object.ρ .= vcat(real.(x), imag.(x))
    signal = simulate(object, params.sequence, params.scanner; sim_params=params.sim_params, verbose=false)
    signal[params.image_samples, :, 1]
end

f(x, T1, T2, params) = sum(abs2, A(x, T1, T2, params) - params.b)

function f_and_gradient(x, T1, T2, params)
    result = Enzyme.gradient(Enzyme.ReverseWithPrimal, f, x, T1, T2, Enzyme.Const(params))
    result.val, result.derivs[1], result.derivs[2], result.derivs[3]
end

function save_iteration(x, iteration)
    image = reverse(reshape(abs.(Array(x)), IMAGE_SIZE); dims=1)
    figure = plot_image(image; title="AD iteration $iteration")
    savefig(figure, joinpath(OUTPUT_DIRECTORY, "iteration_$(lpad(iteration, 2, '0')).png"))
end

function run_final_ad()
    total_start = time_ns()
    mkpath(OUTPUT_DIRECTORY)
    params = load_problem()
    x = Reactant.to_rarray(zeros(ComplexF32, prod(IMAGE_SIZE)))
    T1 = Reactant.to_rarray(Float32[1])
    T2 = Reactant.to_rarray(Float32[1])

    compile_start = time_ns()
    gradient = Reactant.@allowscalar Reactant.compile(
        f_and_gradient,
        (x, T1, T2, params);
        sync=true,
    )
    compile_seconds = (time_ns() - compile_start) / 1e9

    losses = Float64[]
    relaxation_values = Tuple{Float32,Float32}[]
    optimization_start = time_ns()
    for iteration in 0:ITERATIONS
        value, ∇f, ∇T1, ∇T2 = gradient(x, T1, T2, params)
        push!(losses, Reactant.to_number(value))
        push!(relaxation_values, (only(Array(T1)), only(Array(T2))))
        save_iteration(x, iteration)
        println("iteration=$iteration loss=$(last(losses)) T1=$(last(relaxation_values)[1]) T2=$(last(relaxation_values)[2])")
        if iteration != ITERATIONS
            x = x .- STEP_SIZE .* ∇f
            T1 = max.(T1 .- RELAXATION_STEP_SIZE .* ∇T1, eps(Float32))
            T2 = max.(T2 .- RELAXATION_STEP_SIZE .* ∇T2, eps(Float32))
        end
    end
    optimization_seconds = (time_ns() - optimization_start) / 1e9

    rows = ["iteration,loss,T1,T2"; ["$iteration,$(losses[iteration + 1]),$(relaxation_values[iteration + 1][1]),$(relaxation_values[iteration + 1][2])" for iteration in 0:ITERATIONS]]
    write(joinpath(OUTPUT_DIRECTORY, "loss.csv"), join(rows, '\n') * "\n")
    total_seconds = (time_ns() - total_start) / 1e9
    println("compile_seconds=$compile_seconds optimization_seconds=$optimization_seconds total_seconds=$total_seconds")
    (; x=Array(x), losses, compile_seconds, optimization_seconds, total_seconds)
end

abspath(PROGRAM_FILE) == (@__FILE__) && run_final_ad()
