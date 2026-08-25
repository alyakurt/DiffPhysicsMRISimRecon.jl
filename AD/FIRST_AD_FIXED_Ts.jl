using Enzyme
using KomaMRI
using KomaMRIPlots
using MRICoilSensitivities: espirit
using Reactant

const IMAGE_SIZE = (128, 128)
const ITERATIONS = 20
const STEP_SIZE = 2f-5
const FIXED_T1 = 0.04154f0
const FIXED_T2 = 0.30352f0
const NAVIGATORS = 3
const DATA_DIRECTORY = isempty(ARGS) ? joinpath(homedir(), "Desktop/Archive (1)") : first(ARGS)
const OUTPUT_DIRECTORY = joinpath(@__DIR__, "FINAL_AD_FIXED_Ts")

Reactant.set_default_backend("cpu")
Reactant.allowscalar(false)

centered_axis(width, count) = Float32.(range(-width / 2 + width / (2count); step=width / count, length=count))

function load_problem()
    reference_path = joinpath(DATA_DIRECTORY, "mrd_hdf5/meas_MID01094_FID34194_hard_epi_20interleaves_5avg_fatsat.mrd")
    measured_path = joinpath(DATA_DIRECTORY, "mrd_hdf5/meas_MID01109_FID34203_hard_epi_2x_20interleaves_5avg_fatsat.mrd")
    sequence_path = joinpath(DATA_DIRECTORY, "seq/hard_epi_2x_20interleaves_5avg_fatsat.seq")

    sequence = resolve_triggers(read_seq(sequence_path), CardiacSignal(; heart_rate=1))
    shots = Int(sequence.DEF["EpiShots"])
    acquired_shots = parse.(Int, split(sequence.DEF["EpiAcquiredShots"], ',')) .- 1
    image_profiles = sum(length(shot:shots:(IMAGE_SIZE[2] - 1)) for shot in acquired_shots)

    raw_reference = RawAcquisitionData(ISMRMRDFile(reference_path))
    fov = Float32.(raw_reference.params["reconFOV"]) .* 1f-3
    raw_reference.profiles = raw_reference.profiles[(NAVIGATORS + 1):(NAVIGATORS + IMAGE_SIZE[2])]
    reference = AcquisitionData(raw_reference)
    reference.traj[1].circular = false
    maps = espirit(reference, (6, 6), 30, IMAGE_SIZE; eigThresh_1=0.02, eigThresh_2=0.0)

    measured = RawAcquisitionData(ISMRMRDFile(measured_path))
    measured.profiles = measured.profiles[(NAVIGATORS + 1):(NAVIGATORS + image_profiles)]
    b = reduce(vcat, ComplexF32.(profile.data) for profile in measured.profiles)

    adc_blocks = findall(block -> is_ADC_on(sequence[block]), eachindex(sequence.DUR))
    sequence = sequence[1:adc_blocks[NAVIGATORS + image_profiles]]
    samples_per_profile = size(first(measured.profiles).data, 1)
    first_image_sample = NAVIGATORS * samples_per_profile + 1
    image_samples = first_image_sample:(first_image_sample + size(b, 1) - 1)

    x_axis, y_axis = centered_axis.(fov[1:2], IMAGE_SIZE)
    spin_x = repeat(x_axis, IMAGE_SIZE[2])
    spin_y = repeat(y_axis; inner=IMAGE_SIZE[1])
    spin_z = zeros(Float32, prod(IMAGE_SIZE))

    map_x = collect(LinRange(-fov[1] / 2, fov[1] / 2, IMAGE_SIZE[1]))
    map_y = collect(LinRange(-fov[2] / 2, fov[2] / 2, IMAGE_SIZE[2]))
    map_z = Float32[-fov[3] / 2, 0, fov[3] / 2]
    receiver = ArbitraryCoilSens(map_x, map_y, map_z, repeat(maps, 1, 1, length(map_z), 1))
    coil_values = ComplexF32.(get_sens(receiver, spin_x, spin_y, spin_z))
    count = prod(IMAGE_SIZE)
    T1 = fill(FIXED_T1, count)
    T2 = fill(FIXED_T2, count)
    T2s = fill(1f0, count)
    object = Phantom(; x=spin_x, y=spin_y, z=spin_z, ρ=ones(Float32, count), T1, T2, T2s)
    scanner = Scanner(; receiver=KomaMRICore.CoilSensitivities(Reactant.to_rarray(coil_values), nothing))
    sim_params = Dict{String,Any}("sim_method" => Bloch(), "gpu" => false, "Nthreads" => 1, "return_type" => "mat", "precision" => "f32")

    (; object=Reactant.to_rarray(object), sequence, scanner, sim_params, image_samples, b=Reactant.to_rarray(b))
end

function A(x, params)
    object = copy(params.object)
    object.ρ .= real.(x)
    real_signal = simulate(object, params.sequence, params.scanner; sim_params=params.sim_params, verbose=false)
    object.ρ .= imag.(x)
    imaginary_signal = simulate(object, params.sequence, params.scanner; sim_params=params.sim_params, verbose=false)
    return real_signal[params.image_samples, :, 1] .+ ComplexF32(0, 1) .* imaginary_signal[params.image_samples, :, 1]
end

f(x, params) = sum(abs2, A(x, params) - params.b)

function f_and_gradient(x, params)
    result = Enzyme.gradient(Enzyme.ReverseWithPrimal, f, x, Enzyme.Const(params))
    result.val, result.derivs[1]
end

function save_iteration(x, iteration)
    image = reshape(abs.(Array(x)), IMAGE_SIZE)
    figure = plot_image(image; title="Fixed-T AD iteration $iteration")
    savefig(figure, joinpath(OUTPUT_DIRECTORY, "iteration_$(lpad(iteration, 2, '0')).png"))
end

function run_final_ad()
    total_start = time_ns()
    mkpath(OUTPUT_DIRECTORY)
    params = load_problem()
    x = Reactant.to_rarray(zeros(ComplexF32, prod(IMAGE_SIZE)))

    compile_start = time_ns()
    gradient = Reactant.@compile sync=true f_and_gradient(x, params)
    compile_seconds = (time_ns() - compile_start) / 1e9

    losses = Float64[]
    optimization_start = time_ns()
    for iteration in 0:ITERATIONS
        value, ∇f = gradient(x, params)
        push!(losses, Reactant.to_number(value))
        save_iteration(x, iteration)
        println("iteration=$iteration loss=$(last(losses)) T1=$FIXED_T1 T2=$FIXED_T2")
        iteration == ITERATIONS || (x = x .- STEP_SIZE .* ∇f)
    end
    optimization_seconds = (time_ns() - optimization_start) / 1e9

    rows = ["iteration,loss,T1,T2"; ["$iteration,$(losses[iteration + 1]),$FIXED_T1,$FIXED_T2" for iteration in 0:ITERATIONS]]
    write(joinpath(OUTPUT_DIRECTORY, "loss.csv"), join(rows, '\n') * "\n")
    total_seconds = (time_ns() - total_start) / 1e9
    println("compile_seconds=$compile_seconds optimization_seconds=$optimization_seconds total_seconds=$total_seconds")
    (; x=Array(x), losses, compile_seconds, optimization_seconds, total_seconds)
end

abspath(PROGRAM_FILE) == (@__FILE__) && run_final_ad()
