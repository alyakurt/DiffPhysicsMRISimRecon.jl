# Measured-versus-simulated comparison from the KomaMRI coil-sensitivity how-to

using Pkg
repository_directory = normpath(joinpath(@__DIR__, ".."))
Pkg.activate(repository_directory)
Pkg.instantiate()

using KomaMRI, MRICoilSensitivities
using KomaMRIBase: ArbitraryCoilSens
using Statistics: quantile

fully_sampled_mrd_file = joinpath(
    homedir(),
    "Desktop/Archive (1)/mrd_hdf5/meas_MID01094_FID34194_hard_epi_20interleaves_5avg_fatsat.mrd",
)
accelerated_2x_mrd_file = joinpath(
    homedir(),
    "Desktop/Archive (1)/mrd_hdf5/meas_MID01109_FID34203_hard_epi_2x_20interleaves_5avg_fatsat.mrd",
)
accelerated_2x_seq_file = joinpath(
    homedir(),
    "Desktop/Archive (1)/seq/hard_epi_2x_20interleaves_5avg_fatsat.seq",
)
phantom_2D_3T_file = joinpath(
    homedir(),
    "Downloads/brain2D_3T_fat_z1cm_2x_xy.phantom",
)

recon_size = (128, 128)
R = 2
seq = read_seq(accelerated_2x_seq_file)

number_of_shots = Int(seq.DEF["EpiShots"])
acquired_shot_indices =
    parse.(Int, split(seq.DEF["EpiAcquiredShots"], ',')) .- 1
acquired_line_indices = [
    line_index for shot_index in acquired_shot_indices for
    line_index in shot_index:number_of_shots:(recon_size[2] - 1)
]
@assert length(acquired_line_indices) == recon_size[2] ÷ R

function select_profiles!(raw, line_indices)
    navigator_count = 3
    raw.profiles = raw.profiles[
        (navigator_count + 1):(navigator_count + length(line_indices))
    ]
    for (profile, line_index) in zip(raw.profiles, line_indices)
        profile.head.idx.kspace_encode_step_1 = UInt16(line_index)
    end
    nothing
end

raw_reference = RawAcquisitionData(ISMRMRDFile(fully_sampled_mrd_file))
raw_reference.profiles = raw_reference.profiles[4:(3 + recon_size[2])]
acq_reference = AcquisitionData(raw_reference)
acq_reference.traj[1].circular = false
sensitivity_maps = espirit(
    acq_reference,
    (6, 6),
    30,
    recon_size;
    eigThresh_1=0.02,
    eigThresh_2=0.0,
)

raw_measured = RawAcquisitionData(ISMRMRDFile(accelerated_2x_mrd_file))
select_profiles!(raw_measured, acquired_line_indices)
acq_measured = AcquisitionData(raw_measured)
acq_measured.traj[1].circular = false

direct_params = Dict{Symbol,Any}(
    :reco => "direct",
    :reconSize => recon_size,
)
sense_params = Dict{Symbol,Any}(
    :reco => "multiCoil",
    :reconSize => recon_size,
    :senseMaps => sensitivity_maps,
    :iterations => 20,
    :densityWeighting => false,
    :toeplitz => false,
)

fov = Float32.(raw_reference.params["reconFOV"]) .* 1f-3
x = collect(LinRange(-fov[1] / 2, fov[1] / 2, recon_size[1]))
y = collect(LinRange(-fov[2] / 2, fov[2] / 2, recon_size[2]))
z = Float32[-fov[3] / 2, 0, fov[3] / 2]
receiver = ArbitraryCoilSens(
    x,
    y,
    z,
    repeat(sensitivity_maps, 1, 1, length(z), 1),
)

obj = read_phantom(phantom_2D_3T_file)
raw_simulated = simulate(
    obj,
    seq,
    Scanner(; receiver);
    physio=CardiacSignal(; heart_rate=1),
    verbose=false,
)

output_directory = joinpath(repository_directory, "MRIRecoResults")
mkpath(output_directory)
simulated_mrd_file = joinpath(output_directory, "simulated_acquisition.mrd")
save(ISMRMRDFile(simulated_mrd_file), raw_simulated)
raw_simulated = RawAcquisitionData(ISMRMRDFile(simulated_mrd_file))
select_profiles!(raw_simulated, acquired_line_indices)

acq_simulated = AcquisitionData(raw_simulated)
acq_simulated.traj[1].circular = false

magnitude_image(image) = abs.(Array(image[:, :, 1, 1, 1, 1]))
plot_reconstruction(image, title) = plot_image(
    image;
    title,
    zmin=0,
    zmax=quantile(vec(image), 0.995),
)

measured_direct = reconstruction(acq_measured, direct_params)
measured_sense = reconstruction(acq_measured, sense_params)
simulated_direct = reconstruction(acq_simulated, direct_params)
simulated_sense = reconstruction(acq_simulated, sense_params)

results = (
    (
        "acquisition_direct.png",
        reverse(magnitude_image(measured_direct); dims=(1, 2)),
        "Measured direct (R=2)",
    ),
    (
        "acquisition_sense.png",
        reverse(magnitude_image(measured_sense); dims=(1, 2)),
        "Measured SENSE (R=2)",
    ),
    (
        "simulated_mrd_direct.png",
        magnitude_image(simulated_direct),
        "Simulated MRD direct (R=2)",
    ),
    (
        "simulated_mrd_sense.png",
        magnitude_image(simulated_sense),
        "Simulated SENSE (R=2)",
    ),
)

foreach(results) do (filename, image, title)
    output_file = joinpath(output_directory, filename)
    savefig(plot_reconstruction(image, title), output_file)
    println("Saved: ", output_file)
end

println("Saved: ", simulated_mrd_file)
