using Pkg

const REPOSITORY_DIRECTORY = normpath(joinpath(@__DIR__, ".."))
Base.active_project() == joinpath(REPOSITORY_DIRECTORY, "Project.toml") ||
    Pkg.activate(REPOSITORY_DIRECTORY)

using KomaMRI
using MRICoilSensitivities: espirit
using Statistics: quantile

const IMAGE_SIZE = (128, 128)
const ITERATIONS = 20
const DATA_DIRECTORY = isempty(ARGS) ?
    joinpath(homedir(), "Desktop/Data/cleaner brain data acq 21 august") : first(ARGS)

function load_acquisition(path)
    acquisition = AcquisitionData(RawAcquisitionData(ISMRMRDFile(path)))
    acquisition.traj[1].circular = false
    acquisition
end

function save_png(image, path)
    display_max = max(quantile(vec(image), 0.995), eps(eltype(image)))
    pixels = round.(UInt8, 255 .* clamp.(image ./ display_max, 0, 1))
    mktempdir() do directory
        pgm_path = joinpath(directory, "image.pgm")
        open(pgm_path, "w") do io
            write(io, "P5\n$(size(pixels, 2)) $(size(pixels, 1))\n255\n")
            write(io, permutedims(pixels))
        end
        run(`sips -s format png $pgm_path --out $path`)
    end
    nothing
end

reference = load_acquisition(joinpath(
    DATA_DIRECTORY,
    "brain_gre_3t_acc/meas_MID00488_FID42180_gaussian_fatsat.mrd",
))
sensitivity_maps = espirit(
    reference,
    (6, 6),
    30,
    IMAGE_SIZE;
    eigThresh_2=0.0,
)

measured = load_acquisition(joinpath(
    DATA_DIRECTORY,
    "brain_gre_3t_acc/meas_MID00489_FID42181_gaussian_fatsat_2x.mrd",
))
reconstructed = reconstruction(
    measured,
    Dict{Symbol,Any}(
        :reco => "multiCoil",
        :reconSize => IMAGE_SIZE,
        :senseMaps => sensitivity_maps,
        :iterations => ITERATIONS,
    ),
)
magnitude = rotr90(reverse(
    abs.(Array(reconstructed[:, :, 1, 1, 1, 1]));
    dims=(1, 2),
))
output_file = joinpath(@__DIR__, "Stand_FINAL.png")
save_png(magnitude, output_file)
println("Saved: ", output_file)
