module RenderOpenMOLLIADResults

using DelimitedFiles
using KomaMRIPlots
using KomaMRIPlots: QMRIColors

const Plotly = KomaMRIPlots.PlotlyBase
const IMAGE_SIZE = (192, 144)

function color_scale()
    colors = replace.(
        string.(QMRIColors.relaxationColorMap("T1") .* 255),
        "RGB{Float64}" => "rgb",
    )
    positions = range(0.0, 1.0; length=length(colors))
    [(position, color) for (position, color) in zip(positions, colors)]
end

function render_map(input_file, support, title, output_file; smooth=false)
    image = Array{Float32}(undef, IMAGE_SIZE)
    read!(input_file, image)
    image = reverse(image; dims=(1, 2))
    support = reverse(support; dims=(1, 2))
    values = Matrix{Union{Missing,Float32}}(undef, IMAGE_SIZE)
    values[support] .= 1f3 .* image[support]
    values[.!support] .= missing
    trace = Plotly.heatmap(;
        z=values,
        zmin=0,
        zmax=2000,
        zsmooth=smooth ? "best" : false,
        colorscale=color_scale(),
        colorbar=Plotly.attr(;
            title=Plotly.attr(; text="T1 (ms)", side="right"),
            tickvals=[0, 500, 1000, 1500, 2000],
            thickness=96,
            len=0.82,
        ),
        hovertemplate="T1: %{z:.0f} ms<extra></extra>",
    )
    layout = Plotly.Layout(;
        title=Plotly.attr(; text=title, x=0.5, xanchor="center"),
        width=3000,
        height=3600,
        margin=Plotly.attr(; t=270, l=160, r=460, b=140),
        xaxis=Plotly.attr(; visible=false, constrain="domain"),
        yaxis=Plotly.attr(; visible=false, autorange="reversed", scaleanchor="x"),
        paper_bgcolor="white",
        plot_bgcolor="black",
        font=Plotly.attr(; family="Arial", size=56, color="#263c5c"),
    )
    KomaMRIPlots.savefig(
        Plotly.Plot(trace, layout), output_file; width=3000, height=3600,
    )
end

function render_loss(loss_file, output_file)
    data = readdlm(loss_file, ',', Float64; header=true)[1]
    trace = Plotly.scatter(;
        x=data[:, 1],
        y=data[:, 2],
        mode="lines",
        line=Plotly.attr(; color="#263c5c", width=4),
        hovertemplate="Iteration %{x:.0f}<br>Loss %{y:.6f}<extra></extra>",
    )
    layout = Plotly.Layout(;
        title=Plotly.attr(; text="Koma AD T1 loss convergence", x=0.5, xanchor="center"),
        width=900,
        height=550,
        margin=Plotly.attr(; t=80, l=95, r=35, b=80),
        xaxis=Plotly.attr(; title="Iteration", range=[0, 80], gridcolor="#e7e9ed"),
        yaxis=Plotly.attr(;
            title="Normalized measured-data loss",
            gridcolor="#e7e9ed",
            rangemode="tozero",
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=Plotly.attr(; family="Arial", size=17, color="#263c5c"),
    )
    KomaMRIPlots.savefig(
        Plotly.Plot(trace, layout), output_file; width=2400, height=1400,
    )
end

function render(directory)
    support_bytes = Array{UInt8}(undef, IMAGE_SIZE)
    read!(joinpath(directory, "support.u8"), support_bytes)
    support = Bool.(support_bytes)
    for iteration in (0, 20, 40, 80)
        render_map(
            joinpath(directory, "iteration_$(lpad(iteration, 3, '0')).f32"),
            support,
            "openMOLLIst Koma AD T1 - iteration $iteration",
            joinpath(directory, "iteration_$(lpad(iteration, 3, '0'))_map.png"),
        )
    end
    render_map(
        joinpath(directory, "quantitative_t1.f32"),
        support,
        "openMOLLIst Koma AD quantitative T1 map",
        joinpath(directory, "koma_ad_quantitative_t1_map.png"),
    )
    render_map(
        joinpath(directory, "quantitative_t1.f32"),
        support,
        "openMOLLIst Koma AD quantitative T1 map - display interpolated",
        joinpath(directory, "koma_ad_quantitative_t1_map_display_interpolated.png");
        smooth=true,
    )
    render_loss(
        joinpath(directory, "loss.csv"),
        joinpath(directory, "loss_convergence.png"),
    )
end

export render

end
