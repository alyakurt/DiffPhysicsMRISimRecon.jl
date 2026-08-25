using KomaMRIPlots

function save_loss_curve(losses, filename, title)
    plotly = KomaMRIPlots.PlotlyBase
    trace = plotly.scatter(
        x=0:(length(losses) - 1),
        y=max.(losses, eps(Float64)),
        mode="lines+markers",
    )
    layout = plotly.Layout(
        title=title,
        xaxis_title="Iteration",
        yaxis_title="Loss",
        yaxis_type="log",
        template="plotly_white",
    )
    KomaMRIPlots.savefig(plotly.Plot(trace, layout), filename)
    nothing
end
