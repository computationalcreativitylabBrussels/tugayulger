module Plotting

using DataFrames
using PlotlyJS

export plot_frequency_power, plot_frequency_normalized_power, plot_normalized_frequency_normalized_power, plot_spectrum, plot_grouped_resonances, print_reconstructed_values, plot_frequency_normalized_amplitude, dynamic_threshold_filter, plot_analyzed_frequencies

# plot in one window, frequency against power to determine peaks in power
function plot_frequency_power(df::DataFrame)
    trace = scatter(
        mode="markers",
        x=df.frequency,
        y=df.power,
        marker=attr(
            size=5,
            color="blue"
        ),
        name="Frequency vs Power"
    )

    layout = Layout(
        title="Frequency vs Power",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Power"
    )

    p = plot([trace], layout)
    #savefig(p, "frequency_power.png")

    return p
end

function plot_frequency_normalized_power(df::DataFrame)
    min_value = minimum(df[!, :power])
    max_value = maximum(df[!, :power])
    df[!, :normalized_power] = (df[!, :power] .- min_value) ./ (max_value .- min_value)

    trace = scatter(
        mode="markers",
        x=df.frequency,
        y=df.normalized_power,
        marker=attr(
            size=5,
            color="blue"
        ),
        name="Frequency vs Normalized Power"
    )

    layout = Layout(
        title="Frequency vs Normalized Power",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Normalized Power"
    )

    p = plot([trace], layout)
    savefig(p, "frequency_normalized_power.png")

    return p
end

function plot_normalized_frequency_normalized_power(df::DataFrame)
    # Normalize power
    min_power = minimum(df[!, :power])
    max_power = maximum(df[!, :power])
    df[!, :normalized_power] = (df[!, :power] .- min_power) ./ (max_power .- min_power)

    # Normalize frequency
    min_frequency = minimum(df[!, :frequency])
    max_frequency = maximum(df[!, :frequency])
    df[!, :normalized_frequency] = (df[!, :frequency] .- min_frequency) ./ (max_frequency .- min_frequency)

    # Create scatter plot with normalized frequency and power
    trace = scatter(
        mode="markers",
        x=df.normalized_frequency,
        y=df.normalized_power,
        marker=attr(
            size=5,
            color="blue"
        ),
        name="Normalized Frequency vs Normalized Power"
    )

    layout = Layout(
        title="Normalized Frequency vs Normalized Power",
        xaxis_title="Normalized Frequency",
        yaxis_title="Normalized Power"
    )

    p = plot([trace], layout)
    savefig(p, "normalized_frequency_normalized_power.png")

    return p
end

function plot_spectrum(processed_resonances, normalization_factors)
    min_value, max_value, min_frequency, max_frequency = normalization_factors

    original_data = DataFrame(onset=Float64[], frequency=Float64[], power=Float64[])
    for (onset, points) in processed_resonances
        for (normalized_frequency, normalized_power) in points
            original_frequency = normalized_frequency * (max_frequency - min_frequency) + min_frequency
            original_power = normalized_power * (max_value - min_value) + min_value
            push!(original_data, (onset / 16000, original_frequency, original_power))
        end
    end

    trace = scatter(
        mode="markers",
        x=original_data.onset,
        y=original_data.frequency,
        marker=attr(
            size=5,
            color=original_data.power,
            colorscale="Viridis",
            colorbar=attr(title="Power")
        ),
        name="Frequency vs Onset"
    )

    layout = Layout(
        title="Frequency vs Onset (Original Values)",
        xaxis_title="Onset (s)",
        yaxis_title="Frequency (Hz)"
    )

    p = plot([trace], layout)
    savefig(p, "original_spectrum.png")

    return p
end

function plot_grouped_resonances(grouped_resonances, normalization_factors)
    min_value, max_value, min_frequency, max_frequency = normalization_factors

    # Prepare data for plotting
    original_data = DataFrame(onset=Float64[], frequency=Float64[])
    for (onset, groups) in grouped_resonances
        for group in groups
            for row in eachrow(group)
                original_frequency = row[:normalized_frequency] * (max_frequency - min_frequency) + min_frequency
                push!(original_data, (onset / 16000, original_frequency))
            end
        end
    end

    # Create scatter plot
    trace = scatter(
        mode="markers",
        x=original_data.onset,
        y=original_data.frequency,
        marker=attr(
            size=5,
            color="blue"
        ),
        name="Grouped Resonances"
    )

    layout = Layout(
        title="Grouped Resonances (Original Values)",
        xaxis_title="Onset (s)",
        yaxis_title="Frequency (Hz)"
    )

    p = plot([trace], layout)
    savefig(p, "grouped_resonances.png")

    return p
end

function print_reconstructed_values(data, normalization_factors)
    min_value, max_value, min_frequency, max_frequency = normalization_factors

    println("Reconstructed Values:")
    for (onset, points) in data
        println("Onset (s): ", onset / 16000)
        for (normalized_frequency, normalized_value) in points
            original_frequency = normalized_frequency * (max_frequency - min_frequency) + min_frequency
            original_value = normalized_value * (max_value - min_value) + min_value
            println("  Frequency (Hz): ", original_frequency, ", Value: ", original_value)
        end
    end
end

function plot_frequency_normalized_amplitude(df::DataFrame)
    min_value = minimum(df[!, :amplitude])
    max_value = maximum(df[!, :amplitude])
    df[!, :normalized_amplitude] = (df[!, :amplitude] .- min_value) ./ (max_value .- min_value)

    trace = scatter(
        mode="markers",
        x=df.frequency,
        y=df.normalized_amplitude,
        marker=attr(
            size=5,
            color="blue"
        ),
        name="Frequency vs normalized_amplitude"
    )

    layout = Layout(
        title="Frequency vs normalized_amplitude",
        xaxis_title="Frequency (Hz)",
        yaxis_title="normalized_amplitude"
    )

    p = plot([trace], layout)
    savefig(p, "frequency_amplitude.png")

    return p
end

function dynamic_threshold_filter(df::DataFrame, column::Symbol=:amplitude, factor::Float64=0.5)
    # Calculate mean and standard deviation of the column
    mean_value = mean(df[!, column])
    std_dev = std(df[!, column])

    # Define the dynamic threshold
    dynamic_threshold = mean_value + factor * std_dev
    println("Dynamic Threshold: ", dynamic_threshold)

    # Filter rows based on the dynamic threshold
    filtered_df = filter(row -> row[column] >= dynamic_threshold, df)

    return filtered_df
end

function plot_analyzed_frequencies(df::DataFrame, analyzed_freq::Vector{Any})
    df[!, :time] = df[!, :onset] ./ 16000

    filtered_window = filter(:frequency => x -> x <= 5000, df)

    df = filtered_window

    trace1 = scatter(
        mode="markers",
        #x=df.time,
        x=DataFrame(),
        y=DataFrame(),
        #y=df.frequency,
        marker=attr(
           size=5,
            color="blue"
        ),
        name="Original Frequencies"
    )

    traces = [trace1]
    colors = ["red", "green", "orange", "purple"]
    for (time_onset, resonances) in analyzed_freq
        for (i, row) in enumerate(eachrow(resonances))
            trace_resonances = scatter(
                mode="markers",
                x=[time_onset / 16000],
                y=[row.mean_frequency],
                marker=attr(
                    size=8,
                    color=colors[(i - 1) % length(colors) + 1]
                ),
                name="Formant $i"
            )
            push!(traces, trace_resonances)
        end
    end

    layout = Layout(
        title="Frequency vs Time with Resonances",
        xaxis_title="Time (s)",
        yaxis_title="Frequency (Hz)"
    )

    p = plot(traces, layout)
    #savefig(p, "frequency_time_resonances.png")

    return p
end

end