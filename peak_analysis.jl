module PeakAnalysis

using DataFrames, Statistics, SciPy

export identify_peaks, process_peaks, determine_peaks

function identify_peaks(df::DataFrame, window_size::Float64, step_size::Float64, threshold::Float64, column::Symbol=:power, frequency_diff::Float64 = 100.0, group_size_tolerance::Float64 = 100.0)
    # Normalize the specified column
    min_value = minimum(df[!, column])
    max_value = maximum(df[!, column])
    df[!, :normalized_value] = (df[!, column] .- min_value) ./ (max_value - min_value)

    # Normalize frequency
    min_frequency = minimum(df[!, :frequency])
    max_frequency = maximum(df[!, :frequency])
    df[!, :normalized_frequency] = (df[!, :frequency] .- min_frequency) ./ (max_frequency - min_frequency)

    # Prepare output
    windowed_groups = []

    # Set initial window parameters based on actual onset values
    current_time = minimum(df.onset)
    end_time = maximum(df.onset)

    while current_time + window_size <= end_time
        window_start = current_time
        window_end = current_time + window_size

        # Filter data within the window and above threshold
        resonances = filter(row -> row.onset >= window_start && row.onset < window_end &&
                                   row.normalized_value >= threshold, df)

        if isempty(resonances)
            current_time += step_size
            continue
        end

        # Sort by normalized frequency
        sort!(resonances, :normalized_frequency)

        groups = []
        current_group = resonances[1:1, :]

        first_frequency = resonances[1, :normalized_frequency]
        last_frequency = resonances[1, :normalized_frequency]
        for i in 2:nrow(resonances)
            freq = resonances[i, :normalized_frequency]
            if abs(freq - last_frequency) <= frequency_diff && abs(freq - first_frequency) <= group_size_tolerance
                push!(current_group, resonances[i, :])
                last_frequency = freq
            else
                push!(groups, current_group)
                current_group = resonances[i:i, :]
                first_frequency = freq
                last_frequency = freq
            end
        end
        push!(groups, current_group)

        push!(windowed_groups, (window_start, groups))

        current_time += step_size
    end

    return windowed_groups, (min_value, max_value, min_frequency, max_frequency)
end

function determine_peaks(df::DataFrame, window_size::Float64, step_size::Float64, threshold::Float64, column::Symbol=:power, frequency_diff::Float64 = 100.0, group_size_tolerance::Float64 = 100.0)
    # Normalize the specified column
    min_value = minimum(df[!, column])
    max_value = maximum(df[!, column])
    df[!, :normalized_value] = (df[!, column] .- min_value) ./ (max_value - min_value)

    # Normalize the frequency column
    min_frequency = minimum(df[!, :frequency])
    max_frequency = maximum(df[!, :frequency])
    df[!, :normalized_frequency] = (df[!, :frequency] .- min_frequency) ./ (max_frequency - min_frequency)

    # Prepare output
    windowed_groups = []

    # Set initial window parameters based on actual onset values
    current_time = minimum(df.onset)
    end_time = maximum(df.onset)

    while current_time + window_size <= end_time
        window_start = current_time
        window_end = current_time + window_size

        # Filter data within the window
        resonances = filter(row -> row.onset >= window_start && row.onset < window_end, df)

        if isempty(resonances)
            current_time += step_size
            continue
        end

        # Use SciPy's find_peaks to identify peaks
        values = resonances[!, :normalized_value]
        peaks_indices, _ = SciPy.signal.find_peaks(values, height=threshold)

        # Extract peak rows
        peaks = resonances[peaks_indices .+ 1, :]

        # Group peaks based on frequency difference
        sort!(peaks, :normalized_frequency)
        groups = []
        current_group = peaks[1:1, :]

        first_frequency = peaks[1, :normalized_frequency]
        last_frequency = peaks[1, :normalized_frequency]
        for i in 2:nrow(peaks)
            freq = peaks[i, :normalized_frequency]
            if abs(freq - last_frequency) <= frequency_diff && abs(freq - first_frequency) <= group_size_tolerance
                push!(current_group, peaks[i, :])
                last_frequency = freq
            else
                push!(groups, current_group)
                current_group = peaks[i:i, :]
                first_frequency = freq
                last_frequency = freq
            end
        end
        push!(groups, current_group)

        push!(windowed_groups, (window_start, groups))

        current_time += step_size
    end

    return windowed_groups, (min_value, max_value, min_frequency, max_frequency)
end

function process_peaks(windowed_groups)
    processed_output = []

    for (window_start, groups) in windowed_groups
        window_result = []

        for group in groups
            mean_frequency = mean(group[!, :normalized_frequency])
            total_power = maximum(group[!, :normalized_value])
            push!(window_result, (mean_frequency, total_power))
        end

        push!(processed_output, (window_start, window_result))
    end

    return processed_output
end

end