# NOTE: f0 has the additional row "ID", so code slightly different
using PlotlyJS, ClusterAnalysis, StatsBase, DataFrames, CSV, LinearAlgebra, Statistics
using PyCall
using Conda
using Statistics, Distributions
using ScikitLearn
using DSP: hamming, filter
using Random
using LinearAlgebra
np = pyimport("numpy")
ENV["PYTHON"]="C:/Users/tugay/Anaconda3/python.exe"
push!(pyimport("sys")."path", "C:/Users/tugay/Desktop/master_clone/Master-Thesis/code/")
push!(pyimport("sys")."path", "./")
kneed = pyimport("kneed")
# import libraries
@sk_import preprocessing: (StandardScaler)
@sk_import metrics: (silhouette_samples, silhouette_score)
@sk_import cluster: (KMeans)

# filename = "flute_syrinx_3"
#filename = "K331-Tri_short"
filename = "CJF0_SA1"

# PATH = "./code/fpt/data/output/scores/" * filename * ".csv"
# PATH_OUTPUT = "./code/fpt/data/output/scores/clustered/" * filename * ".csv"
# PATH_PNG = "./code/fpt/data/output/scores/" * filename * ".png"

#PREFIX = "C:/Users/tugay/Desktop/master_clone/Master-Thesis/code/fpt/data/output/polyphonic/K331-Tri_short.csv"

#PATH = "./code/fpt/data/output/polyphonic/" * filename * ".csv"
#PATH_OUTPUT = "./code/fpt/data/output/polyphonic/" * filename * "_clustered.csv"
#PATH_PNG = "./code/fpt/data/output/polyphonic/" * filename * ".png"

#PATH = "C:/Users/tugay/Desktop/master_clone/Master-Thesis/code/fpt/data/output/polyphonic/" * filename * ".csv"
#PATH_OUTPUT = "C:/Users/tugay/Desktop/master_clone/Master-Thesis/code/fpt/data/output/polyphonic/" * filename * "_clustered.csv"
#PATH_PNG = "C:/Users/tugay/Desktop/master_clone/Master-Thesis/code/fpt/data/output/polyphonic/" * filename * ".png"

PATH = "C:/Users/tugay/Desktop/master_clone/Master-Thesis/code/fpt/data/output/timit/" * filename * ".csv"
PATH_OUTPUT = "C:/Users/tugay/Desktop/master_clone/Master-Thesis/code/fpt/data/output/timit/" * filename * "_clustered.csv"
PATH_PNG = "C:/Users/tugay/Desktop/master_clone/Master-Thesis/code/fpt/data/output/timit/" * filename * ".png"

EPS = 0.05
PTS = 4


function main(path, accuracy)
    raw = DataFrame(CSV.File(path))

    # Additional id column for hierarchical knowledge representation
    raw[!,:id] = collect(1:size(raw)[1])

    # remove the negative resonances to perform machine learning techniques only on the real part 
    pos_raw = filter(:frequency => x -> x > 0, raw)

    # cluster the f0 subset
    #f0_raw = pos_raw[isequal.(pos_raw.f0,1), :]
    #f0_raw = filter(:f0 => isequal(1), f0_raw)
    f0_raw = filter(:power => x -> x > 0.00001, pos_raw)

    #f0_raw = apply_filter(f0_raw, 50)
    f0_raw = harmonic_filter(f0_raw, 200.0, 300.0)

    clustered_f0 = hyperparameterTuning(f0_raw, accuracy)

    for (id, f0) in zip(clustered_f0.id, clustered_f0.f0)
        indices = findall(x -> x == id, pos_raw.id)
        pos_raw[indices, :f0] .= f0
    end

    pos_raw[!, :harmonic] .= -1
    pos_raw[!, :likeliness] .= 0.5

    temp = pos_raw
    max = maximum(clustered_f0.f0)

    # Power denoiser
    for i in 1:max
        avg_power, n_elements = avg_power_cluster(pos_raw, i)

        println("avg power: ", avg_power)
        println("n_elements: ", n_elements)

        if (avg_power < 0.0001 || n_elements < 10)
            temp = temp[temp.f0 .!= i, :]
        end
    end

    for i in 1:maximum(clustered_f0.f0)
        # Harmonics
        pos_raw = overtoneSlice(pos_raw, i)
    end


    CSV.write(PATH_OUTPUT, temp)
    lim_pos_raw = temp[temp.likeliness .<= 1, :]
    overtones_limFreq = lim_pos_raw[lim_pos_raw.frequency .<= 5000, :]
    filter_nonf0 = overtones_limFreq[overtones_limFreq.f0 .!= 0, :]

    #plotharmonic(overtones_limFreq)
    plotf0(filter_nonf0)
end

function avg_power_cluster(df, i)
    cluster = df[df.f0 .== i, :]
    vec_cluster = collect(cluster.power)
    avg_cluster = mean(vec_cluster)
    #avg_power = round(avg_cluster, digits = 2)
    println(avg_cluster)

    n_elements = nrow(cluster)

    return avg_cluster, n_elements
end

function overtoneSlice(df, i)
    f0_value = getf0(df, i)
    f0_resonances = df[df.f0 .== i, :]
    
    min = minimum(f0_resonances.onset)
    max = maximum(f0_resonances.onset)

    println("Cluster ", i, ", min: ",min, ", max: ",max)

    note_slice = findall(x -> min <= x <= max, df.onset)
    df[note_slice, :likeliness] .= (df[note_slice, :].frequency ./ f0_value) .% 1
        
    # Define the parameters for the Gaussian function
    mu = 0.5  # Mean
    sigma = 0.5  # Standard deviation

    # Calculate the Gaussian function values
    gaussian_values = pdf(Normal(mu, sigma), df[note_slice, :likeliness])

    # Multiply the likeliness column with the Gaussian values
    df[note_slice, :likeliness] .= df[note_slice, :likeliness] .* gaussian_values

    # Classify overtones with same id as f0
    likeliness = intersect(findall(x -> min <= x <= max, df.onset), findall(x -> x <= 0.01, df.likeliness))
    df[likeliness, :harmonic] .= i

    return df
end

function getf0(df, i)
    cluster = df[df.f0 .== i, :]
    vec_cluster = collect(cluster.frequency)
    avg_cluster = mean(vec_cluster)
    frequency_pred = round(avg_cluster, digits = 2)
    println(frequency_pred)

    return frequency_pred
end

# Euclidean distance onset/frequency
function featureNormalization(df)
    data = DataFrame(onset=df.onset, frequency=df.frequency)
    mapper = DataFrameMapper([([:onset], StandardScaler()),
                            ([:frequency], StandardScaler())
                            ]);
    mapper = fit_transform!(mapper, copy(data))
end

function silhouetteScore(X, accuracy)
    max_silouette = 0
    best_pts = 0
    best_eps = 0

    knee_eps = knee_epsilonTuning(X)
    
    for min_pts in 3:20 
        for eps in range(0.01, step=0.01, length=accuracy) # TODO: user can adjust accuracy of the algorithm to have more or less notes found!! length is the parameter that will be adjusted in this case
        # Run DBSCAN 
            m = dbscan(X, eps, min_pts); #returns object dbscan!
            # Put labels from clustering back to a dataframe
            cluster_labels = m.labels

            # Ignore tuning where all resonances are labeled as noise
            if (!all(y->y==cluster_labels[1],cluster_labels))
                # Metric for the evaluation of the quality of a clustering technique
                silhouette_avg = silhouette_score(X, cluster_labels)
                # println("for min_pts=", min_pts, "and eps", eps, "the average silhouette_score is :", silhouette_avg)
                if (silhouette_avg > max_silouette)
                    max_silouette = silhouette_avg
                    best_pts = min_pts
                    best_eps = eps
                end
            end
        end
    end


    
    println("--------------")
    println("knee method eps:", knee_eps) 
    println("silhoutte pts:", best_pts)
    println("silhoutte eps:", best_eps)
    println("--------------")

    return best_pts, best_eps, knee_eps

end

function hyperparameterTuning(df, accuracy)
    # no denoise needed
    df[!,:onset_s] = (df.onset ./ df.sample_rate)
    normalize!(df.power, 2)
    # Convert data to a normalized matrix
    df[!,:similarity] = resonanceSimilarity(df)

    X = featureNormalization(df)

    # Calculate Silhouette score and knee
    best_pts, best_eps, knee_eps = silhouetteScore(X, accuracy)

    best_clustering = dbscan(X, best_eps, best_pts); # Best: eps = 0.06, pts = 9

    df[!,:f0] = best_clustering.labels

    return df
end

############################## EXPERIMENTS WITH OTHER NORMALIZATION METHODS #############################
# Test: Euclidean distance between amplitude/decay functions

function onsetNormalization(df)
    data = DataFrame(onset=df.onset)
    mapper = DataFrameMapper([([:onset], StandardScaler())
                            ]);
    mapper = fit_transform!(mapper, copy(data))
end


function similarityNormalization(df)

    formatted_d = map(x -> replace(x, 'j' => "im", '(' => "", ')' => ""), df.d)
    formatted_w = map(x -> replace(x, 'j' => "im", '(' => "", ')' => ""), df.w)

    d = map(x -> parse(ComplexF64, x), formatted_d)
    w = map(x -> parse(ComplexF64, x), formatted_w)


    data = DataFrame(similarity=df.similarity)
    mapper = DataFrameMapper([
                            #[:w], StandardScaler()),
                            #([:frequency], StandardScaler())
                            ([:similarity], StandardScaler())
                            ]);
    mapper = fit_transform!(mapper, copy(data))
end

# Similary distance resonances (cos d_{jk})
function resonanceSimilarity(df)

    formatted_d = map(x -> replace(x, 'j' => "im", '(' => "", ')' => ""), df.d)
    formatted_w = map(x -> replace(x, 'j' => "im", '(' => "", ')' => ""), df.w)

    d = map(x -> parse(ComplexF64, x), formatted_d)
    w = map(x -> parse(ComplexF64, x), formatted_w)

    djdk = map((x,y) -> x.*y, d[1:end-1], d[2:end])
    diff_wjwk = diff(w)


    dj_absPow = abs.(d[1:end-1]).^2
    dk_absPow = abs.(d[2:end]).^2

    gj = df.decay[1:end-1]
    gk = df.decay[2:end]

    numerator = real(djdk./diff_wjwk)

    similarity = numerator ./ (dj_absPow./gj).*(dk_absPow./gk)

    # last element has 0 similarity with first one
    push!(similarity,0)

    similarity[similarity.>=1] .= 1
    similarity[similarity.<=-1] .= -1
    #similarity[.>]

    maximum(similarity) = 0

    #similarity = map(x -> if (x >= 1) x = 0 end, similarity)

    similarity
end

##########################################################################################################


# Experimental setup, did not give appropriate results.
function knee_epsilonTuning(X)
    df_distance = DataFrame([[],[]], ["index", "distance", ])

    l_X = size(X, 1)-1
    for i in 1:l_X
        dist = np.linalg.norm(X[i, :]-X[i+1, :])
        push!(df_distance, [string(i), dist])
    end 
    df_distance = sort!(df_distance, :distance)

    # Knee extraction, Satopaa 2011
    distances = df_distance.distance
    i = 1:length(distances)
    knee = kneed.KneeLocator(i, distances, S=1, curve="convex", direction="increasing", interp_method="polynomial")
    # Returns the epsilon
    distances[knee.knee]
end


############################## PLOT FUNCTIONS ###############################

function plotharmonic(df)
    # non-harmonic data
    trace1 = scatter(
    mode="markers",
    x=df[df.harmonic .== -1, :].onset,
    y=df[df.harmonic .== -1, :].frequency,
    opacity=0.5,
    marker=attr(
        size=2,
        color="646FFB"
    ),
    name="data"
    )

    # harmonic data
    trace2 = scatter(
    mode="markers",
    x=df[df.harmonic .!== -1, :].onset,
    y=df[df.harmonic .!== -1, :].frequency,
    opacity=1,
    marker=attr(
        color=df[df.harmonic .!== -1, :].harmonic,
        size=4
    ),
    name="overtones"
    )

    p = plot([trace1, trace2], Layout(title="Overtone seperation", yaxis_title="Frequency (Hz)", xaxis_title="Time"))
    savefig(p, "overtones.png")

    p
end

function plotf0(df)
    # https://plotly.com/julia/reference/scatter3d/
    p = plot(
        df, 
        Layout(scene = attr(
                        xaxis_title="Time (s)",
                        yaxis_title="Frequency (Hz)",
                        zaxis_title="Power"
                        ),
                        #margin=attr(r=100, b=150, l=50, t=50)
                        ),
        x=:onset, 
        y=:frequency, 
        z=:power, 
        color=:f0, # choose one of the two, dependent on harmonic
        type="scatter3d", 
        mode="markers", 
        marker_size=2
    )

    name = "Clustering of resonances"
    # Default parameters which are used when `layout.scene.camera` is not provided
    camera = attr(
        up=attr(x=0, y=0, z=1),
        center=attr(x=0, y=0, z=0),
        eye=attr(x=-1.55, y=-1.55, z=1.55)
    )
    relayout!(p, scene_camera=camera, title=name)

    # path = "./code/add-ons/plots-demo/"*string(EPS)*"-"*string(PTS)*".png"
    # savefig(p, path)

    # open("./example.html", "w") do io
    #     PlotlyBase.to_html(io, p.plot)
    # end

    p
end

#############################################################################

# Formant analyzer

#############################################################################

# Try to remove noise - keep resonances around harmonics
function harmonic_filter(df::DataFrame, fundamental_freq::Float64, tolerance::Float64)
    harmonics = [fundamental_freq * n for n in 1:10]  # Generate the first 10 harmonics
    filtered_data = filter(row -> any(abs(row[:frequency] - h) <= tolerance for h in harmonics), df)
    return filtered_data
end

function determine_voiced(df::DataFrame, window_size::Int64, step_size::Int64, frequency_threshold::Float64, power_threshold::Float64)
    # Look for the presence of a low frequency band
    boundaries = []
    was_voiced = false

    for j in 0:step_size:(nrow(df) - step_size)
        power_sum = 0
        for i in 1:1:window_size # from i (onset index) start to end of window
            # For each resonance look if frequency is below threshold
            # If below threshold, sum power
            if df[i + j, :frequency] < frequency_threshold
                power_sum += df[i + j, :power]
            end
        end

        is_voiced = power_sum > power_threshold

        if is_voiced != was_voiced
            push!(boundaries, (df[j, :onset], is_voiced))
            was_voiced = is_voiced
        end

    end

    for (index, voiced) in boundaries
        if voiced
            println("Start of voiced band at index: ", index / 16000)
        else
            println("End of voiced band at index: ", index / 16000)
        end
    end

    return
end

# TODO STILL WRONG NOT WINDOWING IN THE RIGTH WAY!!! ROW = ! WINDOW
function analyze_frequencies(df::DataFrame, window_size::Int64, step_size::Int64, threshold::Float64, n_clusters::Int64 = 5, column::Symbol=:power)
    # Return all resonances that are above the threshold
    resonances = DataFrame()
    mean_frequencies = []

    # Normalize the specified column, use relative threshold instead of absolute, such that varying levels do not have an impact
    # notice that it is not properly normalized yet - we could standardize with X = x - x min / x max - x min
    min_value = minimum(df[!, column])
    max_value = maximum(df[!, column])
    df[!, :normalized_value] = (df[!, column] .- min_value) ./ (max_value .- min_value)
    #df[!, :normalized_value] = df[!, column] ./ max_value

    j = 1
    while j <= nrow(df)
        window_start = df[j, :onset]
        window_end = window_start + window_size

        for i in j:nrow(df)
            current_onset = df[i, :onset]
            if current_onset >= window_start && current_onset < window_end
                if df[i, :normalized_value] >= threshold # Select all resonances that exceed threshold
                    push!(resonances, df[i, :])
                end
            end
        end

        # If no resonances above threshold, skip this window
        if isempty(resonances)
            resonances = DataFrame()
            j += step_size
            continue
        end

        if nrow(resonances) < n_clusters
            println("Not enough resonances for clustering in this window.")
            resonances = DataFrame()
            j += step_size
            continue
        end



        # Sort resonances by frequency
        sort!(resonances, :frequency)

        # Prepare columns for clustering
        X = hcat(resonances.frequency)

        # Perform KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters).fit(X, sample_weight=resonances.normalized_value)
        kmeans = KMeans(n_clusters=n_clusters).fit(X)
        labels = kmeans.labels_

        resonances[!, :cluster] = labels

        # Calculate mean to get something like a formant
        mean_freqs = combine(groupby(resonances, :cluster), :frequency => mean => :mean_frequency)
        sort!(mean_freqs, :mean_frequency)
        push!(mean_frequencies, (window_start, mean_freqs))

        # Reset
        resonances = DataFrame()
        j += step_size
    end

    return mean_frequencies
end

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

function extract_formants(onset_vectors::Vector{Any}, tolerance::Float64, min_distance::Float64 = 0.01)
    harmonics_output = []

    for (onset, points) in onset_vectors

        harmonics = extract_f0_and_harmonics(onset_vectors, 0.05)[1][2]
        harmonic_frequencies = [h[1] for h in harmonics]

        F1 = points[argmax(map(x -> x[2], points))]
        detected_formants = [F1]

        filtered_points = []
        for point in points
            if point in detected_formants
                println("Point $(point[1] * 8000) Hz is already in detected_formants.")
                push!(filtered_points, point)
                push!(detected_formants, point)
                continue
            end

            is_far_from_harmonics = all(abs(point[1] - h) > tolerance for h in harmonic_frequencies)
            if !is_far_from_harmonics
                println("Point $(point[1] * 8000) Hz is too close to a harmonic frequency.")
            end

            is_far_from_formants = all(abs(point[1] - formant[1]) >= min_distance for formant in detected_formants)
            if !is_far_from_formants
                println("Point $(point[1] * 8000) Hz is too close to an existing formant.")
            end

            is_above_F1 = point[1] >= F1[1]
            if !is_above_F1
                println("Point $(point[1] * 8000) Hz is below F1's frequency $(F1[1] * 8000) Hz.")
            end

            if is_far_from_harmonics && is_far_from_formants && is_above_F1
                println("Point $(point[1] * 8000) Hz passed all checks and is added.")
                push!(filtered_points, point)
                push!(detected_formants, point)
            else
                println("Filtered out: Frequency $(point[1] * 8000) Hz at onset $(onset / 16000)")
            end
        end

        push!(harmonics_output, (onset, filtered_points))
    end

    return harmonics_output
end

function extract_f0_and_harmonics(onset_vectors::Vector{Any}, tolerance::Float64, max_harmonics::Int = 10)
    harmonics_output = []

    for (onset, points) in onset_vectors
        # Find the lowest frequency point as f0
        lowest_point = points[argmin(map(x -> x[1], points))]
        f0 = lowest_point[1]
        harmonics = [lowest_point]

        # Start searching for harmonics from f0 * 2
        for n in 2:max_harmonics
            harmonic_freq = f0 * n
            found_harmonic = false

            for point in points
                freq = point[1]
                if abs(freq - harmonic_freq) <= tolerance
                    push!(harmonics, point)
                    found_harmonic = true
                    break
                end
            end

            # Stop searching if the current harmonic is not found
            if !found_harmonic
                break
            end
        end

        push!(harmonics_output, (onset, harmonics))
    end

    return harmonics_output
end


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

function select_window(df::DataFrame, begin_onset::Int64, end_onset::Int64)
    window = filter(row -> row[:onset] >= begin_onset && row[:onset] <= end_onset, df)
    return window
end

# accuracy must be a value between 10 and 50, since 0.1 <= eps <= 0.5
# The higher the value, the less accuracy (just inverse for user later), 
# mainly used for pieces where notes vary strongly in time
# note: increase accuracy increases running time as well
#main(PATH, 6) # MOET NOG LAGER VOOR DIE SYNTHETISCHE NUMMERS

df = DataFrame(CSV.File(PATH))

#window = select_window(df, 14850, 20200)
#window = select_window(df, 34715, 36080)
#window = select_window(df, 10200, 11000)
#window = select_window(df, 4559, 5723)
#window = select_window(df, 0, 50000)
#window = select_window(df, 0, 4500)

#debugging
#window = select_window(df, 4559, 5723) # total vowel
#window = select_window(df, 4559, 5001) # segment that works
#window = select_window(df, 5049, 5723) # segment that doesn't work
#window = select_window(df, 5049, 5301)
#window = select_window(df, 5300, 5351)
#window = select_window(df, 5351, 5723)

# VOWELS
#window = select_window(df, 4559, 5723)
#window = select_window(df, 6642, 8772) # hard both for me and praat
#window = select_window(df, 10337, 11517)
#window = select_window(df, 12640, 14714) # ---- problem with threshold, where F3 and F4 are below 0.1 for amplitude
#window = select_window(df, 18088, 20417)
#window = select_window(df, 24229, 25566)
#window = select_window(df, 27156, 28064)
#window = select_window(df, 29660, 31719)
#window = select_window(df, 34715, 36080)
#window = select_window(df, 37556, 39561)
#window = select_window(df, 42059, 43479)

filtered_window = filter(:frequency => x -> x > 0, window)

#plot_frequency_normalized_amplitude(filtered_window)
plot_frequency_normalized_power(filtered_window)
#plot_normalized_frequency_normalized_power(filtered_window)
#plot_frequency_amplitude(window)
#determine_voiced(raw, 800, 800, 300.0, 0.008)

#n_clusters = 5
#resonances = analyze_frequencies(filtered_window, 400, 100, 0.01, n_clusters, :power) # 0.00005 threshold for absolute power value, step_size and window that seemingly worked 800 - 800
#plot_analyzed_frequencies(filtered_window, resonances)

#grouped_resonances, normalization_factors = identify_peaks(filtered_window, 400., 200.0, 0.01, :power, 0.01, 0.01)
grouped_resonances, normalization_factors = identify_peaks(filtered_window, 400., 200.0, 0.1, :amplitude, 0.01, 0.01)

processed_resonances = process_peaks(grouped_resonances)

#harmonics = extract_f0_and_harmonics(processed_resonances, 0.05)

formants = extract_formants(processed_resonances, 0.1)

#plot_spectrum(processed_resonances, normalization_factors)

#plot_grouped_resonances(grouped_resonances, normalization_factors)

print_reconstructed_values(formants, normalization_factors)