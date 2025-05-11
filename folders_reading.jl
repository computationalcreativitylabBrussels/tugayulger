module FoldersReading

using CSV
using UUIDs
using DataFrames
include("clustering.jl")
using .Clustering

FOLDER_PATH = "C:/Users/tugay/Desktop/pilot-project/TIMIT/data/lisa/data/timit/raw/output_sentence/fpt_outputs"

META_DATA = "C:/Users/tugay/Desktop/pilot-project/TIMIT/data/lisa/data/timit/raw/output_sentence/metadata.csv"

FOLDERS_WAV_PATH = "C:/Users/tugay/Desktop/pilot-project/TIMIT/data/lisa/data/timit/raw/output_sentence/wav_files"

PRAAT_EXE = "C:/Users/tugay/Desktop/praat6427_win-intel64/praat.exe"

PRAAT_SCRIPT = "C:/Users/tugay/Desktop/pilot-project/TIMIT/data/lisa/data/timit/raw/output_sentence/wav_files/formant.praat"

OUTPUT_CSV = "C:/Users/tugay/Desktop/pilot-project/TIMIT/data/lisa/data/timit/raw/output_sentence/formants.csv"

OUTPUT_OWN_CSV = "C:/Users/tugay/Desktop/pilot-project/TIMIT/data/lisa/data/timit/raw/output_sentence/formants_own.csv"

OUTPUT_CSV_3 = "C:/Users/tugay/Desktop/pilot-project/TIMIT/data/lisa/data/timit/raw/output_sentence/formants_3.csv"

OUTPUT_OWN_CSV_3 = "C:/Users/tugay/Desktop/pilot-project/TIMIT/data/lisa/data/timit/raw/output_sentence/formants_own_3.csv"


OUTPUT_PEAKS_CSV = "C:/Users/tugay/Desktop/pilot-project/TIMIT/data/lisa/data/timit/raw/output_sentence/peaks.csv"

const VOWELS = [
    "iy", "ih", "eh", "ey", "ae", "aa", "aw", "ay",
    "ah", "ao", "oy", "ow", "uh", "uw", "ux", "er",
    "ax", "ix", "axr", "ax-h"
]

function process_metadata_and_phn_to_formants_with_praat(
    window_size::Int,  # Window size in samples
    step_size::Int,    # Step size in samples
    metadata_path::String,
    folder_path::String,
    folder_wav_path::String,
    praat_path::String,
    script_path::String,
    output_csv_path::String
)
    # Read metadata
    metadata_df = CSV.read(metadata_path, DataFrame)

    # Open the output CSV file for writing
    open(output_csv_path, "w") do file
        # Write the header
        write(file, "Speaker_ID,Sentence_ID,WAV_Path,Onset_Start,Onset_End,Formant_Frequency,Formant_Power\n")

        # Process each row in the metadata
        for row in eachrow(metadata_df)
            speaker_id = row.Speaker_ID
            sentence_id = row.Sentence_ID

            # Construct file paths
            phn_file_path = joinpath(folder_path, "$(speaker_id)_$sentence_id", "$sentence_id.PHN")
            wav_file_path = joinpath(folder_wav_path, "$(speaker_id)_$sentence_id", "$(speaker_id)_$sentence_id.wav")
            wav_file_path = replace(wav_file_path, "\\" => "/")

            # Read PHN file
            phn_data = CSV.File(phn_file_path, delim=' ', header=false) |> DataFrame
            rename!(phn_data, [:onset_start, :onset_end, :label])

            # Filter for vowel rows
            vowel_rows = filter(row -> row.label in VOWELS, eachrow(phn_data))
            vowel_ranges = [(r.onset_start, r.onset_end) for r in vowel_rows]  # Keep raw sample points

            # Process each vowel range
            for (begin_onset, end_onset) in vowel_ranges
                current_onset = ceil(begin_onset / 50) * 50
                while current_onset + window_size <= end_onset
                    try
                        # Convert sample points to seconds
                        window_start_sec = current_onset / 16000.0
                        window_end_sec = (current_onset + window_size) / 16000.0

                        # Temporary file for Praat output
                        tmp_out_file = tempname() * ".txt"

                        # Construct Praat command
                        cmd = `$(praat_path) --run $(script_path) "$(wav_file_path)" $(window_start_sec) $(window_end_sec) "$(tmp_out_file)"`
                        run(cmd)

                        # Read and process Praat output
                        output = read(tmp_out_file, String)
                        rm(tmp_out_file, force=true)

                        if !isempty(output)
                            formant_strings = split(strip(output), ',')
                            formants = tryparse.(Float64, formant_strings)

                            # Ensure formants have exactly 4 entries
                            if length(formants) != 4
                                println("Unexpected number of formants returned for range ($window_start_sec, $window_end_sec). Skipping.")
                                current_onset += step_size
                                continue
                            end

                            # Write formants to the output CSV
                            for frequency in formants
                                write(file, "$(speaker_id),$(sentence_id),$(wav_file_path),$(window_start_sec),$(window_end_sec),$(frequency),1.0\n")
                            end
                        else
                            println("Empty output for range ($window_start_sec, $window_end_sec). Skipping this range.")
                        end
                    catch e
                        println("Error processing range ($current_onset, $(current_onset + window_size)): $e")
                    end

                    # Move to the next window
                    current_onset += step_size
                end
            end
        end
    end
end

function process_metadata_and_phn_to_csv(window_size::Float64, step_size::Float64, metadata_path::String, folder_path::String, output_csv_path::String)
    metadata_df = CSV.read(metadata_path, DataFrame)

    open(output_csv_path, "w") do file
        # Write the header
        write(file, "Speaker_ID,Sentence_ID,WAV_Path,Onset_Start,Onset_End,Formant_Frequency,Formant_Power\n")

        for row in eachrow(metadata_df)
            speaker_id = row[:Speaker_ID]
            sentence_id = row[:Sentence_ID]

            phn_file_path = joinpath(folder_path, "$(speaker_id)_$sentence_id", "$sentence_id.PHN")
            csv_file_path = joinpath(folder_path, "$(speaker_id)_$sentence_id", "$(speaker_id)_$sentence_id.csv")
            wav_file_path = joinpath(folder_path, "$(speaker_id)_$sentence_id", "$(speaker_id)_$sentence_id.wav")

            phn_data = CSV.File(phn_file_path, delim=' ', header=false) |> DataFrame
            rename!(phn_data, [:onset_start, :onset_end, :label])

            vowel_rows = filter(row -> row[:label] in VOWELS, phn_data)
            vowel_ranges = [(row[:onset_start], row[:onset_end]) for row in eachrow(vowel_rows)]

            # Process formants
            formants = Clustering.process_windows(csv_file_path, vowel_ranges, window_size, step_size)

            for (onset, formant_data) in formants
                for (frequency, power) in formant_data
                    onset_start = onset / 16000
                    onset_end = (onset + window_size) / 16000
                    write(file, "$(speaker_id),$(sentence_id),$(wav_file_path),$(onset_start),$(onset_end),$(frequency * 5000),$(power)\n")
                end
            end
        end
    end
end


function process_metadata_and_phn(metadata_path::String, folder_path::String)
    metadata_df = CSV.read(metadata_path, DataFrame)

    vowel_onset_ranges = []

    for row in eachrow(metadata_df)
        speaker_id = row[:Speaker_ID]
        sentence_id = row[:Sentence_ID]

        phn_file_path = joinpath(folder_path, "$(speaker_id)_$sentence_id", "$sentence_id.PHN")

        phn_data = CSV.File(phn_file_path, delim=' ', header=false) |> DataFrame
        rename!(phn_data, [:onset_start, :onset_end, :label])

        vowel_rows = filter(row -> row[:label] in VOWELS, phn_data)

        vowel_ranges = [(row[:onset_start], row[:onset_end]) for row in eachrow(vowel_rows)]

        append!(vowel_onset_ranges, vowel_ranges)
    end

    return metadata_df, vowel_onset_ranges
end

function process_metadata_and_phn_to_formants(metadata_path::String, folder_path::String)
    metadata_df = CSV.read(metadata_path, DataFrame)

    all_formants = []

    for row in eachrow(metadata_df)
        speaker_id = row[:Speaker_ID]
        sentence_id = row[:Sentence_ID]

        phn_file_path = joinpath(folder_path, "$(speaker_id)_$sentence_id", "$sentence_id.PHN")

        csv_file_path = joinpath(folder_path, "$(speaker_id)_$sentence_id", "$(speaker_id)_$sentence_id.csv")

        phn_data = CSV.File(phn_file_path, delim=' ', header=false) |> DataFrame
        rename!(phn_data, [:onset_start, :onset_end, :label])

        vowel_rows = filter(row -> row[:label] in VOWELS, phn_data)

        vowel_ranges = [(row[:onset_start], row[:onset_end]) for row in eachrow(vowel_rows)]

        #for (begin_onset, end_onset) in vowel_ranges
        #    formants = Clustering.process(phn_file_path, begin_onset, end_onset)
        #    append!(all_formants, formants)
        #end
        formants = Clustering.process(csv_file_path, vowel_ranges)

        if length(formants) != 4 * length(vowel_ranges)
            println("Warning: Mismatch in formant count for $speaker_id $sentence_id")
            continue
        end
        append!(all_formants, formants)
    end

    return metadata_df, all_formants
end


#metadata_df, all_formants = process_metadata_and_phn_to_formants(META_DATA, FOLDER_PATH)

#println("Metadata DataFrame:")
#println(metadata_df)

#println("\nAll Formants:")
#println(all_formants)

#process_metadata_and_phn_to_formants_with_praat(400, 200, META_DATA, FOLDER_PATH, FOLDERS_WAV_PATH,PRAAT_EXE, PRAAT_SCRIPT, OUTPUT_CSV)
process_metadata_and_phn_to_csv(400.0, 200.0,META_DATA, FOLDER_PATH, OUTPUT_OWN_CSV)

end