import os
import sys
import csv
import librosa
import soundfile as sf
import pandas as pd
from matplotlib import pyplot as plt
from os.path import exists
import traceback

sys.path.insert(0, './code/fpt')
from utilities_store import get_spectrogram, plot_spectrogram, get_f0, plot_f0
import serialization as ser

ACCURACY = 500 # should be a power of 2
STEP_SIZE = 50
MAX_FREQ = 2000

PREFIXES = ["SA", "SX", "SI"]

def read_speaker_info(speaker_info_file):
    speaker_info = {}
    with open(speaker_info_file, "r") as f:
        lines = f.readlines()
    for line in lines:
        if not line.startswith(";") and line.strip():
            parts = line.strip().split()
            speaker_id = parts[0]
            gender = parts[1]
            speaker_info[speaker_id] = gender
    return speaker_info

def read_speaker_sentences(speaker_mapping_file):
    speaker_sentences = {}
    with open(speaker_mapping_file, "r") as f:
        lines = f.readlines()
    for line in lines:
        if not line.startswith(";"):
            parts = line.strip().split()
            speaker_id = parts[0]
            sentence_ids = parts[1:]
            speaker_sentences[speaker_id] = sentence_ids
    return speaker_sentences

def process_sentence(wav_file, wav_output_dir, fpt_output_dir, speaker_id, dialect, gender, sentence_id):
    y, sr = librosa.load(wav_file, sr=None)
    basename = f"{speaker_id}_{sentence_id}"
    output_wav = os.path.join(wav_output_dir, basename + ".wav")
    sf.write(output_wav, y, sr)
    print("Saved WAV:", output_wav)
    run_fpt(output_wav, fpt_output_dir, basename)

def run_fpt(input_wav, fpt_output_dir, output_prefix):
    csv_path = os.path.join(fpt_output_dir, output_prefix + ".csv")
    print("FPT on:", input_wav, "| CSV:", csv_path)
    if not exists(input_wav):
        print("WAV not found:", input_wav)
        return

    import soundfile as sf
    import numpy as np
    signal, sr = sf.read(input_wav)
    """
    if len(signal) < ACCURACY:
        signal = np.pad(signal, (0, ACCURACY - len(signal)), mode='constant')
        sf.write(input_wav, signal, sr)
    elif len(signal) > ACCURACY:
        signal = signal[:ACCURACY]
        sf.write(input_wav, signal, sr)"""

    try:
        spectrogram, signal = get_spectrogram(
            file=input_wav,
            folder='absolute',
            N=ACCURACY,
            step_size=STEP_SIZE,
            power_threshold=1e-9,
            amp_threshold=1e-8
        )
    except Exception as e:
        print(f"Error processing {input_wav}: {e}")
        print("input wav: ", input_wav, "fpt output dir: ", fpt_output_dir, "output_prefix: ", output_prefix)
        traceback.print_exc()
        return

    plot_spectrogram(spectrogram, max_freq=MAX_FREQ, scale=100)
    ser.to_csv(spectrogram, output_prefix, fpt_output_dir)
    plt.savefig(os.path.join(fpt_output_dir, output_prefix + "_base"))

    word_boundary_file = os.path.join(fpt_output_dir, output_prefix + ".wrd")
    syllable_boundary_file = os.path.join(fpt_output_dir, output_prefix + ".phn")

    if exists(word_boundary_file) and exists(syllable_boundary_file):
        word_boundaries = pd.read_csv(word_boundary_file, sep=' ', header=None, names=['start_time', 'end_time', 'word'])
        syllable_boundaries = pd.read_csv(syllable_boundary_file, sep=' ', header=None, names=['start_time', 'end_time', 'syllable'])

        sample_rate = 16000
        word_boundaries['start_time_sec'] = word_boundaries['start_time'] / sample_rate
        word_boundaries['end_time_sec'] = word_boundaries['end_time'] / sample_rate
        syllable_boundaries['start_time_sec'] = syllable_boundaries['start_time'] / sample_rate
        syllable_boundaries['end_time_sec'] = syllable_boundaries['end_time'] / sample_rate

        for index, row in word_boundaries.iterrows():
            plt.axvline(x=row['start_time_sec'], color='blue', linestyle='--', label='Word Boundary' if index == 0 else "")
            plt.axvline(x=row['end_time_sec'], color='blue', linestyle='--')
            plt.text((row['start_time_sec'] + row['end_time_sec']) / 2, 0.95, row['word'], color='blue', fontsize=12, ha='center', va='top', transform=plt.gca().get_xaxis_transform())

        for index, row in syllable_boundaries.iterrows():
            plt.axvline(x=row['start_time_sec'], color='red', linestyle=':', label='Syllable Boundary' if index == 0 else "")
            plt.axvline(x=row['end_time_sec'], color='red', linestyle=':')
            plt.text((row['start_time_sec'] + row['end_time_sec']) / 2, 0.90, row['syllable'], color='red', fontsize=10, ha='center', va='top', transform=plt.gca().get_xaxis_transform())

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())

        plt.savefig(os.path.join(fpt_output_dir, output_prefix + "_annotated"))
        plt.close()

    try:
        f0 = get_f0(spectrogram)
        df = pd.read_csv(csv_path)
        df['f0'] = [1 if frequency in f0[0] else 0 for frequency in df['frequency']]
        df.to_csv(csv_path, index=False)
    except Exception as e:
        print(f"Error updating CSV for {input_wav}: {e}")
        traceback.print_exc()

def process_all_dialects(timit_root, speaker_info_file, speaker_mapping_file, wav_output_dir, fpt_output_dir, metadata_file):
    speaker_info = read_speaker_info(speaker_info_file)
    speaker_sentences = read_speaker_sentences(speaker_mapping_file)
    dialect_folders = [f for f in os.listdir(timit_root) if f.startswith("DR")]
    metadata = []

    total_dialects = len(dialect_folders)

    for idx, dialect in enumerate(dialect_folders):
        print("-" * 80)
        print(f"Dialect {idx+1} of {total_dialects}")
        dialect_path = os.path.join(timit_root, dialect)
        speaker_folders = os.listdir(dialect_path)
        if speaker_folders:
            speaker_folder = speaker_folders[0]  # Process only the first speaker
            gender = speaker_folder[0]
            speaker_id = speaker_folder[1:]
            if speaker_id in speaker_info and speaker_info[speaker_id] == gender:
                print(f"Processing speaker {speaker_id} ({gender}) in dialect {dialect}")
                speaker_path = os.path.join(dialect_path, speaker_folder)
                if speaker_id not in speaker_sentences or len(speaker_sentences[speaker_id]) != 10:
                    print(f"Skipping {speaker_id}, incorrect number of sentence IDs")
                    continue
                sa_sentences = speaker_sentences[speaker_id][:2]
                sx_sentences = speaker_sentences[speaker_id][2:7]
                si_sentences = speaker_sentences[speaker_id][7:10]
                all_sentences = (
                    [(f"SA{sa_sentences[i]}", sa_sentences[i]) for i in range(2)] +
                    [(f"SX{sx_sentences[i]}", sx_sentences[i]) for i in range(5)] +
                    [(f"SI{si_sentences[i]}", si_sentences[i]) for i in range(3)]
                )

                for full_sentence_id, sentence_id in all_sentences:
                    wav_file = os.path.join(speaker_path, f"{full_sentence_id}.WAV")
                    if os.path.exists(wav_file):
                        process_sentence(
                            wav_file,
                            wav_output_dir,  # Save wav files here
                            fpt_output_dir,  # FPT outputs are stored separately
                            speaker_id, dialect, gender, full_sentence_id
                        )
                        metadata.append([speaker_id, gender, dialect, full_sentence_id])
                    else:
                        print(f"Skipping {full_sentence_id} for {speaker_id} (missing files)")     
                    break # Process only the first sentence
            break  # Process only the first speaker
        break # Process only the first dialect

    with open(metadata_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Speaker ID", "Gender", "Dialect", "Sentence ID"])
        writer.writerows(metadata)
    print(f"Metadata saved to {metadata_file}")

def main():
    TIMIT_ROOT = "C:/Users/tugay/Desktop/pilot-project/TIMIT/data/lisa/data/timit/raw/TIMIT/TRAIN"
    SPEAKER_INFO_FILE = "C:/Users/tugay/Desktop/pilot-project/TIMIT/data/lisa/data/timit/raw/TIMIT/DOC/SPKRINFO.TXT"
    SPEAKER_MAPPING_FILE = "C:/Users/tugay/Desktop/pilot-project/TIMIT/data/lisa/data/timit/raw/TIMIT/DOC/SPKRSENT.TXT"
    OUTPUT_DIR = "C:/Users/tugay/Desktop/pilot-project/TIMIT/data/lisa/data/timit/raw/output_sentence"
    METADATA_FILE = os.path.join(OUTPUT_DIR, "metadata.csv")

    wav_output_dir = os.path.join(OUTPUT_DIR, "wav_files")
    fpt_output_dir = os.path.join(OUTPUT_DIR, "fpt_outputs")
    os.makedirs(wav_output_dir, exist_ok=True)
    os.makedirs(fpt_output_dir, exist_ok=True)

    process_all_dialects(
        timit_root=TIMIT_ROOT,
        speaker_info_file=SPEAKER_INFO_FILE,
        speaker_mapping_file=SPEAKER_MAPPING_FILE,
        wav_output_dir=wav_output_dir,
        fpt_output_dir=fpt_output_dir,
        metadata_file=METADATA_FILE
    )

if __name__ == "__main__":
    main()