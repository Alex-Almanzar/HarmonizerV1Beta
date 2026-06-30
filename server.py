"""
Harmony Generator -- v1 (simplified diatonic voicing)
------------------------------------------------------
Generates multiple harmony parts from a single melody/vocal recording.

Supports:
  - "trio"           : Soprano (3rd up) / Alto (5th up) / Tenor (3rd down)
  - "mens_quartet"    : First Tenor / Second Tenor (lead) / Baritone / Bass
  - "womens_quartet"  : Soprano / Mezzo-Soprano / Alto / Melody (lead)

This version uses SEGMENT-based pitch shifting (not one global average
shift like a naive first pass would) so each part of the song gets the
correct interval as the melody moves through the scale. It still uses
the ORIGINAL singer's timbre -- true distinct-voice timbre swapping is
a separate later stage (voice conversion / SVC), not part of this file.

The voicing offsets below are simplified diatonic intervals, not real
barbershop / close-harmony voice-leading rules. Swap VOICING_PRESETS
(or make it chord-position-aware) once you're ready to refine accuracy.
"""

import os
import numpy as np
import librosa
import soundfile as sf
from pedalboard import Pedalboard, PitchShift

# ---------------------------------------------------------------------------
# CONFIG -- edit these for your run
# ---------------------------------------------------------------------------
INPUT_VOCALS = r"C:\AI_Audio\input_vocals.flac"
OUTPUT_DIR = r"C:\AI_Audio\Harmonies"
SONG_KEY = "G"
SCALE_TYPE = "major"  # "major" or "minor"
VOICING = "trio"  # "trio", "mens_quartet", or "womens_quartet"

os.makedirs(OUTPUT_DIR, exist_ok=True)

NOTE_MAP = {
    'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5,
    'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11,
}

# ---------------------------------------------------------------------------
# VOICING PRESETS
# ---------------------------------------------------------------------------
# Each part maps to a scale-degree offset relative to the melody's
# current scale degree:
#   0  = unison (passthrough / lead sings the melody as-is)
#   2  = a 3rd up      -2 = a 3rd down
#   4  = a 5th up       -4 = a 5th down
VOICING_PRESETS = {
    "trio": {
        "soprano_3rd_up": 2,
        "alto_5th_up": 4,
        "tenor_3rd_down": -2,
    },
    "mens_quartet": {  # Tenor / Lead / Baritone / Bass
        "first_tenor": 2,
        "second_tenor_lead": 0,  # melody itself, passthrough
        "baritone": -2,
        "bass": -4,
    },
    "womens_quartet": {  # Soprano / Mezzo / Alto / Melody (Lead)
        "soprano": 2,
        "mezzo_soprano": -2,
        "alto": -4,
        "melody_lead": 0,  # melody itself, passthrough
    },
}


def get_scale_pitches(key, scale_type):
    root = NOTE_MAP[key.upper()]
    intervals = [0, 2, 4, 5, 7, 9, 11] if scale_type == "major" else [0, 2, 3, 5, 7, 8, 10]
    return [(root + i) % 12 for i in intervals]


def analyze_melody(vocal_file):
    """Load audio once and extract pitch info shared by every harmony part."""
    y, sr = librosa.load(vocal_file, sr=None)
    f0, voiced_flag, _ = librosa.pyin(
        y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C6'), sr=sr
    )
    return y, sr, f0, voiced_flag


def per_frame_shifts(f0, voiced_flag, scale_pitches, degree_offset):
    """
    Compute the semitone shift needed, frame by frame, to move the melody
    to the target scale-degree offset.

    Returns an array the same length as f0, with np.nan for
    unvoiced/unpitched frames.
    """
    if degree_offset == 0:
        return np.zeros(len(f0))  # passthrough part (lead/melody)

    shifts = np.full(len(f0), np.nan)
    for i, f in enumerate(f0):
        if np.isnan(f) or not voiced_flag[i] or f <= 0:
            continue

        current_note = int(round(librosa.hz_to_midi(f)))
        current_pc = current_note % 12

        if current_pc not in scale_pitches:
            continue  # off-scale note (vibrato/noise) -- leave unshifted

        scale_idx = scale_pitches.index(current_pc)
        target_idx = (scale_idx + degree_offset) % 7
        target_pc = scale_pitches[target_idx]

        semitone_shift = target_pc - current_pc
        if degree_offset > 0 and semitone_shift < 0:
            semitone_shift += 12
        elif degree_offset < 0 and semitone_shift > 0:
            semitone_shift -= 12

        shifts[i] = semitone_shift

    return shifts


def render_harmony(y, sr, shifts, hop_length=512, chunk_seconds=0.25, crossfade_seconds=0.05):
    """
    Render a harmony part using SEGMENT-based pitch shifting: the audio is
    broken into small chunks, each chunk is shifted by the most common
    semitone value within it, and chunks are crossfaded together to avoid
    clicks at the boundaries. This is what actually fixes the original
    "one average shift for the whole song" bug.
    """
    frame_times = librosa.frames_to_time(np.arange(len(shifts)), sr=sr, hop_length=hop_length)
    chunk_samples = int(chunk_seconds * sr)
    crossfade_samples = int(crossfade_seconds * sr)

    output = np.zeros(len(y) + chunk_samples)
    pos = 0

    while pos < len(y):
        end = min(pos + chunk_samples, len(y))
        chunk = y[pos:end]
        if len(chunk) == 0:
            break

        t_start, t_end = pos / sr, end / sr
        in_range = (frame_times >= t_start) & (frame_times < t_end)
        chunk_shifts = shifts[in_range]
        chunk_shifts = chunk_shifts[~np.isnan(chunk_shifts)]

        if len(chunk_shifts) > 0:
            values, counts = np.unique(np.round(chunk_shifts), return_counts=True)
            semitone = float(values[np.argmax(counts)])
        else:
            semitone = 0.0  # unvoiced chunk -- leave pitch unchanged

        if semitone != 0.0:
            board = Pedalboard([PitchShift(semitones=semitone)])
            shifted_chunk = board(chunk.astype(np.float32), sr)
            if shifted_chunk.ndim > 1:
                shifted_chunk = shifted_chunk[0]
        else:
            shifted_chunk = chunk

        fade_len = min(crossfade_samples, len(shifted_chunk))
        if fade_len > 0:
            shifted_chunk = shifted_chunk.copy()
            shifted_chunk[:fade_len] *= np.linspace(0, 1, fade_len)

        usable = min(len(shifted_chunk), len(output) - pos)
        output[pos:pos + usable] += shifted_chunk[:usable]

        pos += chunk_samples - crossfade_samples

    return output[:len(y)]


def build_harmony_parts(vocal_file, key, scale_type, voicing):
    if voicing not in VOICING_PRESETS:
        raise ValueError(f"Unknown voicing '{voicing}'. Choose from: {list(VOICING_PRESETS)}")

    y, sr, f0, voiced_flag = analyze_melody(vocal_file)
    scale_pitches = get_scale_pitches(key, scale_type)

    parts = {}
    for part_name, degree_offset in VOICING_PRESETS[voicing].items():
        print(f"Computing {part_name} (offset {degree_offset})...")
        shifts = per_frame_shifts(f0, voiced_flag, scale_pitches, degree_offset)
        parts[part_name] = render_harmony(y, sr, shifts)

    return parts, sr


if __name__ == "__main__":
    parts, sr = build_harmony_parts(INPUT_VOCALS, SONG_KEY, SCALE_TYPE, VOICING)

    for part_name, audio in parts.items():
        out_path = os.path.join(OUTPUT_DIR, f"{VOICING}_{part_name}.flac")
        sf.write(out_path, audio, sr, format="FLAC")
        print(f"Saved {out_path}")

    print(f"All harmony parts generated for '{VOICING}' inside {OUTPUT_DIR}")
