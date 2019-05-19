import sys, os
import os.path as path
import glob
import argparse
import numpy as np
import librosa
import pretty_midi as pm


def midi_preprocess(mid_path, fs=86):
    pm.pretty_midi.MAX_TICK = 1e10
    midi_obj = pm.PrettyMIDI(mid_path)
    piano_roll = midi_obj.get_piano_roll(fs=fs)
    piano_roll = piano_roll > 0.
    piano_roll = piano_roll.astype(float)
    piano_roll = piano_roll[21: 109]
    return piano_roll


def main(opts):
    midi_path = path.join(os.getcwd(), opts.midi_file)
    midi_name = path.splitext(path.basename(midi_path))[0]
    output_path = path.join(os.getcwd(), opts.output_path, midi_name + '.npy')
    fs = opts.sample_rate
    piano_roll = midi_preprocess(midi_path, fs=fs)
    np.save(output_path, piano_roll)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("midi_file")
    parser.add_argument("-o", "--output_path", default="gt")
    parser.add_argument("-s", "--sample_rate", type=int, default=None)

    args = parser.parse_args()
    main(args)