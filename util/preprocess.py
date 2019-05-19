import sys, os
import os.path as path
import random
import glob
import argparse
import numpy as np
import librosa
import pretty_midi as pm


def wav_preprocess(wav_path, hop_len=512, sr=None, bins_per_pitch=1, mid_low=21, mid_high=109):
    num_pitches, f_min = mid_high - mid_low, librosa.midi_to_hz(mid_low)
    num_bins, bins_per_oct = num_pitches*bins_per_pitch, 12*bins_per_pitch
    data, sr = librosa.load(wav_path, sr=sr)
    cqt = librosa.cqt(data, hop_length=hop_len, n_bins=num_bins, bins_per_octave=bins_per_oct, sr=sr, fmin=f_min)
    return np.abs(cqt)


def mid_preprocess(mid_path, fs=86):
    midi_obj = pm.PrettyMIDI(mid_path)
    piano_roll = midi_obj.get_piano_roll(fs=fs)
    piano_roll = piano_roll > 0.
    piano_roll = piano_roll.astype(float)
    piano_roll = piano_roll[21: 109]
    return piano_roll


def preprocess(midi_path, wav_path, gt_path, cqt_path, fs=86, bins_per_note=1):
    midi_files = glob.glob(path.join(midi_path, '*.midi')) + glob.glob(path.join(midi_path, '*.mid'))
    wav_files = glob.glob(path.join(wav_path, '*.wav'))
    midi_names, wav_names = ([path.splitext(path.basename(p))[0] for p in s] for s in (midi_files, wav_files))
    midi_dict, wav_dict = {n: f for n, f in zip(midi_names, midi_files)}, {n: f for n, f in zip(wav_names, wav_files)}
    midi_names, wav_names = frozenset(midi_names), frozenset(wav_names)
    matched = midi_names | wav_names
    unmatched = midi_names ^ wav_names
    if unmatched:
        print('[WARNING] the following files are unmatched and will be ignored:')
        for f in unmatched:
            print(f, file=sys.stderr)
        sys.stderr.flush()
    label_list = []
    for name in matched:
        print('processing {}'.format(name), file=sys.stderr)
        midi_file, wav_file = midi_dict[name], wav_dict[name]
        cqt_pp_arr = wav_preprocess(wav_file, bins_per_pitch=bins_per_note)
        gt_pp_arr = mid_preprocess(midi_file, fs)
        cqt_len, gt_len = cqt_pp_arr.shape[-1], gt_pp_arr.shape[-1]
        if abs(cqt_len - gt_len) > 0.0001*gt_len:
            # print('[WARNING] {}\n\tgt preprocessed length: {}\n\taudio preprocessed length: {}\n\tresampling'.format(
            #     name, gt_pp_arr.shape[-1], cqt_pp_arr.shape[-1]
            # ), file=sys.stderr, end=' ')
            longer, shorter = (gt_pp_arr, cqt_pp_arr) if gt_len > cqt_len else (cqt_pp_arr, gt_pp_arr)
            step_size = float(longer.shape[-1])/shorter.shape[-1]
            idxs = np.round(np.arange(0, longer.shape[-1] - 1, step_size)).astype(int)
            if gt_len > cqt_len:
                gt_pp_arr = gt_pp_arr[:, idxs]
                # print('gt to {}'.format(gt_pp_arr.shape[-1]), file=sys.stderr)
            else:
                cqt_pp_arr = cqt_pp_arr[:, idxs]
                # print('audio to {}'.format(cqt_pp_arr.shape[-1]), file=sys.stderr)

        gt_file, cqt_file = path.join(gt_path, name + '.gt'), path.join(cqt_path, name + '.cqt')
        np.save(gt_file, gt_pp_arr)
        np.save(cqt_file, cqt_pp_arr)
        label_list.append((gt_file, cqt_file))
        print('gt: {}\tcqt: {}'.format(gt_pp_arr.shape, cqt_pp_arr.shape), file=sys.stderr)
        sys.stderr.flush()

    return label_list


def main(opts):
    midi_path = path.join(os.getcwd(), opts.mid)
    wav_path = path.join(os.getcwd(), opts.wav)
    output_path = path.join(os.getcwd(), opts.output_data)
    gt_path = path.join(output_path, 'gt')
    cqt_path = path.join(output_path, 'cqt')
    for p in (gt_path, cqt_path):
        os.makedirs(p, exist_ok=True)
    sr, fs = opts. sample_rate, opts.spec_freq
    file_pairs = preprocess(midi_path, wav_path, gt_path, cqt_path, bins_per_note=opts.bin_mult)
    with open(path.join(output_path, 'labels.txt'), 'w') as fp:
        for gt, cqt in file_pairs:
            fp.write('{},{}\n'.format(cqt, gt))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--wav", default="data/maestro/wav")
    parser.add_argument("-m", "--mid", default="data/maestro/midi")
    parser.add_argument("-o", "--output_data", default="data/preprocessed/maestro")
    parser.add_argument("-s", "--sample_rate", type=int, default=None)
    parser.add_argument("-f", "--spec_freq", type=int, default=86)
    parser.add_argument("-b", "--bin_mult", type=int, default=1)

    args = parser.parse_args()
    main(args)