import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pretty_midi

p = os.path.join(os.path.abspath("."), "main")
sys.path.insert(1, p)
from glob import glob

import hydra
import torch
import torchaudio
from tqdm import tqdm
from utils import LogMelSpectrogram


def get_pianoroll_and_mel(config, wav_path):
    midi_path = wav_path.replace(".wav", ".midi")
    # get midi and mel
    wav, sr = torchaudio.load(wav_path)  # mono

    wav = torchaudio.functional.resample(
        wav[0], sr, config.sample_rate.sample_rate
    ).unsqueeze(0)

    Mel_transform = LogMelSpectrogram()
    mel = Mel_transform(wav)
    mel = mel.squeeze(0).numpy()

    midi = pretty_midi.PrettyMIDI(midi_path)
    # get pianoroll
    pianoroll = midi.get_piano_roll(
        fs=config.sample_rate.sample_rate // config.data.hop_length
    )
    # align length
    lens = min(len(pianoroll[0]), len(mel[0]))
    mel = mel[:, :lens]
    pianoroll = pianoroll[:, :lens]
    # trim
    trim_f = 0
    trim_o = 2400
    mel = mel[:, trim_f:-trim_o]
    pianoroll = pianoroll[:, trim_f:-trim_o]
    return mel, pianoroll


def get_npz(config, wav_paths, data_type):
    for i in tqdm(range(len(wav_paths))):
        filename = wav_paths[i].split("/")[-1]
        data_save_path = os.path.join(config.mastero.save_path, data_type)
        os.makedirs(data_save_path, exist_ok=True)

        mels, pianorolls = get_pianoroll_and_mel(config, wav_paths[i])

        # for test, we dont need to split data
        if data_type == "test":
            save_path = os.path.join(data_save_path, f"{filename}.npz")
            np.savez(save_path, m=mels, p=pianorolls)

        else:
            lens = len(mels[0])
            frame = config.data.split_frame
            # split
            t = 0
            for j in range(frame, lens, frame):
                mel = mels[:, j - frame : j]
                pianoroll = pianorolls[:, j - frame : j]
                save_path = os.path.join(data_save_path, f"{i}_{t}.npz")
                np.savez(save_path, m=mel, p=pianoroll)
                t += 1


@hydra.main(config_path="../configs", config_name="train")
def Main(config):
    wav_paths = glob(os.path.join(config.mastero.data_path, "**", "*.wav"))
    train_rate = config.mastero.train_rate
    val_rate = config.mastero.val_rate

    data_nums = len(wav_paths)
    train_nums = int(data_nums * train_rate)
    val_nums = int(data_nums * val_rate)
    test_nums = data_nums - train_nums - val_nums

    get_npz(config, wav_paths[:train_nums], "train")
    get_npz(config, wav_paths[train_nums : train_nums + val_nums], "val")
    get_npz(config, wav_paths[-1 * test_nums :], "test")


if __name__ == "__main__":
    Main()
