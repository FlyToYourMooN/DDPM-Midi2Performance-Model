import json
import os
import pickle
import sys

import h5py
import hydra
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
from hydra.utils import get_original_cwd
from intervaltree import Interval, IntervalTree
from tqdm import tqdm

p = os.path.join(os.path.abspath("."), "main")
sys.path.insert(1, p)
from utils import LogMelSpectrogram


def process_mel(wav):
    wav = torch.from_numpy(wav).unsqueeze(0).float()
    Mel_transform = LogMelSpectrogram()
    mel = Mel_transform(wav)
    mel = mel.squeeze(0).numpy()
    return mel


def process_pianoroll(pianoroll, length, hop_length):
    pianoroll_result = np.zeros((length, 128))
    t = IntervalTree()

    for j in pianoroll:
        t[int(j.begin) : int(j.end)] = j.data

    for window in range(pianoroll_result.shape[0]):
        # For score, set all notes to 1 if they are played at this window timestep
        labels = t[window * hop_length]
        for label in labels:
            pianoroll_result[window, label.data[1]] = label.data[0]

    return pianoroll_result.T


def process_data(config, pianoroll_list, wav_list):
    num_musics = len(pianoroll_list)
    hop = config.musicnet.sample_rate // config.data.hop_length

    for i in tqdm(range(num_musics)):
        mel = process_mel(wav_list[i])
        pianoroll = process_pianoroll(
            pianoroll_list[i], len(mel[0]), config.data.hop_length
        )

        lens = min(len(pianoroll[0]), len(mel[0]))
        mel = mel[:, :lens]
        pianoroll = pianoroll[:, :lens]

        # data save path
        test_save_path = os.path.join(config.musicnet.save_path, "test")
        train_save_path = os.path.join(config.musicnet.save_path, "train")
        os.makedirs(test_save_path, exist_ok=True)
        os.makedirs(train_save_path, exist_ok=True)

        # test set
        if i == num_musics - 1:
            test_save_path = os.path.join(test_save_path, "test.npz")
            np.savez(test_save_path, m=mel, p=pianoroll)

        # train set, split data for training
        else:
            lens = len(mel[0])
            frame = config.data.split_frame
            # split
            t = 0
            for j in range(frame, lens, frame):
                m = mel[:, j - frame : j]
                p = pianoroll[:, j - frame : j]
                save_path = os.path.join(train_save_path, f"{t}.npz")
                np.savez(save_path, m=m, p=p)
                t += 1


@hydra.main(config_path="../configs", config_name="train")
def Main(config):
    # load music type
    original_cwd = get_original_cwd()
    json_path = os.path.join(original_cwd, "main/configs/musicnet_type.json")
    with open(json_path, "r") as file:
        data = json.load(file)

    music_type = config.musicnet.music_type
    music_ids = data[music_type]

    # load musicnet dataset
    dataset = np.load(
        open("/disk2/MusicNet/musicnet.npz", "rb"), encoding="latin1", allow_pickle=True
    )

    pianoroll_list, wav_list = [], []
    for music_id in music_ids:
        wav, pianoroll = dataset[str(music_id)]
        wav_list.append(wav)
        pianoroll_list.append(pianoroll)

    # extract data
    process_data(config, pianoroll_list, wav_list)


if __name__ == "__main__":
    Main()
