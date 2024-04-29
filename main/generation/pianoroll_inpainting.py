import os
import sys

import librosa

p = os.path.join(os.path.abspath("."), "main")
sys.path.insert(1, p)
print(sys.path)
import copy
import glob

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import soundfile as sf
import torch
from models.diffusion import DDPM, DDPMWrapper, SuperResModel, UNetModel
from models.diffusion.dpm_solver_pytorch import (
    DPM_Solver,
    NoiseScheduleVP,
    model_wrapper,
)
from models.FireflyGAN import FireflyBase
from pytorch_lightning import seed_everything
from tqdm import tqdm
from utils import *


def __parse_str(s):
    split = s.split(",")
    return [int(s) for s in split if s != "" and s is not None]


@hydra.main(config_path="../configs", config_name="train")
def Inpainting(config):
    seed_everything(config.evaluation.seed, workers=True)

    # Load pretrained wrapper
    dim_mults = __parse_str(config.model.dim_mults)
    decoder = SuperResModel(
        in_channels=1,
        model_channels=config.model.dim,
        out_channels=1,
        num_res_blocks=config.model.n_residual,
        channel_mult=dim_mults,
        use_checkpoint=False,
        dropout=config.model.dropout,
    )

    ema_decoder = copy.deepcopy(decoder)
    decoder.eval()
    ema_decoder.eval()

    online_ddpm = DDPM(
        decoder,
        beta_1=config.model.beta1,
        beta_2=config.model.beta2,
        T=config.model.n_timesteps,
        var_type=config.evaluation.variance,
    )
    target_ddpm = DDPM(
        ema_decoder,
        beta_1=config.model.beta1,
        beta_2=config.model.beta2,
        T=config.model.n_timesteps,
        var_type=config.evaluation.variance,
    )

    # in the pretrained DDPM state_dict
    ddpm_wrapper = DDPMWrapper.load_from_checkpoint(
        config.evaluation.chkpt_path,
        online_network=online_ddpm,
        target_network=target_ddpm,
        conditional=True,
        strict=False,
        pred_steps=config.model.n_timesteps,
    )

    device = torch.device(config.evaluation.device)
    n_steps = config.evaluation.n_steps

    # Load test file
    test_npz = glob.glob(os.path.join(config.evaluation.test_npz_path, "*.npz"))

    with torch.no_grad():
        # Load Dpm-Solver
        noise_schedule = NoiseScheduleVP(schedule="linear")
        Unetmodel = ddpm_wrapper.load_model().eval()
        model_fn = model_wrapper(
            Unetmodel.to(device),
            noise_schedule,
            is_cond_classifier=False,
            time_input_type="1",
            total_N=1000,
        )
        dpm_solver = DPM_Solver(model_fn, noise_schedule)

        FFGan = FireflyBase(config.evaluation.vocoder_path)
        FFGan.eval()

        for fnpz in tqdm(test_npz[1:2]):
            file_name = fnpz.strip().split("/")[-1]
            npz = np.load(fnpz)
            mel, pianoroll = npz["m"], npz["p"]
            ori = np.copy(mel)

            # origin data too long
            data_start = config.evaluation.data_start
            data_frame = config.evaluation.data_frame
            npz = np.load(fnpz)
            mel, pianoroll = npz["m"], npz["p"]
            mel = mel[:, data_start : data_start + data_frame]
            pianoroll = pianoroll[:, data_start : data_start + data_frame]

            # preprocess
            mel = normalize(mel, config.data.min_level_db)
            p_max = np.max(pianoroll)
            if p_max > 1:
                pianoroll /= np.max(pianoroll)
            # pianoroll /= 128

            # set inpainting area
            start = 0
            inpainting_length = data_frame
            end = start + inpainting_length

            # get pianoroll
            mel[:, start:end] = pianoroll[:, start:end]
            mel = torch.from_numpy(mel)

            cond = mel.unsqueeze(0).unsqueeze(0)
            x_T = torch.randn_like(cond)

            output = (
                dpm_solver.sample(
                    x_T.to(device),
                    y=None,
                    cond=cond.float().to(device),
                    steps=n_steps,
                    eps=1e-4,
                    adaptive_step_size=False,
                    fast_version=True,
                )
                .squeeze(1)
                .view(1, config.data.num_mels, -1)
            )  # [1, num_mels, mel_length]

            output = output[0][:, :data_frame]
            mel[:, start:end] = output[:, start:end]
            plt.imshow(mel)
            plt.savefig("1.png")

            mel = denormalize(mel, config.data.min_level_db)
            gen_wav = FFGan(mel.unsqueeze(0)).numpy()[0][0]
            sf.write("gen.wav", gen_wav, config.data.sample_rate)

            ori_wav = FFGan(
                torch.from_numpy(
                    ori[:, data_start : data_start + data_frame]
                ).unsqueeze(0)
            ).numpy()[0][0]
            sf.write("ori.wav", ori_wav, config.data.sample_rate)


if __name__ == "__main__":
    Inpainting()
