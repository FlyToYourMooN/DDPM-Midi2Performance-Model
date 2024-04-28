
import os
import librosa
import sys
p = os.path.join(os.path.abspath("."), "main")
sys.path.insert(1, p)
print(sys.path)
import glob
from models.diffusion.dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
import copy
import hydra
import pytorch_lightning as pl
import torch
from tqdm import tqdm
from models.diffusion import DDPM, DDPMWrapper, SuperResModel, UNetModel
from models.FireflyGAN import FireflyBase
from pytorch_lightning import seed_everything
import pretty_midi
from utils import *

def __parse_str(s):
    split = s.split(",")
    return [int(s) for s in split if s != "" and s is not None]


@hydra.main(config_path="../configs", config_name="train")
def Generation(config):

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
        dropout=config.model.dropout
    )

    ema_decoder = copy.deepcopy(decoder)
    decoder.eval()
    ema_decoder.eval()

    online_ddpm = DDPM(
        decoder,
        beta_1=config.model.beta1,
        beta_2=config.model.beta2,
        T=config.model.n_timesteps,
        var_type=config.evaluation.variance
    )
    target_ddpm = DDPM(
        ema_decoder,
        beta_1=config.model.beta1,
        beta_2=config.model.beta2,
        T=config.model.n_timesteps,
        var_type=config.evaluation.variance
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
    dataset_type = config.evaluation.dataset_type

    if dataset_type == "mastero":
        config_dataset = config.mastero
    if dataset_type == "musicnet":
        config_dataset = config.musicnet
    
    sample_rate = config_dataset.sample_rate
    hop_length = config.data.hop_length
    #Load test file 
    test_midi = glob.glob(os.path.join(config.evaluation.test_midi_path, "*mid"))

    with torch.no_grad():
        # Load Dpm-Solver
        noise_schedule = NoiseScheduleVP(schedule='linear')
        Unetmodel=ddpm_wrapper.load_model().eval()
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
        
        for fmidi in tqdm(test_midi):

            file_name = fmidi.strip().split("/")[-1]
            midi = pretty_midi.PrettyMIDI(fmidi)    
            pianoroll = midi.get_piano_roll(fs=sample_rate//hop_length)

            #pianoroll /= 128
            pianoroll = pianoroll/pianoroll.max()
            pianoroll = torch.from_numpy(pianoroll)

            #avoid OOM
            data_frame = len(pianoroll[0])
            max_frame = config.evaluation.max_frame
            if data_frame > max_frame:
                pianoroll = padding_spec(pianoroll, max_frame)
                one = max_frame
            else:
                pianoroll = padding_spec(pianoroll, 256)
                one = len(pianoroll[0])

            #split, avoid OOM
            pianoroll_split = torch.split(pianoroll, one, dim=1)
            result_mel = []
            for pianoroll in pianoroll_split:

                cond = pianoroll.unsqueeze(0).unsqueeze(0).float()
                x_T = torch.randn_like(cond)

                output = dpm_solver.sample( 
                                x_T.to(device),
                                y = None,
                                cond = cond.float().to(device),
                                steps = n_steps,
                                eps = 1e-4,
                                adaptive_step_size = False,
                                fast_version = True,
                                ).squeeze(1).cpu()
                
                output = denormalize(output, config.data.min_level_db)
                result_mel.append(output)

            result_mel = torch.cat(result_mel, dim=-1)

            '''
            The final audio spec may have obvious artifacts when cat the audio after generation. 
            Therefore, it is necessary for the vocoder to perform inference on the entire length in one go, 
            which may take a considerable amount of time (depending on the length of the audio).
            '''
            result_wav = FFGan(result_mel)[0][0].numpy()
            sf.write(f"{file_name}.wav", result_wav, sample_rate)
            
if __name__ == "__main__":
    Generation()
