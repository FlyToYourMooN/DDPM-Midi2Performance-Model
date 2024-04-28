# A Midi(pianoroll) to Performace model based on DDPM
![](imgs/generation_example.png "Pipline")
## TO DO
- [x] Piano Solo(Mastero Dataset) model train
- [x] The script for processing the Mastero and Musicnet dataset
- [x] Inpainting and Generation Piano Solo based on MIDI
- [ ]  M2P models trained on some multi-instrument data from the Musicnet dataset(String Quartet et.al)
- [ ]  Inpainting and Generation Multi-Instrument based on MIDI
- [ ]  Unconditional Generation model
## Requirements
1. install torch == 2.2.0 *(Versions below this will not be able to use Vocoder, and versions above this do not have adapted Lighting)* , torchaudio

2. run
```bash
pip install -r requirements.txt
```
## Inference for Midi to Performace
1. Download pretained M2P model in releases
2. Download pretained Vocoder in https://github.com/fishaudio/vocoder/releases/download/1.0.0/firefly-gan-base-generator.ckpt
3. Adjust `train.yaml`, 
**Notice use Absolute path**
```yaml
evaluation:
    chkpt_path: "/disk2/Opensource-DDPM-M2P/M2P_model/checkpoints/ddpmv2-2048-512-2048-2res-epoch=227-loss=0.0271.ckpt"         #M2P model path
    device: "cuda"
    vocoder_path: "/disk2/Opensource-DDPM-M2P/M2P_model/firefly-gan-base-generator.ckpt"  #Vocoder path
    test_midi_path: "/disk2/Opensource-DDPM-M2P/midis" #midi path
    data_start: 0
    max_frame: 2048  #Once inference length
    dataset_type: mastero  
```

2. run
```bash
python main/generation/pianoroll_generation.py 
```

## Train Piano Solo M2P model
0. download mastero dataset and unzip
```
mastero
├───...
├───2014
│   ├───...wav
│   ├───...midi
│   └───...
├───2015
├───...
```
1. Adjust `train.yaml`

```yaml
mastero:
  data_path: /disk2/Piano-Solo/mastero    
  save_path: /disk2/Piano-Solo/Processed_data  
```

2. run
```bash
python main/prepare_data/prepare_mastero_data.py 
```
3. run (optional setting, just for checking whether the dataset is aligned.)
```bash
python main/prepare_data/check_dataset.py 
```
4. train model (the model can produce comprehensible results after training for 80 epochs)
```bash
python main/train_ddpm.py 
```

## Something TO DO ...

## Reference
Vocoder: https://github.com/fishaudio/vocoder 

Musicnet dataset script: https://github.com/bwang514/PerformanceNet
## Cite
```bibtex
@INPROCEEDINGS{10095769,
  author={Liu, Kaiyang and Gan, Wendong and Yuan, Chenchen},
  booktitle={ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={MAID: A Conditional Diffusion Model for Long Music Audio Inpainting}, 
  year={2023},
  pages={1-5},
  doi={10.1109/ICASSP49357.2023.10095769}}
'''

