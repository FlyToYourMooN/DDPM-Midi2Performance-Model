import collections as C
import pandas as pd
import yaml
import json

data = pd.read_csv("/disk2/MusicNet/musicnet_metadata.csv")
music_types = data['ensemble'].unique()

yaml_dict = {}
for music_type in music_types:
    music_type_id = data[data["ensemble"]==music_type]
    yaml_dict[music_type] = list(music_type_id["id"])
    with open("/disk2/Opensource-DDPM-M2P/main/configs/musicnet_instrument.json", 'w') as file:
        json.dump(yaml_dict, file, indent=4)

