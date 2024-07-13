import pandas as pd
import os

annotations = pd.read_csv("/network/scratch/y/yuyan.chen/anuraset/metadata.csv")
data_root = "/network/scratch/y/yuyan.chen/anuraset/"
audio_dir = os.path.join(data_root, "audio")
for index in range(5):
    site = annotations.iloc[index].loc["site"]
    fname = annotations.iloc[index].loc["fname"]
    min_t = annotations.iloc[index].loc["min_t"]
    max_t = annotations.iloc[index].loc["max_t"]
    path = os.path.join(audio_dir, f"{site}/{fname}_{min_t}_{max_t}.wav")
    print(path)
