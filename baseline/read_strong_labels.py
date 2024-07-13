import os
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

import torchaudio
import torchaudio.transforms as T
import torch

SCRATCH = os.environ["SCRATCH"]


def clip_samples():
    df = pd.read_csv(f"{SCRATCH}/anuraset/clipped/INCT17/filename.csv")
    filenames = list(df.filename)
    audio_list, clip_fn_list, species_list, min_list, max_list, duration_list = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for fn in filenames:
        wave_fn = f"{SCRATCH}/anuraset/raw_data/INCT17/{fn}.wav"
        waveform, sample_rate = torchaudio.load(wave_fn)
        label_fn = f"{SCRATCH}/anuraset/strong_labels/INCT17/{fn}.txt"

        with open(label_fn, "r", encoding="UTF-8") as f:
            for i, line in enumerate(f):
                splited = line[:-1].split("\t")
                min_t = float(splited[0])
                max_t = float(splited[1])
                species = splited[-1][:-2]

                start_sample = int(min_t * sample_rate)
                end_sample = int(max_t * sample_rate)
                clipped_waveform = waveform[:, start_sample:end_sample]

                clip_fn = f"/network/scratch/y/yuyan.chen/anuraset/clipped/INCT17/{fn}_{i}.wav"

                audio_list.append(wave_fn)
                clip_fn_list.append(clip_fn)
                species_list.append(species)
                duration_list.append(max_t - min_t)
                min_list.append(min_t)
                max_list.append(max_t)

                torchaudio.save(clip_fn, clipped_waveform, sample_rate)

            break

    df = pd.DataFrame(
        list(
            zip(
                audio_list,
                clip_fn_list,
                species_list,
                min_list,
                max_list,
                duration_list,
            )
        ),
        columns=["audio_path", "clip_path", "species", "min_t", "max_t", "length"],
    )

    df.to_csv(f"{SCRATCH}/anuraset/clipped/INCT17/metadata.csv", index=False)


def strong_labels():
    sites = ["INCT17"]
    # sites = ["INCT17", "INCT20955", "INCT4", "INCT41"]
    species_dict = dict()

    for s in sites:
        directory = f"/network/scratch/y/yuyan.chen/anuraset/strong_labels/{s}"

        fn = [name.split(".txt")[0] for name in os.listdir(directory)]
        df = pd.DataFrame({"filename": fn})
        df.to_csv(f"{SCRATCH}/anuraset/clipped/INCT17/filename.csv", index=False)

    #         # Open file
    #         with open(os.path.join(directory, name)) as f:
    #             for i, line in enumerate(f):
    #                 splited = line[:-1].split("\t")
    #                 try:
    #                     min_t = float(splited[0])
    #                     max_t = float(splited[1])
    #                     species = splited[-1][:-2]
    #                     if species not in species_dict.keys():
    #                         species_dict[species] = [max_t - min_t]
    #                     else:
    #                         species_dict[species] += [max_t - min_t]
    #                 except ValueError:
    #                     print(name)

    # species_count = dict()
    # for k in species_dict.keys():
    #     species_count[k] = len(species_dict[k])

    # species_count = {
    #     k: v for k, v in sorted(species_count.items(), key=lambda item: item[1])
    # }

    # for k, v in species_count.items():
    #     print(k, v)

    # top_10 = list(species_count.keys())[-10:]
    # top_10_dict = dict((k, species_dict[k]) for k in top_10)
    # print(top_10_dict)

    # plt.figure(figsize=(10, 6))
    # plt.boxplot(top_10_dict.values(), patch_artist=True)
    # plt.xlabel("species")
    # plt.ylabel("time")
    # plt.xticks(list(range(1, 11)), top_10_dict.keys())
    # plt.savefig("clip_length.png")


if __name__ == "__main__":
    strong_labels()
    clip_samples()
