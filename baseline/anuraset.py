import os
import torch
import torchaudio
import pandas as pd

from torch.utils.data import Dataset


class AnuraSet(Dataset):
    """AnuraSet: A dataset for bioacoustic classification of tropical anurans

    Args:
        annotations_file (string): path of the metadata csv table with labels of
            AnuraSet and audio samples information
        audio_dir(string): path of the folder with audio samples of the AnuraSet
            associated with the metadata table
        transformation (callable?): L A function/transform that takes audios before
            feature extraction and returns a transformed version.This tranformantions
            include melspectogram and augmentations.
        device (string): if using cuda (GPU) or CPU
        train (bool): If True, creates dataset from using 'train' samples in the
                'subset' column of the metadata
    """

    def __init__(
        self,
        annotations_file,
        audio_dir,
        transformation,
        id_species=None,
        osr_detection=False,
    ):

        if isinstance(annotations_file, str):
            df = pd.read_csv(annotations_file)
        else:
            df = annotations_file.copy()

        self.annotations = df
        self.audio_dir = audio_dir
        self.transformation = transformation
        self.id_species = id_species
        self.osr_detection = osr_detection

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_id_sample_label(index)
        signal, _ = torchaudio.load(audio_sample_path)

        signal = self.transformation(signal)

        if self.osr_detection:
            osr_label = self._get_osr_label(index)
            all_label = self._get_all_label(index)
            return signal, label, index, osr_label, all_label
        else:
            return signal, label, index

    def _get_audio_sample_path(self, index):
        site = self.annotations.iloc[index].loc["site"]
        fname = self.annotations.iloc[index].loc["fname"]
        min_t = self.annotations.iloc[index].loc["min_t"]
        max_t = self.annotations.iloc[index].loc["max_t"]
        path = os.path.join(self.audio_dir, f"{site}/{fname}_{min_t}_{max_t}.wav")
        return path

    def _get_audio_id_sample_label(self, index):
        return torch.Tensor(self.annotations[self.id_species].iloc[index])

    def _get_all_label(self, index):
        return torch.Tensor(self.annotations.iloc[index, 8:])

    def _get_osr_label(self, index):
        return torch.Tensor(self.annotations[["has_ood", "has_id"]].iloc[index])
