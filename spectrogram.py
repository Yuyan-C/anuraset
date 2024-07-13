import torchaudio
import matplotlib.pyplot as plt


def plot_spectrogram(specgram, title=None, ylabel="freq_bin", ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.imshow(
        librosa.power_to_db(specgram),
        origin="lower",
        aspect="auto",
        interpolation="nearest",
    )

    fig.save()


mel_spectrogram = torchaudio.transforms.MelSpectrogram(
    n_fft=512, hop_length=128, n_mels=128
)

signal, rate = torchaudio.load(
    "/network/scratch/y/yuyan.chen/anuraset/audio/INCT17/INCT17_20191113_040000_0_3.wav"
)

print(mel_spectrogram(signal))


fig, axs = plt.subplots(2, 1)
