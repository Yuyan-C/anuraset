import os
import argparse
import yaml
import glob

import torch
import torchaudio

from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize, Resize
from torchaudio.transforms import AmplitudeToDB
from torchmetrics.classification import MultilabelF1Score
from tqdm import trange
from anuraset import AnuraSet
from models import ResNetClassifier
from time import time

from util import init_seed, min_max_normalize

import wandb

SCRATCH = os.environ["SCRATCH"]
SLURM_JOBID = os.environ["SLURM_JOB_ID"]


def load_model(cfg):
    """
    Creates a model instance and loads the latest model state weights.
    """
    folder_name = cfg["folder_name"]
    checkpoint_dir = f"{SCRATCH}/{folder_name}/{SLURM_JOBID}/model_states/"
    model_instance = ResNetClassifier(
        model_type=cfg["model_type"], num_classes=cfg["num_classes"]
    )  # create an object instance of our CustomResNet18 class

    # load latest model state
    model_states = glob.glob(f"{checkpoint_dir}/*.pt")
    if len(model_states):
        # at least one save state found; get latest
        model_epochs = [
            int(m.replace(checkpoint_dir, "").replace(".pt", "")) for m in model_states
        ]
        start_epoch = max(model_epochs)

        # load state dict and apply weights to model
        print(f"Resuming from epoch {start_epoch}")
        state = torch.load(
            open(f"{checkpoint_dir}/{start_epoch}.pt", "rb"),
            map_location="cpu",
        )
        model_instance.load_state_dict(state["model"])

    else:
        # no save state found; start anew
        print("Starting new model")
        start_epoch = 0

    return model_instance, start_epoch


def save_model(cfg, epoch, model, stats):
    # make sure save directory exists; create if not
    folder_name = cfg["folder_name"]
    checkpoint_dir = f"{SCRATCH}/{folder_name}/{SLURM_JOBID}/model_states/"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # get model parameters and add to stats...
    stats["model"] = model.state_dict()

    # ...and save
    torch.save(stats, open(f"{checkpoint_dir}/{epoch}.pt", "wb"))

    # also save config file if not present
    cfpath = f"{checkpoint_dir}/config.yaml"
    if not os.path.exists(cfpath):
        with open(cfpath, "w") as f:
            yaml.dump(cfg, f)


def train(model, data_loader, loss_fn, optimiser, metric_fn, device):

    num_batches = len(data_loader)
    model.train()
    ## running averages
    loss_total, metric_total = 0.0, 0.0
    size = len(data_loader.dataset)
    progressBar = trange(len(data_loader), leave=False)
    for batch_idx, (input, target, index) in enumerate(data_loader):
        input, target = input.to(device), target.to(device)
        # calculate loss
        prediction = model(input)
        loss = loss_fn(prediction, target)
        # backpropagate error and update weights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        # log statistics
        # the .item() command retrieves the value of a single-valued tensor, regardless of its data type and device of tensor
        loss_total += loss.item()
        # log metrics
        metric = metric_fn(prediction, target)
        metric_total += metric.item()

        progressBar.set_description(
            "[Train] Loss: {:.4f}; F1-score macro: {:.4f} [{:>5d}/{:>5d}]".format(
                loss_total / (batch_idx + 1),
                metric_total / (batch_idx + 1),
                (batch_idx + 1) * len(input),
                size,
            )
        )
        progressBar.update(1)
    progressBar.close()  # end of epoch; finalize
    # you should avoid last batch due different size
    loss_total /= (
        num_batches  # shorthand notation for: loss_total = loss_total / len(dataLoader)
    )
    metric_total /= num_batches

    return loss_total, metric_total


def validate(model, data_loader, loss_fn, metric_fn, device):
    """
    Validation function. Looks like training
    function, except that we don't use any optimizer or gradient steps.
    """

    num_batches = len(data_loader)

    # put the model into evaluation mode
    model.eval()

    # running averages # correct
    loss_total, metric_total = (
        0.0,
        0.0,
    )  # for now, we just log the loss and overall accuracy (OA)

    # iterate over dataLoader
    progressBar = trange(len(data_loader), leave=False)

    with torch.no_grad():  # don't calculate intermediate gradient steps: we don't need them, so this saves memory and is faster

        for batch_idx, (input, target, index) in enumerate(data_loader):
            # put data and labels on device
            input, target = input.to(device), target.to(device)

            prediction = model(input)
            loss = loss_fn(prediction, target)

            # log statistics
            loss_total += loss.item()
            # log metrics
            metric = metric_fn(prediction, target)
            metric_total += metric.item()

            progressBar.set_description(
                "[Validation] Loss: {:.4f}; F1-score macro: {:.4f}".format(
                    loss_total / (batch_idx + 1),
                    metric_total / (batch_idx + 1),
                )
            )
            progressBar.update(1)
        progressBar.close()  # end of epoch; finalize
    loss_total /= num_batches
    metric_total /= num_batches

    return loss_total, metric_total


def main():

    parser = argparse.ArgumentParser(description="Domain shift.")
    parser.add_argument(
        "--config", help="Path to config file", default="configs/exp_resnet152.yaml"
    )
    args = parser.parse_args()

    # load config
    print(f'Using config "{args.config}"')

    config = yaml.safe_load(open(args.config, "r"))

    # init random number generator seed (set at the start)
    init_seed(config.get("seed", None))

    device = torch.device(config["device"])
    print(f"Using device {device}")

    # Define Transformation

    resamp = torchaudio.transforms.Resample(orig_freq=22050, new_freq=16000)
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        n_fft=512, hop_length=128, n_mels=128
    )
    time_mask = torchaudio.transforms.TimeMasking(
        time_mask_param=60,  # mask up to 60 consecutive time windows
    )
    freq_mask = torchaudio.transforms.FrequencyMasking(
        freq_mask_param=8,  # mask up to 8 consecutive frequency bins
    )
    train_transform = nn.Sequential(  # Transforms. Here's where we could add data augmentation (see Björn's lecture on August 11).
        # resamp,                             # resample to 16 kHz
        mel_spectrogram,  # convert to a spectrogram
        AmplitudeToDB(),  # Turn a spectrogram from the power/amplitude scale to the decibel scale.
        # Normalize(),                      # normalize so min is 0 and max is 1
        time_mask,  # randomly mask out a chunk of time
        freq_mask,  # randomly mask out a chunk of frequencies
        Resize(config["image_size"]),
    )
    val_transform = nn.Sequential(  # Transforms. Here's where we could add data augmentation (see Björn's lecture on August 11).
        # resamp,                                             # resample to 16 kHz
        mel_spectrogram,  # convert to a spectrogram
        torchaudio.transforms.AmplitudeToDB(),
        # torchvision.transforms.Lambda(min_max_normalize),   # normalize so min is 0 and max is 1
        Resize(config["image_size"]),
    )

    ANNOTATIONS_FILE = os.path.join(config["data_root"], config["metadata"])

    AUDIO_DIR = os.path.join(config["data_root"], "audio")

    training_data = AnuraSet(
        annotations_file=ANNOTATIONS_FILE,
        audio_dir=AUDIO_DIR,
        transformation=train_transform,
        train=True,
    )
    print(f"There are {len(training_data)} samples in the training set.")

    # TODO: call val not test!
    val_data = AnuraSet(
        annotations_file=ANNOTATIONS_FILE,
        audio_dir=AUDIO_DIR,
        transformation=val_transform,
        train=False,
    )
    print(f"There are {len(val_data)} samples in the test set.")

    train_dataloader = DataLoader(
        training_data,
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=4,
    )

    val_dataloader = DataLoader(
        val_data,
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=4,
    )

    multi_label = config["multilabel"]
    if multi_label:
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        loss_fn = nn.CrossEntropyLoss()

    # initialize model
    model_instance, current_epoch = load_model(config)

    model_instance.to(device)

    optimiser = torch.optim.Adam(
        model_instance.parameters(), lr=config["learning_rate"]
    )

    metric_fn = MultilabelF1Score(num_labels=config["num_classes"]).to(device)

    start = time()
    progress_bar_epoch = trange(config["num_epochs"])
    if config["wandb_project"] is not None:
        wandb.init(
            # Set the project where this run will be logged
            project=config["wandb_project"],
            entity=config["wandb_entity"],
            name=config["wandb_name"] + f"_{SLURM_JOBID}",
            resume="allow",
            config=config,
        )

    print("Starting training")
    for current_epoch in range(current_epoch, config["num_epochs"]):
        start_epoch = time()
        current_epoch += 1

        loss_train, metric_train = train(
            model_instance, train_dataloader, loss_fn, optimiser, metric_fn, device
        )
        loss_val, metric_val = validate(
            model_instance, val_dataloader, loss_fn, metric_fn, device
        )
        progress_bar_epoch.update(1)
        progress_bar_epoch.write(
            "Epoch: {:.0f}: Loss val: {:.4f} ; F1-score macro val: {:.4f} - Epoch time: {:.1f}s; Total time: {:.1f}s - {:.0f}%".format(
                (current_epoch),
                loss_val,
                metric_val,
                (time() - start_epoch),
                (time() - start),
                (100 * (current_epoch) / config["num_epochs"]),
            )
        )

        # combine stats and save
        stats = {
            "loss_train": loss_train,
            "loss_val": loss_val,
            "metric_train": metric_train,
            "metric_val": metric_val,
        }
        # TODO: wandb or YAML?
        save_model(config, current_epoch, model_instance, stats)
    progress_bar_epoch.close()
    print("Finished training")

    # save model
    folder_name = config["folder_name"]
    model_path = f"{SCRATCH}/{folder_name}/{SLURM_JOBID}/model_states/final.pth"
    torch.save(model_instance.state_dict(), model_path)
    wandb.log_artifact(model_path, name="model", type="model")

    print(f"Trained feed forward net saved at {model_path}")


if __name__ == "__main__":
    # This block only gets executed if you call the "train.py" script directly
    main()
