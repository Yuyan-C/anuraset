import os
import argparse
import yaml
import glob
import datetime

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

from timm.utils import AverageMeter
from util import init_seed, min_max_normalize

import wandb

SCRATCH = os.environ["SCRATCH"]
SLURM_JOBID = os.environ["SLURM_JOB_ID"]


def print_metrics(metrics):
    return " - ".join([f"{k}: {v}" for k, v in metrics.items()])


def load_model(cfg):
    """
    Creates a model instance and loads the latest model state weights.
    """
    folder_name = cfg["folder_name"]
    checkpoint_dir = f"{SCRATCH}/{folder_name}/{SLURM_JOBID}/model_states/"
    model_instance = ResNetClassifier(
        model_type=cfg["model_type"], num_classes=len(cfg['id_species'])
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


def train(model, data_loader, loss_fn, optimiser, metric_fn, device, config, epoch):
    model.train()
    ## running averages
    # loss_total, metric_total = 0.0, 0.0
    train_loss = AverageMeter()
    train_metric = AverageMeter()
    batch_time = AverageMeter()
    batch_end = time()
    steps_per_epoch = len(data_loader)
    for i, (input, target, index) in enumerate(data_loader):
        input, target = input.to(device), target.to(device)
        # calculate loss
        prediction = model(input)
        loss = loss_fn(prediction, target)
        # backpropagate error and update weights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        train_loss.update(loss.detach().cpu(), input.size()[0])
        metric = metric_fn(prediction, target)
        train_metric.update(metric, input.size()[0])

        batch_time.update(time() - batch_end)
        batch_end = time()

        if i % config["log_frequency"] == 0:
            eta = batch_time.avg * (steps_per_epoch - i)
            print(
                f"[{epoch:02d}/{config['epochs']:02d}][{i:05d}/{steps_per_epoch:05d}]",
                f" ETA: {datetime.timedelta(seconds=int(eta))} -",
                f" loss: {train_loss.avg:.4f} -",
                f" f1 macro: {100*train_metric.avg:.2f}%",
                flush=True,
            )

    return train_loss, train_metric


def validate(model, data_loader, loss_fn, metric_fn, device):
    """
    Validation function. Looks like training
    function, except that we don't use any optimizer or gradient steps.
    """

    num_batches = len(data_loader)

    # put the model into evaluation mode
    model.eval()

    val_loss = AverageMeter()
    val_metric = AverageMeter()

    with torch.no_grad():  # don't calculate intermediate gradient steps: we don't need them, so this saves memory and is faster

        for batch_idx, (input, target, index) in enumerate(data_loader):
            # put data and labels on device
            input, target = input.to(device), target.to(device)

            prediction = model(input)
            loss = loss_fn(prediction, target)

            # log statistics
            val_loss.update(loss, target.size()[0])
            # log metrics
            metric = metric_fn(prediction, target)
            val_metric.update(metric, target.size()[0])

    return val_loss, val_metric


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

    device = "cuda" if torch.cuda.is_available() else "cpu"
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

    training_data = AnuraSet(
        annotations_file=config["train_metadata"],
        audio_dir=config["data_root"],
        transformation=train_transform,
        id_species=config["id_species"],
    )

    print(f"There are {len(training_data)} samples in the training set.")

    # TODO: call val not test!
    val_data = AnuraSet(
        annotations_file=config["val_metadata"],
        audio_dir=config["data_root"],
        transformation=val_transform,
        id_species=config["id_species"],
    )
    print(f"There are {len(val_data)} samples in the test set.")
    num_workers = get_num_workers()

    train_dataloader = DataLoader(
        training_data,
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=num_workers,
    )

    val_dataloader = DataLoader(
        val_data,
        batch_size=config["batch_size"],
        shuffle=False,
        drop_last=True,
        pin_memory=True,
        num_workers=num_workers,
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

    metric_fn = MultilabelF1Score(num_labels=len(config['id_species'])).to(device)

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
    for current_epoch in range(current_epoch, config["epochs"]):
        start_epoch = time()
        current_epoch += 1

        epoch_metrics = {}
        epoch_metrics_formated = {}

        loss_train, metric_train = train(
            model_instance,
            train_dataloader,
            loss_fn,
            optimiser,
            metric_fn,
            device,
            config,
            current_epoch,
        )
        loss_val, metric_val = validate(
            model_instance, val_dataloader, loss_fn, metric_fn, device
        )

        epoch_metrics["train/loss"] = loss_train.avg
        epoch_metrics_formated["train_loss"] = f"{loss_train.avg:.4f}"
        epoch_metrics["train/macro-f1"] = metric_train.avg
        epoch_metrics_formated["train_macro-f1"] = f"{metric_train.avg:.4f}"

        epoch_metrics["val/loss"] = loss_val.avg
        epoch_metrics_formated["val_loss"] = f"{loss_val.avg:.4f}"
        epoch_metrics["val/macro-f1"] = metric_val.avg
        epoch_metrics_formated["val_macro-f1"] = f"{metric_val.avg:.4f}"

        print(
            f"Epoch {current_epoch:02d}/{config['epochs']:02d} -"
            f" {print_metrics(epoch_metrics_formated)}",
            flush=True,
        )

        if config["wandb_project"] != None:
            wandb.log(
                {
                    **epoch_metrics,
                    **{"time_per_epoch": int((time() - start_epoch))},
                }
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
    print("Finished training")

    # save model
    folder_name = config["folder_name"]
    model_path = f"{SCRATCH}/{folder_name}/{SLURM_JOBID}/model_states/final.pth"
    torch.save(model_instance.state_dict(), model_path)
    if config["wandb_project"] is not None:
        wandb.log_artifact(model_path, name="model", type="model")

    print(f"Trained feed forward net saved at {model_path}")


def get_num_workers() -> int:
    """Gets the optimal number of DatLoader workers to use in the current job."""
    if "SLURM_CPUS_PER_TASK" in os.environ:
        return int(os.environ["SLURM_CPUS_PER_TASK"])
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return torch.multiprocessing.cpu_count()


if __name__ == "__main__":
    # This block only gets executed if you call the "train.py" script directly
    main()
