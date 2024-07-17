import os
import yaml
import argparse
from tqdm import trange

import torch
import torchaudio
import numpy as np

from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import Resize


from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MultilabelF1Score,
    # https://torchmetrics.readthedocs.io/en/stable/classification/f1_score.html#multilabelf1score
    MultilabelROC,
    # https://torchmetrics.readthedocs.io/en/stable/classification/roc.html#multilabelroc
    MultilabelAveragePrecision,
    # https://torchmetrics.readthedocs.io/en/stable/classification/average_precision.html#multilabelaverageprecision
    MultilabelPrecisionRecallCurve,
    # https://torchmetrics.readthedocs.io/en/stable/classification/precision_recall_curve.html#multilabelprecisionrecallcurve
)

from anuraset import AnuraSet
from models import ResNetClassifier
import torch.nn.functional as F

SCRATCH = os.environ["SCRATCH"]

to_np = lambda x: x.data.cpu().numpy()

# How to modify when there are less classes?
class_mapping = [
    "SPHSUR",
    "BOABIS",
    "SCIPER",
    "DENNAH",
    "LEPLAT",
    "RHIICT",
    "BOALEP",
    "BOAFAB",
    "PHYCUV",
    "DENMIN",
    "ELABIC",
    "BOAPRA",
    "DENCRU",
    "BOALUN",
    "BOAALB",
    "PHYMAR",
    "PITAZU",
    "PHYSAU",
    "LEPFUS",
    "DENNAN",
    "PHYALB",
    "LEPLAB",
    "SCIFUS",
    "BOARAN",
    "SCIFUV",
    "AMEPIC",
    "LEPPOD",
    "ADEDIP",
    "ELAMAT",
    "PHYNAT",
    "LEPELE",
    "RHISCI",
    "SCINAS",
    "LEPNOT",
    "ADEMAR",
    "BOAALM",
    "PHYDIS",
    "RHIORN",
    "LEPFLA",
    "SCIRIZ",
    "DENELE",
    "SCIALT",
]


def save_inferences(test_data, samples, preds, dir):
    df_inferences = test_data.annotations.copy()
    df_inferences.iloc[samples, 8:] = preds
    df_inferences.to_csv(f"{dir}/inferences.csv", index=False)
    print(f"Inferences saved in: {dir}/inferences.csv")


def save_metrics(dir, metrics):
    list_doc = dict()

    list_doc["MultilabelAveragePrecision"] = (
        metrics["MultilabelAveragePrecision"].to("cpu").numpy().tolist()
    )
    list_doc["MultilabelF1Score"] = (
        metrics["MultilabelF1Score"].to("cpu").numpy().tolist()
    )
    list_doc["class_mapping"] = class_mapping

    with open(f"{dir}/metrics.yaml", "w") as f:
        yaml.dump(list_doc, f, default_flow_style=False)


def calculate_metrics(preds, targets, fn_metrics):
    # print(fn_metrics(preds, targets.long()))
    return fn_metrics(preds, targets.long())


def get_logits(model, data_loader, device):

    logits_all = torch.tensor([])
    model.eval()
    with torch.no_grad():
        for batch_idx, (input, target, index) in enumerate(data_loader):
            # put data and labels on device
            input, target = input.to(device), target.to(device)

            logits = model(input).cpu()
            logits_all = torch.cat((logits_all, logits), dim=0)

            if batch_idx == 3:
                break

    return logits_all


def evaluate(model, data_loader, loss_fn, metric_fn, device):

    sample_idx_all = []
    preds_all = []
    sigmoid = nn.Sigmoid()

    num_batches = len(data_loader)
    # set model to evaluation mode
    model.eval()

    # running averages # correct
    loss_total, metric_total = (
        0.0,
        0.0,
    )  # for now, we just log the loss and overall accuracy (OA)
    size = len(data_loader.dataset)
    progressBar = trange(len(data_loader), leave=False)
    with torch.no_grad():
        for batch_idx, (input, target, index) in enumerate(data_loader):
            # put data and labels on device
            input, target = input.to(device), target.to(device)

            prediction = sigmoid(model(input))

            loss = loss_fn(prediction, target)

            # log statistics
            loss_total += loss.item()
            # log metrics
            metric = metric_fn(prediction, target)
            metric_total += metric.item()
            # collecting infereces
            sample_idx_all.extend(index.tolist())
            preds_all.extend(prediction.tolist())

            progressBar.set_description(
                "[Evaluation] Loss: {:.4f}; F1-score macro: {:.4f} [{:>5d}/{:>5d}]".format(
                    loss_total / (batch_idx + 1),
                    metric_total / (batch_idx + 1),
                    (batch_idx + 1) * len(input),
                    size,
                )
            )
            progressBar.update(1)
        progressBar.close()
    loss_total /= num_batches
    metric_total /= num_batches

    return sample_idx_all, preds_all


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Domain shift.")
    parser.add_argument(
        "--config", help="Path to config file", default="configs/exp_resnet152.yaml"
    )
    parser.add_argument("--ood", help="", default="logits")
    parser.add_argument("--method", help="", default="max")
    args = parser.parse_args()

    # load config
    print(f'Using config "{args.config}"')
    config = yaml.safe_load(open(args.config, "r"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    print(f"Using device {device}")

    # Define Transformation

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        n_fft=512, hop_length=128, n_mels=128
    )
    test_transform = nn.Sequential(  # Transforms. Here's where we could add data augmentation (see Björn's lecture on August 11).
        mel_spectrogram,  # convert to a spectrogram
        torchaudio.transforms.AmplitudeToDB(),
        Resize([224, 448]),
    )

    id_species = [
        "SPHSUR",
        "BOABIS",
        "SCIPER",
        "DENNAH",
        "LEPLAT",
        "RHIICT",
        "BOALEP",
        "BOAFAB",
        "PHYCUV",
        "DENMIN",
        "ELABIC",
        "BOAPRA",
    ]

    val_transform = nn.Sequential(  # Transforms. Here's where we could add data augmentation (see Björn's lecture on August 11).
        # resamp,                                             # resample to 16 kHz
        mel_spectrogram,  # convert to a spectrogram
        torchaudio.transforms.AmplitudeToDB(),
        # torchvision.transforms.Lambda(min_max_normalize),   # normalize so min is 0 and max is 1
        Resize(config["image_size"]),
    )

    val_data = AnuraSet(
        annotations_file=config["val_metadata"],
        audio_dir=config["data_root"],
        transformation=val_transform,
        id_species=config["id_species"],
    )

    print(f"There are {len(val_data)} samples in the val set.")

    test_dataloader = DataLoader(
        val_data, batch_size=config["batch_size"], shuffle=False
    )

    multi_label = config["multilabel"]
    if multi_label:
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        loss_fn = nn.CrossEntropyLoss()

    metric_fn = MultilabelF1Score(num_labels=len(config["id_species"])).to(device)

    # load back the model
    model_instance = ResNetClassifier(
        model_type=config["model_type"], num_classes=len(config["id_species"])
    ).to(device)

    jobid = 5060485
    folder_name = config["folder_name"]
    model_path = f"{SCRATCH}/{folder_name}/{jobid}/model_states/final.pth"
    state_dict = torch.load(model_path, map_location=torch.device(device))
    model_instance.load_state_dict(state_dict)

    logits = get_logits(model_instance, test_dataloader, device)

    print(logits)
    logits_np = to_np(logits)
    outputs = torch.sigmoid(logits)

    ood = "logits"
    method = "max"

    if ood == "logit":
        if method == "max":
            scores = np.max(logits_np, axis=1)
        if method == "sum":
            scores = np.sum(logits_np, axis=1)
    elif ood == "energy":
        E_f = torch.log(1 + torch.exp(logits))
        if method == "max":
            scores = to_np(torch.max(E_f, dim=1)[0])
        if method == "sum":
            scores = to_np(torch.sum(E_f, dim=1))
        if method == "topk":
            scores = to_np(torch.sum(torch.topk(E_f, k=3, dim=1)[0], dim=1))
    elif ood == "prob":
        if method == "max":
            scores = np.max(to_np(outputs), axis=1)
        if method == "sum":
            scores = np.sum(to_np(outputs), axis=1)
    elif ood == "msp":
        outputs = F.softmax(logits, dim=1)
        scores = np.max(to_np(outputs), axis=1)
    else:
        scores = logits_np

    print(scores)
