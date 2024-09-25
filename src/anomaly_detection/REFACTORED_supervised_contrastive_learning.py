import torch
import numpy as np
import random
import os
from PIL import Image
import cv2
import torchvision
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from typing import Tuple, List
from src.logs import logger


# ---- DATA LOADER ----
class RailwayDataset(Dataset):
    def __init__(
        self, dir_dataset: str, items: List[str], input_shape: Tuple[int, int, int]
    ):
        self.dir_dataset = dir_dataset
        self.items = items
        self.input_shape = input_shape
        self.files = os.listdir(self.dir_dataset + self.items[0])
        self.files = [file for file in self.files if file != "Thumbs.db"]
        self.X = self._load_images()
        self.labels = np.zeros(len(self.files))
        self.indexes = np.arange(self.X.shape[0])

    def _load_images(self) -> np.ndarray:
        X = np.zeros(
            (
                len(self.files),
                self.input_shape[0],
                self.input_shape[1],
                self.input_shape[2],
            )
        )
        for iFile, file in enumerate(self.files):
            for item in range(len(self.items)):
                x = Image.open(os.path.join(self.dir_dataset + self.items[item], file))
                x = np.asarray(x) / 255.0
                x = cv2.resize(x, (self.input_shape[2], self.input_shape[1]))
                X[iFile, item, :, :] = x
        return X

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> np.ndarray:
        return self.X[idx, :, :, :].copy()

    def label_cruces_adif(self):
        for iFile, file in enumerate(self.files):
            i_label = int(file[-5])
            self.labels[iFile] = 1 if i_label >= 1 else 0


# ---- LOSSES ----
def kl_loss(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    return torch.sum(p * torch.log(p / q + 1e-3))


def l2_distance(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean(torch.square(x1 - x2)))


class SupConLoss(torch.nn.Module):
    def __init__(
        self,
        temperature: float = 0.07,
        contrast_mode: str = "all",
        base_temperature: float = 0.07,
    ):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor = None,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        device = torch.device("cuda" if features.is_cuda else "cpu")
        if len(features.shape) < 3:
            raise ValueError(
                "`features` needs to be [bsz, n_views, ...], at least 3 dimensions are required"
            )

        features = features.view(features.shape[0], features.shape[1], -1)
        batch_size = features.shape[0]

        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError("Unknown mode: {}".format(self.contrast_mode))

        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature
        )
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0,
        )
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-3)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-3)

        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


# ---- MODELS ----
class ResNetEncoder(torch.nn.Module):
    def __init__(self, in_channels: int, n_blocks: int = 4, pretrained: bool = False):
        super(ResNetEncoder, self).__init__()
        self.n_blocks = n_blocks
        self.nfeats = 512 // (2 ** (5 - n_blocks))
        self.resnet18_model = torchvision.models.resnet18(pretrained=pretrained)
        self.input = torch.nn.Conv2d(
            in_channels,
            64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        x = self.input(x)
        F = []
        for iBlock in range(1, self.n_blocks + 1):
            x = list(self.resnet18_model.children())[iBlock + 2](x)
            F.append(x)
        return x, F


class PrototypicalNetwork(torch.nn.Module):
    def __init__(
        self,
        encoder: ResNetEncoder,
        projection_dim: int = 128,
        n_classes: int = 2,
        contrastive_loss: bool = True,
    ):
        super(PrototypicalNetwork, self).__init__()
        self.encoder = encoder
        self.projection = torch.nn.Sequential(
            torch.nn.Linear(encoder.nfeats, encoder.nfeats // 4),
            torch.nn.ReLU(),
            torch.nn.Linear(encoder.nfeats // 4, projection_dim),
        )
        self.classifier = torch.nn.Linear(projection_dim, n_classes)
        self.contrastive_loss = contrastive_loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z, _ = self.encoder(x)
        z = torch.squeeze(torch.nn.AdaptiveAvgPool2d(1)(z))
        z = self.projection(z)
        if self.contrastive_loss:
            z = torch.nn.functional.normalize(z, dim=1)
        logits = self.classifier(z)
        return logits


# ---- TRAINING FUNCTION ----


def train_prototypical_network(
    model: PrototypicalNetwork,
    dataloader: DataLoader,
    epochs: int,
    learning_rate: float,
    device: torch.device,
) -> Tuple[List[float], List[float], List[float], List[float]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    model = model.to(device)

    train_loss = []
    train_accuracy = []
    train_precision = []
    train_recall = []
    train_f1 = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        all_labels = []
        all_preds = []

        for X, Y in dataloader:
            X, Y = X.to(device), Y.to(device)
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, Y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += Y.size(0)
            correct += (predicted == Y).sum().item()
            all_labels.extend(Y.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

        epoch_loss = running_loss / len(dataloader)
        epoch_accuracy = correct / total
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="macro"
        )

        train_loss.append(epoch_loss)
        train_accuracy.append(epoch_accuracy)
        train_precision.append(precision)
        train_recall.append(recall)
        train_f1.append(f1)

        logger.info(
            f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}"
        )

    return train_loss, train_accuracy, train_precision, train_recall, train_f1


# ---- EVALUATION FUNCTION ----
def evaluate_model(
    model: PrototypicalNetwork, dataloader: DataLoader, device: torch.device
) -> Tuple[float, float, float, float, np.ndarray]:
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for X, Y in dataloader:
            X, Y = X.to(device), Y.to(device)
            logits = model(X)
            _, predicted = torch.max(logits.data, 1)
            total += Y.size(0)
            correct += (predicted == Y).sum().item()
            all_labels.extend(Y.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    accuracy = correct / total
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro"
    )
    cm = confusion_matrix(all_labels, all_preds)

    return accuracy, precision, recall, f1, cm
