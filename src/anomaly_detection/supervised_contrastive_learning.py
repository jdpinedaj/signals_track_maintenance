import torch
import numpy as np
import random
import os
from PIL import Image
import cv2
import torchvision
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.decomposition import PCA

# ---- DATA LOADING ----


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dir_dataset, items, input_shape):
        self.dir_dataset = dir_dataset
        self.items = items
        self.input_shape = input_shape
        self.files = [
            f for f in os.listdir(self.dir_dataset + self.items[0]) if f != "Thumbs.db"
        ]
        self.X = np.zeros(
            (len(self.files), input_shape[0], input_shape[1], input_shape[2])
        )
        self.labels = np.zeros(len(self.files))

        print("[INFO]: Loading images into memory")
        for iFile in range(len(self.files)):
            print(f"{iFile + 1}/{len(self.files)}", end="\r")
            for item in range(len(self.items)):
                x = Image.open(
                    os.path.join(self.dir_dataset + self.items[item], self.files[iFile])
                )
                x = np.asarray(x) / 255.0
                x = cv2.resize(x, (self.input_shape[2], self.input_shape[1]))
                self.X[iFile, item, :, :] = x

        self.indexes = np.arange(0, self.X.shape[0])

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx, :, :, :].copy()

    def label_cruces_adif(self):
        for iFile in range(len(self.files)):
            i_label = int(self.files[iFile][-5])
            self.labels[iFile] = 1 if i_label >= 1 else 0


class Generator:
    def __init__(self, train_dataset, bs, shuffle=True):
        self.dataset = train_dataset
        self.bs = bs
        self.shuffle = shuffle
        self.indexes = train_dataset.indexes.copy()
        self._idx = 0
        if self.shuffle:
            random.shuffle(self.indexes)

    def __len__(self):
        return round(len(self.indexes) / self.bs)

    def __iter__(self):
        return self

    def __next__(self):
        if self._idx + self.bs >= len(self.indexes):
            self._reset()
            raise StopIteration()

        batch = [
            self.dataset.__getitem__(i) for i in range(self._idx, self._idx + self.bs)
        ]
        self._idx += self.bs
        return np.array(batch)

    def _reset(self):
        if self.shuffle:
            random.shuffle(self.indexes)
        self._idx = 0


class GeneratorFSL:
    def __init__(self, train_dataset, n=4, m=4, shuffle=True, classes=[0, 1]):
        self.dataset = train_dataset
        self.n = n  # queries
        self.m = m  # support
        self.shuffle = shuffle
        self.indexes = train_dataset.indexes.copy()
        self._idx = 0
        self.classes = classes
        self.Y = self.dataset.labels
        if self.shuffle:
            random.shuffle(self.indexes)

    def __len__(self):
        return round(len(self.indexes) / self.n)

    def __iter__(self):
        return self

    def __next__(self):
        if self._idx + self.n >= len(self.indexes):
            self._reset()
            raise StopIteration()

        Xq, Y = self._get_queries()
        Xs = self._get_support()

        self._idx += self.n
        return np.array(Xs), np.array(Xq), Y

    def _get_queries(self):
        Xq = []
        Y = []
        for i in range(self._idx, self._idx + self.n):
            x = self.dataset.__getitem__(self.indexes[i])
            Xq.append(x)
            y_i = np.zeros((int(len(self.classes))))
            y_i[int(self.Y[self.indexes[i]])] = 1.0
            Y.append(y_i)
        return Xq, Y

    def _get_support(self):
        Xs = []
        for iClass in self.classes:
            Xs_i = []
            queries = np.random.choice(
                np.array(self.indexes)[
                    np.argwhere(self.Y[self.indexes] == iClass).flatten()
                ],
                self.m,
            )
            for i in queries:
                x = self.dataset.__getitem__(i)
                Xs_i.append(x)
            Xs.append(Xs_i)
        return Xs

    def _reset(self):
        if self.shuffle:
            random.shuffle(self.indexes)
        self._idx = 0


# ---- LOSS FUNCTIONS ----


def kl_loss(p, q):
    return torch.sum(p * torch.log(p / q + 1e-3))


def l2_distance(x1, x2):
    return torch.sqrt(torch.mean(torch.square(x1 - x2)))


class SupConLoss(torch.nn.Module):
    def __init__(self, temperature=0.07, contrast_mode="all", base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        device = torch.device("cuda:0" if features.is_cuda else "cpu")
        if len(features.shape) < 3:
            raise ValueError(
                "`features` needs to be [bsz, n_views, ...], at least 3 dimensions are required"
            )
        if len(features.shape) > 3:
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


class Resnet(torch.nn.Module):
    def __init__(self, in_channels, n_blocks=4, pretrained=False):
        super(Resnet, self).__init__()
        self.n_blocks = n_blocks
        self.nfeats = 512 // (2 ** (5 - n_blocks))
        self.resnet18_model = torchvision.models.resnet18(pretrained=pretrained)
        self.input = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False,
        )

    def forward(self, x):
        x = self.input(x)
        F = []
        for iBlock in range(1, self.n_blocks + 1):
            x = list(self.resnet18_model.children())[iBlock + 2](x)
            F.append(x)
        return x, F


class Encoder(torch.nn.Module):
    def __init__(
        self,
        mode,
        fin=1,
        zdim=128,
        dense=False,
        n_blocks=4,
        spatial_dim=(8, 16),
        pretrained=False,
    ):
        super(Encoder, self).__init__()
        self.mode = mode  # Supported modes: ae, vae, cavga, anoVAEGAN
        self.fin = fin
        self.zdim = zdim
        self.dense = dense
        self.n_blocks = n_blocks

        # 1) Feature extraction
        self.backbone = Resnet(
            in_channels=self.fin, n_blocks=self.n_blocks, pretrained=pretrained
        )

        # 2) Latent space (dense or spatial)
        if not self.dense:  # spatial
            if self.mode in ["ae", "f_ano_gan"]:
                self.z = torch.nn.Conv2d(self.backbone.nfeats, zdim, (1, 1))
            else:
                self.mu = torch.nn.Conv2d(self.backbone.nfeats, zdim, (1, 1))
                self.log_var = torch.nn.Conv2d(self.backbone.nfeats, zdim, (1, 1))
        else:  # dense
            if self.mode in ["ae", "f_ano_gan"]:
                self.z = torch.nn.Linear(
                    self.backbone.nfeats * spatial_dim[0] * spatial_dim[1], zdim
                )
            else:
                self.mu = torch.nn.Linear(
                    self.backbone.nfeats * spatial_dim[0] * spatial_dim[1], zdim
                )
                self.log_var = torch.nn.Linear(
                    self.backbone.nfeats * spatial_dim[0] * spatial_dim[1], zdim
                )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)  # standard deviation
        eps = torch.randn_like(std)
        return mu + eps * std  # sampling

    def forward(self, x):
        # 1) Feature extraction
        x, allF = self.backbone(x)
        if self.dense:
            x = torch.nn.Flatten()(x)

        # 2) Latent space
        if self.mode in ["ae", "f_ano_gan"]:
            z = self.z(x)
            z_mu, z_logvar = None, None
        else:
            z_mu = self.mu(x)
            z_logvar = self.log_var(x)
            z = self.reparameterize(z_mu, z_logvar)

        return z, z_mu, z_logvar, allF


class Decoder(torch.nn.Module):
    def __init__(
        self,
        fin=256,
        nf0=128,
        n_channels=1,
        dense=False,
        n_blocks=4,
        spatial_dim=(8, 16),
    ):
        super(Decoder, self).__init__()
        self.n_blocks = n_blocks
        self.dense = dense
        self.spatial_dim = spatial_dim
        self.fin = fin

        if self.dense:
            self.dense = torch.nn.Linear(fin, fin * spatial_dim[0] * spatial_dim[1])

        n_filters_in = [fin] + [nf0 // 2**i for i in range(self.n_blocks)]
        n_filters_out = [nf0 // 2 ** (i - 1) for i in range(1, self.n_blocks + 1)] + [
            n_channels
        ]

        self.blocks = torch.nn.ModuleList()
        for i in range(self.n_blocks):
            self.blocks.append(ResBlock(n_filters_in[i], n_filters_out[i]))
        self.out = torch.nn.Conv2d(
            n_filters_in[-1], n_filters_out[-1], kernel_size=(3, 3), padding=(1, 1)
        )

    def forward(self, x):
        if self.dense:
            x = self.dense(x)
            x = torch.nn.Unflatten(
                -1, (self.fin, self.spatial_dim[0], self.spatial_dim[1])
            )(x)

        for i in range(self.n_blocks):
            x = self.blocks[i](x)
        f = x
        out = self.out(f)

        return out, f


class ResBlock(torch.nn.Module):
    def __init__(self, fin, fout):
        super(ResBlock, self).__init__()
        self.conv_straight_1 = torch.nn.Conv2d(
            fin, fout, kernel_size=(3, 3), padding=(1, 1)
        )
        self.bn_1 = torch.nn.BatchNorm2d(fout)
        self.conv_straight_2 = torch.nn.Conv2d(
            fout, fout, kernel_size=(3, 3), padding=(1, 1)
        )
        self.bn_2 = torch.nn.BatchNorm2d(fout)
        self.conv_skip = torch.nn.Conv2d(fin, fout, kernel_size=(3, 3), padding=(1, 1))
        self.upsampling = torch.nn.Upsample(scale_factor=(2, 2))
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x_st = self.upsampling(x)
        x_st = self.conv_straight_1(x_st)
        x_st = self.relu(x_st)
        x_st = self.bn_1(x_st)
        x_st = self.conv_straight_2(x_st)
        x_st = self.relu(x_st)
        x_st = self.bn_2(x_st)

        x_sk = self.upsampling(x)
        x_sk = self.conv_skip(x_sk)

        out = x_sk + x_st
        return out


class ClusteringLayer(torch.nn.Module):
    def __init__(self, nClusters, zdim, centroids_init, alpha=1, distance="l2"):
        super(ClusteringLayer, self).__init__()
        self.nClusters = nClusters
        self.zdim = zdim
        self.alpha = alpha
        self.clusters = torch.nn.Parameter(
            torch.tensor(centroids_init, requires_grad=True)
        )
        self.distance = distance

    def forward(self, x):
        x1 = x.unsqueeze(1).repeat(1, self.nClusters, 1)
        x2 = self.clusters.unsqueeze(0).repeat(x.shape[0], 1, 1)
        if self.distance == "l2":
            d = torch.sum(torch.square(x1 - x2), dim=2)
        elif self.distance == "cosine":
            d = -torch.nn.CosineSimilarity(dim=2, eps=1e-08)(x1, x2)

        q = (1.0 + (d / self.alpha)).pow(-1)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = q / torch.sum(q, dim=1).unsqueeze(1)
        return q

    def target_distribution(self, q):
        weight = q**2 / q.sum(0).unsqueeze(0).repeat(q.shape[0], 1)
        return weight / weight.sum(1).unsqueeze(1).repeat(1, self.nClusters)


class PrototypicalLayer(torch.nn.Module):
    def __init__(self, nClusters, zdim, centroids_init, alpha=1, distance="l2"):
        super(PrototypicalLayer, self).__init__()
        self.nClusters = nClusters
        self.zdim = zdim
        self.alpha = alpha
        self.clusters = torch.nn.Parameter(
            torch.tensor(centroids_init, requires_grad=True)
        )
        self.distance = distance

    def forward(self, x):
        x1 = x.unsqueeze(1).repeat(1, self.nClusters, 1)  # Embeddings
        x2 = self.clusters.unsqueeze(0).repeat(x.shape[0], 1, 1)  # Cluster centers

        if self.distance == "l2":
            d = -torch.sum(torch.square(x1 - x2), dim=2)
        elif self.distance == "cosine":
            d = torch.nn.CosineSimilarity(dim=2, eps=1e-08)(x1, x2)

        p = torch.softmax(d, dim=1)
        return p


# ---- UTILS ----


def plot_image(x, y=None, denorm_intensity=False, channel_first=True):
    if len(x.shape) < 3:
        x = np.expand_dims(x, 0)
    if channel_first:
        x = np.transpose(x, (1, 2, 0))
    if denorm_intensity:
        x = (x * 127.5) + 127.5
        x = x.astype(int)

    plt.imshow(x)
    if y is not None:
        y = np.expand_dims(y[0, :, :], -1)
        plt.imshow(y, cmap="jet", alpha=0.1)
    plt.axis("off")
    plt.show()


def grad_cam(
    activations, output, normalization="relu_min_max", avg_grads=False, norm_grads=False
):
    def normalize(grads):
        l2_norm = torch.sqrt(torch.mean(torch.pow(grads, 2), (-1, -2, -3))) + 1e-5
        l2_norm = l2_norm.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return grads * torch.pow(l2_norm, -1)

    gradients = torch.autograd.grad(
        output,
        activations,
        grad_outputs=None,
        retain_graph=True,
        create_graph=True,
        only_inputs=True,
        allow_unused=True,
    )[0]

    if norm_grads:
        gradients = normalize(gradients)

    if avg_grads:
        gradients = torch.mean(gradients, dim=[2, 3])
        gradients = gradients.unsqueeze(-1).unsqueeze(-1)

    if "relu" in normalization:
        GCAM = torch.sum(torch.relu(gradients * activations), 1)
    else:
        GCAM = torch.sum(gradients * activations, 1)

    if "tanh" in normalization:
        GCAM = torch.tanh(GCAM)
    if "sigm" in normalization:
        GCAM = torch.sigmoid(GCAM)
    if "abs" in normalization:
        GCAM = torch.abs(GCAM)
    if "min" in normalization:
        norm_value = (
            torch.min(torch.max(GCAM, -1)[0], -1)[0].unsqueeze(-1).unsqueeze(-1) + 1e-3
        )
        GCAM = GCAM - norm_value
    if "max" in normalization:
        norm_value = (
            torch.max(torch.max(GCAM, -1)[0], -1)[0].unsqueeze(-1).unsqueeze(-1) + 1e-3
        )
        GCAM = GCAM * norm_value.pow(-1)
    if "clamp" in normalization:
        GCAM = GCAM.clamp(max=1)

    return GCAM


# ---- MAIN FUNCTIONALITY ----


def train_prototypical_networks(
    dir_dataset,
    items,
    input_shape,
    repetitions=10,
    n=4,
    m=4,
    n_blocks=2,
    epochs=200,
    learning_rate=5e-3,
    distance_train="l2",
    distance_inference="l2",
    n_folds=4,
    n_classes=2,
    deterministic=False,
    projection=True,
    detach=False,
    pretrained=False,
    l2_norm=True,
    contrastive_loss=True,
    prototypical_inference=True,
):
    # Set data generator
    dataset_cruces = Dataset(
        dir_dataset=dir_dataset + "cruces/", items=items, input_shape=input_shape
    )
    dataset_cruces.label_cruces_adif()  # Labels from adif-ineco method

    accuracy_repetitions = []
    precision_repetitions = []
    recall_repetitions = []
    cm_repetitions = []
    f1_repetitions = []

    for i_repetition in range(repetitions):
        X_all_cruces = dataset_cruces.X
        labels_all_cruces = dataset_cruces.labels
        preds_all = []
        refs_all = []
        idx_samples = np.arange(0, dataset_cruces.labels.shape[0])
        random.seed(i_repetition)
        random.shuffle(idx_samples)
        step = round(dataset_cruces.labels.shape[0] / n_folds)

        for iFold in range(n_folds):
            print("Fold " + str(iFold + 1) + "/" + str(n_folds))

            # Set cross-validation samples
            idx_test = list(idx_samples[iFold * step : (iFold + 1) * step])
            idx_train = list(idx_samples[0 : iFold * step]) + list(
                idx_samples[(iFold + 1) * step :]
            )

            labels_cruces_train = labels_all_cruces[idx_train]
            X_cruces_train = X_all_cruces[idx_train]
            X_cruces_test = X_all_cruces[idx_test]

            # Set train generator
            dataset_cruces_fold = dataset_cruces
            dataset_cruces_fold.indexes = idx_train
            data_generator_main = GeneratorFSL(
                dataset_cruces_fold, n=n, m=m, shuffle=True, classes=[0, 1]
            )

            # Set model architectures
            E = Resnet(2, n_blocks=n_blocks, pretrained=pretrained)
            if not detach:
                params = list(E.parameters())
            else:
                params = []

            if deterministic:
                Classifier = torch.nn.Sequential(
                    torch.nn.Linear(E.nfeats, n_classes)
                ).cuda()
                params += list(Classifier.parameters())

            if projection:
                Proj = torch.nn.Sequential(
                    torch.nn.Linear(E.nfeats, E.nfeats // 4),
                    torch.nn.ReLU(),
                    torch.nn.Linear(E.nfeats // 4, E.nfeats // 4),
                )
            else:
                Proj = torch.nn.Sequential()
            params += list(Proj.parameters())

            Lcontrastive = SupConLoss().cuda()
            opt = torch.optim.Adam(params, lr=learning_rate)

            E.cuda()
            Proj.cuda()

            for i_epoch in range(epochs):
                for i_iteration, (Xs, Xq, Y) in enumerate(data_generator_main):
                    opt.zero_grad()
                    Xs = torch.tensor(Xs).cuda().float()
                    Xq = torch.tensor(Xq).cuda().float()
                    Y = torch.tensor(Y).cuda().float()

                    zq = torch.squeeze(torch.nn.AdaptiveAvgPool2d(1)(E(Xq)[0]))
                    zq = Proj(zq)
                    if l2_norm:
                        zq = torch.nn.functional.normalize(zq, dim=1)

                    if deterministic:
                        logits = Classifier(zq)
                    else:
                        E.eval()
                        x_ = Xs.view(
                            Xs.shape[0] * Xs.shape[1],
                            Xs.shape[2],
                            Xs.shape[3],
                            Xs.shape[4],
                        )
                        zs = torch.squeeze(torch.nn.AdaptiveAvgPool2d(1)(E(x_)[0]))
                        zs = Proj(zs)
                        if l2_norm:
                            zs = torch.nn.functional.normalize(zs, dim=1)
                        zs = zs.view(Xs.shape[0], Xs.shape[1], zs.shape[-1])
                        E.train()

                        C = torch.mean(zs, dim=1)
                        x1 = zq.unsqueeze(1).repeat(1, 2, 1)
                        x2 = C.unsqueeze(0).repeat(x1.shape[0], 1, 1)

                        if distance_train == "l2":
                            logits = -torch.sum(torch.square(x1 - x2), dim=2)
                        elif distance_train == "cosine":
                            logits = torch.nn.CosineSimilarity(dim=2, eps=1e-08)(x1, x2)

                    p = torch.softmax(logits, dim=1)

                    if not contrastive_loss:
                        Lclust_iteration = torch.nn.BCELoss()(p, Y)
                    else:
                        Lclust_iteration = Lcontrastive(zq.unsqueeze(1), Y[:, 1])
                    Lclust_iteration.backward()
                    opt.step()

                E.eval()

                centroids = []
                for iCluster in [0, 1]:
                    xx = X_cruces_train[
                        np.array(list(np.argwhere(labels_cruces_train == iCluster)))[
                            :, 0
                        ]
                    ]
                    xx = torch.tensor(xx).cuda().float()
                    zz = torch.squeeze(torch.nn.AdaptiveAvgPool2d(1)(E(xx)[0]))
                    if not contrastive_loss:
                        zz = Proj(zz)
                        if l2_norm:
                            zz = torch.nn.functional.normalize(zz, dim=1)
                    centroids.append(zz.mean(0).unsqueeze(0))

                z_support = torch.cat(centroids)
                x_cruce = torch.tensor(X_cruces_test).cuda().float()
                z_cruce, F = E(x_cruce)
                z_cruce = torch.squeeze(torch.nn.AdaptiveAvgPool2d(1)(z_cruce))

                if not contrastive_loss:
                    z_cruce = Proj(z_cruce)
                    if l2_norm:
                        z_cruce = torch.nn.functional.normalize(z_cruce, dim=1)

                x1 = z_cruce.unsqueeze(1).repeat(1, 2, 1)
                x2 = z_support.unsqueeze(0).repeat(x1.shape[0], 1, 1)

                if not prototypical_inference:
                    logits = Classifier(z_cruce)
                else:
                    if distance_inference == "l2":
                        logits = -torch.sum(torch.square(x1 - x2), dim=2)
                    elif distance_inference == "cosine":
                        logits = torch.nn.CosineSimilarity(dim=2, eps=1e-08)(x1, x2)

                p = torch.softmax(logits, dim=1)
                preds = np.argmax(p.detach().cpu().numpy(), 1)
                refs = labels_all_cruces[idx_test]

                acc = accuracy_score(refs, preds)
                f1 = f1_score(refs, preds, average="macro")

                preds_all.extend(list(preds))
                refs_all.extend(list(refs))

            preds_all = np.array(preds_all)
            refs_all = np.array(refs_all)
            cm = confusion_matrix(refs_all, preds_all, labels=[0, 1])

            acc = accuracy_score(refs_all, preds_all)
            precision, recall, _, _ = precision_recall_fscore_support(
                refs_all, preds_all
            )
            precision = precision[1]
            recall = recall[1]
            f1 = 2 * (precision * recall) / ((precision + recall) + 1e-3)

            print("-" * 40)
            print("Repetition metrics: ")
            print(
                f"Accuracy={acc:.6f} ; Precision={precision:.6f} ; Recall={recall:.6f} ; F1={f1:.6f}"
            )
            print("-" * 40)

            accuracy_repetitions.append(acc)
            precision_repetitions.append(precision)
            recall_repetitions.append(recall)
            f1_repetitions.append(f1)
            cm_repetitions.append(cm)

    acc_avg = np.mean(accuracy_repetitions)
    acc_std = np.std(accuracy_repetitions)
    precision_avg = np.mean(precision_repetitions)
    precision_std = np.std(precision_repetitions)
    recall_avg = np.mean(recall_repetitions)
    recall_std = np.std(recall_repetitions)
    f1_avg = np.mean(f1_repetitions)
    f1_std = np.std(f1_repetitions)
    cm_avg = np.mean(np.array(cm_repetitions), axis=0)

    print("-" * 40)
    print("Overall Repetitions metrics: ")
    print(cm_avg)
    print(
        f"Accuracy={acc_avg:.6f}({acc_std:.6f}) ; Precision={precision_avg:.6f}({precision_std:.6f})  ; Recall={recall_avg:.6f}({recall_std:.6f}) ; F1={f1_avg:.6f}({f1_std:.6f})"
    )
    print("-" * 40)

    return {
        "accuracy": accuracy_repetitions,
        "precision": precision_repetitions,
        "recall": recall_repetitions,
        "f1": f1_repetitions,
        "confusion_matrix": cm_avg,
    }
