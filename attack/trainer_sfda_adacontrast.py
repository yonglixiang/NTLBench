"""
Builds upon: https://github.com/DianCh/AdaContrast, https://github.com/tntek/source-free-domain-adaptation
Corresponding paper: https://openaccess.thecvf.com/content/CVPR2022/papers/Chen_Contrastive_Test-Time_Adaptation_CVPR_2022_paper.pdf
"""

import torch
from torch import nn, optim
import torch.nn.functional as F
import utils
import wandb
import numpy as np
from torch.utils.data import DataLoader
from copy import deepcopy
from torch.nn.utils.weight_norm import WeightNorm
from torchvision import transforms
from sklearn.metrics import accuracy_score
from PIL import ImageFilter
import random, time, math
import torch.distributed as dist

from utils.sfda_utils import SFDA_Trainset, op_copy, set_lr, MODEL_FEATURE_DIM
from utils.sfda_loss import Entropy

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        # logging.info("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"
    
def adjust_learning_rate(optimizer, progress, args):
    """
    Decay the learning rate based on epoch or iteration.
    """
    if args.ADACONTRAST_OPTIM_COS:
        decay = 0.5 * (1.0 + math.cos(math.pi * progress / args.ADACONTRAST_FULL_PROGRESS))
    elif args.ADACONTRAST_OPTIM_EXP:
        decay = (1 + 10 * progress / args.ADACONTRAST_FULL_PROGRESS) ** -0.75
    else:
        decay = 1.0
        for milestone in args.ADACONTRAST_SCHEDULE:
            decay *= args.ADACONTRAST_GAMMA if progress >= milestone else 1.0
    for param_group in optimizer.param_groups:
        param_group["lr"] = param_group["lr0"] * decay

    return decay

def del_wn_hook(model):
    for module in model.modules():
        for _, hook in module._forward_pre_hooks.items():
            if isinstance(hook, WeightNorm):
                delattr(module, hook.name)

def restore_wn_hook(model, name='weight'):
    for module in model.modules():
        for _, hook in module._forward_pre_hooks.items():
            if isinstance(hook, WeightNorm):
                hook(module, name)

def deepcopy_model(model):
    ### del weight norm hook
    del_wn_hook(model)

    ### copy model
    model_cp = deepcopy(model)

    ### restore weight norm hook
    restore_wn_hook(model)
    restore_wn_hook(model_cp)

    return model_cp

def classification_loss(logits_w, logits_s, target_labels, cfg):
    if cfg.ADACONTRAST_CE_SUP_TYPE == "weak_weak":
        loss_cls = cross_entropy_loss(logits_w, target_labels, cfg)
        accuracy = calculate_acc(logits_w, target_labels)
    elif cfg.ADACONTRAST_CE_SUP_TYPE == "weak_strong":
        loss_cls = cross_entropy_loss(logits_s, target_labels, cfg)
        accuracy = calculate_acc(logits_s, target_labels)
    else:
        raise NotImplementedError(
            f"{cfg.ADACONTRAST_CE_SUP_TYPE} CE supervision type not implemented."
        )
    return loss_cls, accuracy

def cross_entropy_loss(logits, labels, cfg):
    if cfg.ADACONTRAST_CE_TYPE == "standard":
        return F.cross_entropy(logits, labels)
    raise NotImplementedError(f"{cfg.ADACONTRAST_CE_TYPE} CE loss is not implemented.")

def instance_loss(logits_ins, pseudo_labels, mem_labels, contrast_type):
    # labels: positive key indicators
    labels_ins = torch.zeros(logits_ins.shape[0], dtype=torch.long).cuda()

    # in class_aware mode, do not contrast with same-class samples
    if contrast_type == "class_aware" and pseudo_labels is not None:
        mask = torch.ones_like(logits_ins, dtype=torch.bool)
        mask[:, 1:] = pseudo_labels.reshape(-1, 1) != mem_labels  # (B, K)
        logits_ins = torch.where(mask, logits_ins, torch.tensor([float("-inf")]).cuda())

    loss = F.cross_entropy(logits_ins, labels_ins)

    accuracy = calculate_acc(logits_ins, labels_ins)

    return loss, accuracy

def diversification_loss(logits_w, logits_s, cfg):
    if cfg.ADACONTRAST_CE_SUP_TYPE == "weak_weak":
        loss_div = div(logits_w)
    elif cfg.ADACONTRAST_CE_SUP_TYPE == "weak_strong":
        loss_div = div(logits_s)
    else:
        loss_div = div(logits_w) + div(logits_s)

    return loss_div

@torch.no_grad()
def calculate_acc(logits, labels):
    preds = logits.argmax(dim=1)
    accuracy = (preds == labels).float().mean() * 100
    return accuracy

def get_target_optimizer(model, cfg):
    # if cfg.distributed:
    #     model = model.module
    backbone_params, extra_params = (
        model.src_model.get_params()
        if hasattr(model, "src_model")
        else model.get_params()
    )

    if cfg.OPTIM.METHOD == "sgd":
        optimizer = torch.optim.SGD(
            [
                {
                    "params": backbone_params,
                    "lr": cfg.OPTIM.LR,
                    "momentum": 0.9,
                    "weight_decay": 1e-4,
                    "nesterov": True,
                },
                {
                    "params": extra_params,
                    "lr": cfg.OPTIM.LR * 10,
                    "momentum": 0.9,
                    "weight_decay": 1e-4,
                    "nesterov": True,
                },
            ]
        )
    else:
        raise NotImplementedError(f"{cfg.OPTIM.METHOD} not implemented.")

    for param_group in optimizer.param_groups:
        param_group["lr0"] = param_group["lr"]  # snapshot of the initial lr

    return optimizer


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

class AdaMoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a memory bank
    https://arxiv.org/abs/1911.05722
    """

    def __init__(
        self,
        src_model,
        momentum_model,
        output_dim,
        num_classes,
        K=16384,
        m=0.999,
        T_moco=0.07,
        checkpoint_path=None,
    ):
        """
        dim: feature dimension (default: 128)
        K: buffer size; number of keys
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(AdaMoCo, self).__init__()

        self.K = K
        self.m = m
        self.T_moco = T_moco
        self.queue_ptr = 0

        # create the encoders
        self.src_model = src_model
        self.momentum_model = momentum_model

        # create the fc heads
        feature_dim = output_dim

        # freeze key model
        self.momentum_model.requires_grad_(False)

        # create the memory bank
        self.register_buffer("mem_feat", torch.randn(feature_dim, K, device="cuda"))
        self.register_buffer(
            "mem_labels", torch.randint(0, num_classes, (K,), device="cuda")
        )
        self.mem_feat = F.normalize(self.mem_feat, dim=0).cuda()

        if checkpoint_path:
            self.load_from_checkpoint(checkpoint_path)

    def load_from_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = dict()
        for name, param in checkpoint["state_dict"].items():
            # get rid of 'module.' prefix brought by DDP
            name = name[len("module.") :] if name.startswith("module.") else name
            state_dict[name] = param
        msg = self.load_state_dict(state_dict, strict=False)
        # logging.info(
        #     f"Loaded from {checkpoint_path}; missing params: {msg.missing_keys}"
        # )

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        # encoder_q -> encoder_k
        for param_q, param_k in zip(
            self.src_model.parameters(), self.momentum_model.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def update_memory(self, keys, pseudo_labels):
        """
        Update features and corresponding pseudo labels
        """
        # gather keys before updating queue
        # keys = concat_all_gather(keys)
        # pseudo_labels = concat_all_gather(pseudo_labels)

        start = self.queue_ptr
        end = start + len(keys)
        idxs_replace = torch.arange(start, end) % self.K
        self.mem_feat[:, idxs_replace] = keys.T
        self.mem_labels[idxs_replace] = pseudo_labels
        self.queue_ptr = end % self.K

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q, im_k=None, cls_only=False):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            feats_q: <B, D> query image features before normalization
            logits_q: <B, C> logits for class prediction from queries
            logits_ins: <B, K> logits for instance prediction
            k: <B, D> contrastive keys
        """

        # compute query features
        logits_q, feats_q = self.src_model.forward_f(im_q)
        
        # if 'image' in dset:
        #     feats_q = self.src_model.netF(im_q)
        #     if 'k' in dset:
        #         logits_q = self.src_model.netC(feats_q)
        #     else:
        #         logits_q = self.src_model.masking_layer(self.src_model.netC(feats_q))
        # else :
        #     feats_q = self.src_model.netB(self.src_model.netF(im_q))
        #     logits_q = self.src_model.netC(feats_q)
        if cls_only:
            return feats_q, logits_q

        q = F.normalize(feats_q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            # im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            # if 'image' in dset:
            #     k = self.momentum_model.netF(im_k)
            #     # logits_q = self.src_model.masking_layer(self.src_model.netC(feats_q))
            # else :
            #     k = self.momentum_model.netB(self.momentum_model.netF(im_k))
            _, k = self.momentum_model.forward_f(im_k)
            k = F.normalize(k, dim=1)

            # undo shuffle
            # k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum("nc,ck->nk", [q, self.mem_feat.clone().detach()])

        # logits: Nx(1+K)
        logits_ins = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits_ins /= self.T_moco

        # dequeue and enqueue will happen outside
        return feats_q, logits_q, logits_ins, k

def entropy(p, axis=1):
    return -torch.sum(p * torch.log2(p + 1e-5), dim=axis)

def get_distances(X, Y, dist_type="cosine"):
    if dist_type == "euclidean":
        distances = torch.cdist(X, Y)
    elif dist_type == "cosine":
        distances = 1 - torch.matmul(F.normalize(X, dim=1), F.normalize(Y, dim=1).T)
    else:
        raise NotImplementedError(f"{dist_type} distance not implemented.")

    return distances

@torch.no_grad()
def soft_k_nearest_neighbors(features, features_bank, probs_bank, cfg):
    pred_probs = []
    for feats in features.split(64):
        distances = get_distances(feats, features_bank, cfg.ADACONTRAST_DIST_TYPE)
        _, idxs = distances.sort()
        idxs = idxs[:, : cfg.ADACONTRAST_NUM_NEIGHBORS]
        # (64, num_nbrs, num_classes), average over dim=1
        probs = probs_bank[idxs, :].mean(1)
        pred_probs.append(probs)
    pred_probs = torch.cat(pred_probs)
    _, pred_labels = pred_probs.max(dim=1)

    return pred_labels, pred_probs

@torch.no_grad()
def refine_predictions(
    features,
    probs,
    banks,
    cfg,
    gt_labels=None,
):
    if cfg.ADACONTRAST_REFINE_METHOD == "nearest_neighbors":
        feature_bank = banks["features"]
        probs_bank = banks["probs"]
        pred_labels, probs = soft_k_nearest_neighbors(
            features, feature_bank, probs_bank, cfg
        )
    elif cfg.ADACONTRAST_REFINE_METHOD is None:
        pred_labels = probs.argmax(dim=1)
    else:
        raise NotImplementedError(
            f"{cfg.ADACONTRAST_REFINE_METHOD} refine method is not implemented."
        )
    accuracy = None
    if gt_labels is not None:
        accuracy = (pred_labels == gt_labels).float().mean() * 100

    return pred_labels, probs, accuracy

def contrastive_loss(logits_ins, pseudo_labels, mem_labels):
    # labels: positive key indicators
    labels_ins = torch.zeros(logits_ins.shape[0], dtype=torch.long).cuda()

    mask = torch.ones_like(logits_ins, dtype=torch.bool)
    mask[:, 1:] = torch.all(pseudo_labels.unsqueeze(1) != mem_labels.unsqueeze(0), dim=2)
    logits_ins = torch.where(mask, logits_ins, torch.tensor([float("-inf")]).cuda())

    loss = F.cross_entropy(logits_ins, labels_ins)

    return loss

@torch.no_grad()
def update_labels(banks, idxs, features, logits, cfg):
    # 1) avoid inconsistency among DDP processes, and
    # 2) have better estimate with more data points

    probs = F.softmax(logits, dim=1)

    start = banks["ptr"]
    end = start + len(idxs)
    idxs_replace = torch.arange(start, end).cuda() % len(banks["features"])
    banks["features"][idxs_replace, :] = features
    banks["probs"][idxs_replace, :] = probs
    banks["ptr"] = end % len(banks["features"])

def div(logits, epsilon=1e-8):
    probs = F.softmax(logits, dim=1)
    probs_mean = probs.mean(dim=0)
    loss_div = -torch.sum(-probs_mean * torch.log(probs_mean + epsilon))

    return loss_div

def nl_criterion(output, y, num_class):
    output = torch.log(torch.clamp(1. - F.softmax(output, dim=1), min=1e-5, max=1.))

    labels_neg = ((y.unsqueeze(-1).repeat(1, 1) + torch.LongTensor(len(y), 1).random_(1,
                                                                                      num_class).cuda()) % num_class).view(
        -1)

    l = F.nll_loss(output, labels_neg, reduction='none')

    return l

@torch.no_grad()
def eval_and_label_dataset(epoch,loader, model, banks, cfg):
    model.eval()
    logits, indices, gt_labels = [], [], []
    features = []

    for batch_idx, batch in enumerate(loader):
        imgs, targets, idxs= batch
        # imgs, targets, idxs = batch[0].cuda(), batch[1].cuda(), batch[2].cuda()
        inputs = imgs[1].cuda()
        # imgs, targets, idxs= batch
        targets = torch.argmax(targets, dim=1)
        targets, idxs = targets.long().cuda(), idxs.long().cuda()
        # inputs = imgs[0].cuda()
        feats, logits_cls = model(inputs, cls_only=True)
        features.append(feats)
        gt_labels.append(targets)
        logits.append(logits_cls)
        indices.append(idxs)

    features = torch.cat(features)
    gt_labels = torch.cat(gt_labels)
    logits = torch.cat(logits)
    indices = torch.cat(indices)

    probs = F.softmax(logits, dim=1)
    rand_idxs = torch.randperm(len(features)).cuda()
    banks = {
        "features": features[rand_idxs][: 16384],
        "probs": probs[rand_idxs][: 16384],
        "ptr": 0,
    }

    # refine predicted labels
    pred_labels, _, acc = refine_predictions(
        features, probs, banks, cfg=cfg, gt_labels=gt_labels
    )

    log_str = "\n| Test Epoch #%d\t Accuracy: %.2f%%\n" % (epoch, acc)
    # logging.info(log_str)
    
    return acc, banks, gt_labels, pred_labels

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

def get_augmentation(aug_type, normalize=True):
    if normalize:
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    if aug_type == "moco-v2":
        return transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(64, scale=(0.2, 1.0)),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],
                    p=0.8,  # not strengthened
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
    elif aug_type == "moco-v1":
        return transforms.Compose(
            [   
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(64, scale=(0.2, 1.0)),
                transforms.RandomGrayscale(p=0.2),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
    elif aug_type == "plain":
        return transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((64, 64)),
                transforms.RandomCrop(64),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
    elif aug_type == "clip_inference":
        return transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(64, interpolation=Image.BICUBIC),
                transforms.CenterCrop(64),
                transforms.ToTensor(),
                normalize,
            ]
        )
    elif aug_type == "test":
        return transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((64, 64)),
                transforms.CenterCrop(64),
                transforms.ToTensor(),
                normalize,
            ]
        )
    return None

class NCropsTransform:
    def __init__(self, transform_list) -> None:
        self.transform_list = transform_list

    def __call__(self, x):
        data = [tsfm(x) for tsfm in self.transform_list]
        return data

def get_augmentation_versions(cfg):
    """
    Get a list of augmentations. "w" stands for weak, "s" stands for strong.

    E.g., "wss" stands for one weak, two strong.
    """
    transform_list = []
    for version in 'twss':
        if version == "s":
            transform_list.append(get_augmentation("moco-v2"))
        elif version == "w":
            transform_list.append(get_augmentation("plain"))
        elif version == 't':
            transform_list.append(get_augmentation("test"))
        else:
            raise NotImplementedError(f"{version} version not implemented.")
    transform = NCropsTransform(transform_list)

    return transform

def train_step(train_loader, model, banks, optimizer, epoch, cfg):
    batch_time = AverageMeter("Time", ":6.3f")
    loss_meter = AverageMeter("Loss", ":.4f")
    top1_ins = AverageMeter("SSL-Acc@1", ":6.2f")
    top1_psd = AverageMeter("CLS-Acc@1", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, loss_meter, top1_ins, top1_psd],
        prefix=f"Epoch: [{epoch}]",
    )

    # make sure to switch to train mode
    model.train()

    end = time.time()
    zero_tensor = torch.tensor([0.0]).to("cuda")
    for i, data in enumerate(train_loader):
        # unpack and move data
        images, _, idxs = data
        idxs = idxs.to("cuda")
        images_w, images_q, images_k = (
            images[0].to("cuda"),
            images[1].to("cuda"),
            images[2].to("cuda"),
        )

        # per-step scheduler
        step = i + epoch * len(train_loader)
        adjust_learning_rate(optimizer, step, cfg)
        feats_w, logits_w = model(images_w, cls_only=True)
        with torch.no_grad():
            probs_w = F.softmax(logits_w, dim=1)
            pseudo_labels_w, probs_w, _ = refine_predictions(
                feats_w, probs_w, banks, cfg=cfg
            )
        _, logits_q, logits_ins, keys = model(images_q, images_k)

        # update key features and corresponding pseudo labels
        model.update_memory(keys, pseudo_labels_w)

        # moco instance discrimination
        loss_ins, accuracy_ins = instance_loss(
            logits_ins=logits_ins,
            pseudo_labels=pseudo_labels_w,
            mem_labels=model.mem_labels,
            contrast_type='class_aware',
        )
        # instance accuracy shown for only one process to give a rough idea
        top1_ins.update(accuracy_ins.item(), len(logits_ins))

        # classification
        loss_cls, accuracy_psd = classification_loss(
            logits_w, logits_q, pseudo_labels_w, cfg
        )
        top1_psd.update(accuracy_psd.item(), len(logits_w))

        # diversification
        loss_div = (
            diversification_loss(logits_w, logits_q, cfg)
            if cfg.ADACONTRAST_ETA > 0
            else zero_tensor
        )

        loss = (
            cfg.ADACONTRAST_ALPHA * loss_cls
            + cfg.ADACONTRAST_BETA * loss_ins
            + cfg.ADACONTRAST_ETA * loss_div
        )
        loss_meter.update(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # use slow feature to update neighbor space
        with torch.no_grad():
            # if 'image' in cfg.SETTING.DATASET:
            #     feats_w = model.momentum_model.netF(images_w)
            #     if 'k' in cfg.SETTING.DATASET:
            #         logits_w = model.momentum_model.netC(feats_w)
            #     else:
            #         logits_w = model.momentum_model.masking_layer(model.momentum_model.netC(feats_w))
            # else :
            #     feats_w = model.momentum_model.netB(model.momentum_model.netF(images_w))
            #     logits_w = model.momentum_model.netC(feats_w)

            logits_w, feats_w = model.momentum_model.forward_f(images_w)

        update_labels(banks, idxs, feats_w, logits_w, cfg)

        batch_time.update(time.time() - end)
        end = time.time()

        # if i % cfg.print_freq == 0:
        #     progress.display(i)
    return loss

def ada_contrast(config, dataloader_train_srgt, dataloader_val, dataloader_test, 
         model, datasets_name):
    # shot train dataloader
    aug_transform = get_augmentation_versions(config)
    datafile_ta = SFDA_Trainset(dataloader_train_srgt[1], transform=aug_transform, mode='train')
    datafile_te = SFDA_Trainset(dataloader_train_srgt[1], transform=aug_transform, mode='train')
    # datafile_ta = SFDA_Trainset(dataloader_val[1])
    # datafile_te = SFDA_Trainset(dataloader_val[1])
    trainloader_ta = DataLoader(datafile_ta, batch_size=config.batch_size, 
                             shuffle=True, num_workers=config.num_workers, drop_last=True)
    trainloader_te = DataLoader(datafile_te, batch_size=config.batch_size*3, 
                             shuffle=False, num_workers=config.num_workers, drop_last=False)
    # evaluate dataloader
    if 'cmt' in config.domain_src:
        topk = (1, )  # 2-classification, use top1 acc
    else:
        topk = (1, 5)  # 10-classification, use top1 and top5 acc
    moco_dataloader_val = [  DataLoader(SFDA_Trainset(v, transform=aug_transform), batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, drop_last=False)
                        for v in dataloader_val]
    moco_dataloader_test = [  DataLoader(SFDA_Trainset(v, transform=aug_transform), batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, drop_last=False)
                        for v in dataloader_test]
    evaluators_val = [utils.evaluators.classification_evaluator(
        v, topk, is_moco=True) for v in moco_dataloader_val]
    evaluators_test = [utils.evaluators.classification_evaluator(
        v, topk, is_moco=True) for v in moco_dataloader_test]
    
    # choose the model with the maximum Acc sum of source and target domain
    bestlogger = utils.evaluators.attack_ntl_logger_bestsum()

    # initialize the model and optimizer
    base_model = model
    momentun_model = deepcopy_model(base_model)   
    moco_model = AdaMoCo(
        src_model = base_model,
        momentum_model = momentun_model,
        output_dim=MODEL_FEATURE_DIM[config.teacher_network],
        num_classes=config.num_classes)
    CE = nn.CrossEntropyLoss(reduction='none')
    acc, banks, _, _ = eval_and_label_dataset(0, trainloader_ta, moco_model, banks=None, cfg=config)
    wandb.config.update({
        "ADACONTRAST_FULL_PROGRESS": config.surrogate_epochs * len(trainloader_ta)
    }, allow_val_change=True)
    
    optimizer = optim.SGD(base_model.parameters(), lr=config.surrogate_lr, weight_decay=1e-4,momentum=0.9,nesterov=True)
    for param_group in optimizer.param_groups:
        param_group["lr0"] = param_group["lr"]  # snapshot of the initial lr
    
    bestlogger = utils.evaluators.attack_ntl_logger_bestsum()
    
    moco_model.train()
    for epoch in range(config.surrogate_epochs):
        # train
        loss = train_step(trainloader_ta, moco_model, banks, optimizer, epoch, config)
        
        moco_model.eval()
            
        # validation
        if topk == (1, 5):
            acc1s, _ = utils.evaluators.eval_func(config, evaluators_val, moco_model, is_moco=True)
        elif topk == (1, ):
            acc = utils.evaluators.eval_func(config, evaluators_val, moco_model, is_moco=True)
            acc1s = acc[0]
        tgt_mean = torch.mean(torch.tensor(acc1s[1:])).item()
        if epoch == 0 or bestlogger.log(acc1s[0], tgt_mean):
            if topk == (1, 5):
                test_acc1s, _ = utils.evaluators.eval_func(config, evaluators_test, moco_model, is_moco=True)
            elif topk == (1, ):
                test_acc = utils.evaluators.eval_func(config, evaluators_test, moco_model, is_moco=True)
                test_acc1s = test_acc[0]
        
        wandb_log_dict = {
            'epoch_ft': epoch,
            'loss_ft': loss,
            'Acc_ft_src': acc1s[0]}

        # print validation
        print('[SFDA-AdaContrast] | epoch %03d | train_loss: %.3f | src_val_acc1: %.1f, tgt_val_acc1: ' %
            (epoch, loss, acc1s[0]), end='')
        for dname, acc_tgt in zip(datasets_name[1:], acc1s[1:]):
            print(f'{dname}: {acc_tgt:.2f} ', end='')
            wandb_log_dict[f'Acc_ft_tgt_{dname}'] = acc_tgt
        print('')
        wandb.log(wandb_log_dict)
            
        moco_model.train()
    
    wandb.run.summary['final_valbest_Acc_ft_src'] = bestlogger.result()['src']
    wandb.run.summary['final_valbest_Acc_ft_tgtmean'] = bestlogger.result()['tgt']
    wandb.run.summary['final_test_Acc_ft_src'] = test_acc1s[0]
    for dname, acc_tgt in zip(datasets_name[1:], test_acc1s[1:]):
        wandb.run.summary[f'final_test_Acc_ft_tgt_{dname}'] = acc_tgt
        wandb.run.summary[f'final_test_Acc_ft_tgt'] = acc_tgt
    
    print(f"Val Best Acc (Source): {bestlogger.result()['src']:.2f}")
    print(f"Val Best Acc (Target Mean): {bestlogger.result()['tgt']:.2f}")
    print(f"Test Acc (Source): {test_acc1s[0]:.2f}")
    for dname, acc_tgt in zip(datasets_name[1:], test_acc1s[1:]):
        print(f"Test Acc (Target {dname}): {acc_tgt:.2f}")
    
    return moco_model