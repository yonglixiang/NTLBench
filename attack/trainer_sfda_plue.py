"""
Builds upon: https://github.com/MattiaLitrico/Guiding-Pseudo-labels-with-Uncertainty-Estimation-for-Source-free-Unsupervised-Domain-Adaptation, https://github.com/tntek/source-free-domain-adaptation
Corresponding paper: https://openaccess.thecvf.com/content/CVPR2023/papers/Litrico_Guiding_Pseudo-Labels_With_Uncertainty_Estimation_for_Source-Free_Unsupervised_Domain_Adaptation_CVPR_2023_paper.pdf
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
import random

from utils.sfda_utils import SFDA_Trainset, op_copy, set_lr, MODEL_FEATURE_DIM
from utils.sfda_loss import Entropy

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

class AdaMoCo(nn.Module):
    def __init__(self, src_model, momentum_model, features_length, num_classes, dataset_length, temporal_length):
        super(AdaMoCo, self).__init__()

        self.m = 0.999

        self.first_update = True

        self.src_model = src_model
        self.momentum_model = momentum_model

        self.momentum_model.requires_grad_(False)

        self.queue_ptr = 0
        self.mem_ptr = 0

        self.T_moco = 0.07

        # queue length
        self.K = min(16384, dataset_length)
        self.memory_length = temporal_length

        self.register_buffer("features", torch.randn(features_length, self.K))
        self.register_buffer(
            "labels", torch.randint(0, num_classes, (self.K,))
        )
        self.register_buffer(
            "idxs", torch.randint(0, dataset_length, (self.K,))
        )
        self.register_buffer(
            "mem_labels", torch.randint(0, num_classes, (dataset_length, self.memory_length))
        )

        self.register_buffer(
            "real_labels", torch.randint(0, num_classes, (dataset_length,))
        )

        self.features = F.normalize(self.features, dim=0)

        self.features = self.features.cuda()
        self.labels = self.labels.cuda()
        self.mem_labels = self.mem_labels.cuda()
        self.real_labels = self.real_labels.cuda()
        self.idxs = self.idxs.cuda()

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        # encoder_q -> encoder_k
        for param_q, param_k in zip(
                self.src_model.parameters(), self.momentum_model.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def update_memory(self, epoch, idxs, keys, pseudo_labels, real_label):
        start = self.queue_ptr
        end = start + len(keys)
        idxs_replace = torch.arange(start, end).cuda() % self.K
        self.features[:, idxs_replace] = keys.T
        self.labels[idxs_replace] = pseudo_labels
        self.idxs[idxs_replace] = idxs
        self.real_labels[idxs_replace] = real_label
        self.queue_ptr = end % self.K

        self.mem_labels[idxs, self.mem_ptr] = pseudo_labels
        self.mem_ptr = epoch % self.memory_length

    @torch.no_grad()
    def get_memory(self):
        return self.features, self.labels

    def forward(self, im_q, im_k=None, cls_only=False):
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
            # if 'image' in dset:
            #     k = self.momentum_model.netF(im_k)
            #     # logits_q = self.src_model.masking_layer(self.src_model.netC(feats_q))
            # else :
            #     k = self.momentum_model.netB(self.momentum_model.netF(im_k))
            _, k = self.momentum_model.forward_f(im_k)
            k = F.normalize(k, dim=1)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum("nc,ck->nk", [q, self.features.clone().detach()])

        # logits: Nx(1+K)
        logits_ins = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits_ins /= self.T_moco

        # dequeue and enqueue will happen outside
        return feats_q, logits_q, logits_ins, k


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
def soft_k_nearest_neighbors(features, features_bank, probs_bank, num_neighbors):
    pred_probs = []
    pred_probs_all = []

    for feats in features.split(64):
        distances = get_distances(feats, features_bank)
        _, idxs = distances.sort()
        idxs = idxs[:, : num_neighbors]
        # (64, num_nbrs, num_classes), average over dim=1
        probs = probs_bank[idxs, :].mean(1)
        pred_probs.append(probs)
        # (64, num_nbrs, num_classes)
        probs_all = probs_bank[idxs, :]
        pred_probs_all.append(probs_all)

    pred_probs_all = torch.cat(pred_probs_all)
    pred_probs = torch.cat(pred_probs)

    _, pred_labels = pred_probs.max(dim=1)
    # (64, num_nbrs, num_classes), max over dim=2
    _, pred_labels_all = pred_probs_all.max(dim=2)
    # First keep maximum for all classes between neighbors and then keep max between classes
    _, pred_labels_hard = pred_probs_all.max(dim=1)[0].max(dim=1)

    return pred_labels, pred_probs, pred_labels_all, pred_labels_hard


def refine_predictions(
        features,
        probs,
        banks,
        num_neighbors):
    feature_bank = banks["features"]
    probs_bank = banks["probs"]
    pred_labels, probs, pred_labels_all, pred_labels_hard = soft_k_nearest_neighbors(
        features, feature_bank, probs_bank, num_neighbors
    )

    return pred_labels, probs, pred_labels_all, pred_labels_hard


def contrastive_loss(logits_ins, pseudo_labels, mem_labels):
    # labels: positive key indicators
    labels_ins = torch.zeros(logits_ins.shape[0], dtype=torch.long).cuda()

    mask = torch.ones_like(logits_ins, dtype=torch.bool)
    mask[:, 1:] = torch.all(pseudo_labels.unsqueeze(1) != mem_labels.unsqueeze(0), dim=2)
    logits_ins = torch.where(mask, logits_ins, torch.tensor([float("-inf")]).cuda())

    loss = F.cross_entropy(logits_ins, labels_ins)

    return loss


@torch.no_grad()
def update_labels(banks, idxs, features, logits):
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
def eval_and_label_dataset(epoch, model, loader,cfg):
    model.eval()
    logits, indices, gt_labels = [], [], []
    features = []

    for batch_idx, batch in enumerate(loader):
        imgs, targets, idxs= batch
        # imgs, targets, idxs = batch[0].cuda(), batch[1].cuda(), batch[2].cuda()
        # imgs, targets, idxs= batch
        targets = torch.argmax(targets, dim=1)
        targets, idxs = targets.long().cuda(), idxs.long().cuda()
        inputs = imgs[1].cuda()
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
    pred_labels, _, _, _ = refine_predictions(features, probs, banks, cfg.PLUE_NUM_NEIGHBORS)

    acc = 100. * accuracy_score(gt_labels.to('cpu'), pred_labels.to('cpu'))
    
    return acc, banks, gt_labels, pred_labels


def train_step(epoch, net, moco_model, optimizer, trainloader, banks, cfg, CE):
    loss = 0
    acc = 0

    net.train()
    moco_model.train()
    num_class = cfg.num_classes
    for batch_idx, batch in enumerate(trainloader):
        imgs, y, idxs = batch
        y, idxs = y.long().cuda(), idxs.long().cuda()
        y = torch.argmax(y, dim=1)
        weak_x = imgs[1].cuda()
        strong_x = imgs[2].cuda()
        strong_x2 = imgs[3].cuda()

        feats_w, logits_w = moco_model(weak_x, cls_only=True)

        if cfg.PLUE_LABEL_REFINEMENT:
            with torch.no_grad():
                probs_w = F.softmax(logits_w, dim=1)
                pseudo_labels_w, probs_w, _, _ = refine_predictions(feats_w, probs_w, banks, cfg.PLUE_LABEL_REFINEMENT)
        else:
            probs_w = F.softmax(logits_w, dim=1)
            pseudo_labels_w = probs_w.max(1)[1]

        _, logits_q, logits_ctr, keys = moco_model(strong_x, strong_x2)

        if cfg.PLUE_CTR:
            loss_ctr = contrastive_loss(
                logits_ins=logits_ctr,
                pseudo_labels=moco_model.mem_labels[idxs],
                mem_labels=moco_model.mem_labels[moco_model.idxs]
            )
        else:
            loss_ctr = 0

        # update key features and corresponding pseudo labels
        moco_model.update_memory(epoch, idxs, keys, pseudo_labels_w, y)

        with torch.no_grad():
            # CE weights
            max_entropy = torch.log2(torch.tensor(num_class)+ cfg.PLUE_EPSILON)
            w = entropy(probs_w)

            w = w / max_entropy
            w = torch.exp(-w+cfg.PLUE_EPSILON)

        # Standard positive learning
        if cfg.PLUE_NEG_L:
            # Standard negative learning
            loss_cls = (nl_criterion(logits_q, pseudo_labels_w, num_class)).mean()
            if cfg.PLUE_REWEIGHTING:
                loss_cls = (w * nl_criterion(logits_q, pseudo_labels_w, num_class)).mean()
        else:
            loss_cls = (CE(logits_q, pseudo_labels_w)).mean()
            if cfg.PLUE_REWEIGHTING:
                loss_cls = (w * CE(logits_q, pseudo_labels_w)).mean()

        loss_div = (div(logits_w) + div(logits_q))
        # loss_div = 0

        l = loss_cls + loss_ctr + loss_div

        update_labels(banks, idxs, feats_w, logits_w)

        l.backward()
        optimizer.step()
        optimizer.zero_grad()

        accuracy = 100. * accuracy_score(y.to('cpu'), logits_w.to('cpu').max(1)[1])

        loss += l.item()
        acc += accuracy

        # if batch_idx % 100 == 0:
        #     print('Epoch [%3d/%3d] Iter[%3d/%3d]\t '
        #           % (epoch, cfg.TEST.MAX_EPOCH, batch_idx + 1, len(trainloader)))
        
        #     print("Acc ", acc / (batch_idx + 1))
    f'Training acc =  {epoch}/{cfg.surrogate_epochs} ACC {acc:.2f}%'
    log_str = "Training acc = {:.2f}".format(acc / len(trainloader))
    # logging.info(log_str)
    
    return loss

def PLUE(config, dataloader_train_srgt, dataloader_val, dataloader_test, 
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

    # optimize
    param_group = set_lr(model, config)
    optimizer = torch.optim.SGD(param_group)
    optimizer = op_copy(optimizer)
    

    # initialize the model
    base_model = model
    momentun_model = deepcopy_model(base_model)   
    optimizer = optim.SGD(base_model.parameters(), lr=config.surrogate_lr, weight_decay=5e-4)
    moco_model = AdaMoCo(
        src_model = base_model,
        momentum_model = momentun_model,
        features_length=MODEL_FEATURE_DIM[config.teacher_network],
        num_classes=config.num_classes, 
        dataset_length = len(trainloader_ta.dataset),
        temporal_length=config.PLUE_TEMPORAL_LENGTH)
    CE = nn.CrossEntropyLoss(reduction='none')

    acc, banks, _, _ = eval_and_label_dataset(0, moco_model, trainloader_ta, config)
    
    moco_model.train()
    for epoch in range(config.surrogate_epochs):
        loss = train_step(epoch, base_model, moco_model, optimizer, trainloader_ta, banks, config, CE)
        
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
        print('[SFDA-PLUE] | epoch %03d | train_loss: %.3f | src_val_acc1: %.1f, tgt_val_acc1: ' %
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
