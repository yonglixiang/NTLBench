import torch.nn as nn
import torch
import torch.nn.functional as F
import os
import utils
import wandb
import copy
import numpy as np
import torch.utils.data as data
from torch.utils.data import DataLoader
from scipy.spatial.distance import cdist
from copy import deepcopy
from PIL import Image
import torch
import torch.nn.parallel
import cv2
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torch.nn as nn
from .sfda_loss import Entropy
from sklearn.metrics import confusion_matrix


class SFDA_Trainset(data.Dataset):
    def __init__(self, dataloader_train, transform=None, mode='test'):
        if transform is None:
            self.transform = dataloader_train.dataset.transform
        else:
            self.transform = transform
        self.data_size = len(dataloader_train.dataset)
        self.mode = mode

        src_img = dataloader_train.dataset.list_img
        src_label = dataloader_train.dataset.list_label
        
        # ind = np.arange(self.data_size)
        # ind = np.random.permutation(ind)
        
        self.list_img1 = src_img
        self.list_label1 = src_label
    
    def __getitem__(self, item):
        # get the item
        img = self.list_img1[item]
        label = self.list_label1[item]
        
        # img preprocess: path(optional) -> img -> tensor -> numpy -> transform
        if isinstance(img, str):
            img = cv2.imread(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            img = np.array(img)
        img = self.transform(img)

        # label img preprocess: int(optional) -> one-hot
        if isinstance(label, (int, np.int32, np.int64)):
            label = np.eye(345)[label]  # set to 345 for domain_net dataset
            label = np.array(label)
        
        # unsqueez the label for the case of test
        if self.mode == 'test':
            return img, torch.LongTensor(label).unsqueeze(0), item
        else:
            return img, torch.LongTensor(label), item
       

    def __len__(self):
        return self.data_size


def set_lr(model, config, only_f=False):
    """
    Set the learning rate of the model according to the config.
    
    config.surrogate_LR_DECAY1: the decay factor for the feature extractor
    config.surrogate_LR_DECAY2: the decay factor for the classifier
    
    if the decay factor is 0, the corresponding part of the model will be frozen.
    """
    if only_f:
        surrogate_LR_DECAY2 = 0
    else:
        surrogate_LR_DECAY2 = config.surrogate_LR_DECAY2
    
    if 'vgg' in config.teacher_network:
        param_group = []
        for k, v in model.features.named_parameters():  # feature extractor
            if config.surrogate_LR_DECAY1 > 0:
                param_group += [{'params': v, 
                                    'lr': config.surrogate_lr * config.surrogate_LR_DECAY1}]
            else:
                v.requires_grad = False
            # param_group += [{'params': v}]
        for k, v in model.classifier1.named_parameters():  # classifier
            if surrogate_LR_DECAY2 > 0:
                param_group += [{'params': v, 
                                    'lr': config.surrogate_lr * surrogate_LR_DECAY2}]
            else:
                v.requires_grad = False
    elif 'wide_' in config.teacher_network:
        raise NotImplementedError
    elif 'resnet' in config.teacher_network:
        param_group = []
        for k, v in model.named_parameters():  # feature extractor
            if 'fc' in k: continue
            if config.surrogate_LR_DECAY1 > 0:
                param_group += [{'params': v, 
                                    'lr': config.surrogate_lr * config.surrogate_LR_DECAY1}]
            else:
                v.requires_grad = False
        
        for k, v in model.fc.named_parameters():  # classifier
            if surrogate_LR_DECAY2 > 0:
                param_group += [{'params': v, 
                                    'lr': config.surrogate_lr * surrogate_LR_DECAY2}]
            else:
                v.requires_grad = False
    elif 'vit' in config.teacher_network:
        param_group = []
        for k, v in model.blocks.named_parameters():
            if config.surrogate_LR_DECAY1 > 0:  # feature extractor
                param_group += [{'params': v, 
                                    'lr': config.surrogate_lr * config.surrogate_LR_DECAY1}]
            else:
                v.requires_grad = False
        
        for k, v in model.named_parameters():  # classifier
            if any(x in k for x in ['head', 'fc_norm', 'attn_pool']):
                if surrogate_LR_DECAY2 > 0:
                    param_group += [{'params': v, 
                                        'lr': config.surrogate_lr * surrogate_LR_DECAY2}]
                else:
                    v.requires_grad = False
    
    return param_group

def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def lr_scheduler(cfg, optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = cfg.surrogate_weight_decay
        param_group['momentum'] = cfg.surrogate_momentum
        param_group['nesterov'] = cfg.surrogate_NESTEROV
    return optimizer

MODEL_FEATURE_DIM = {
        'vgg13': 2048,
        'vgg19': 2048,
        'resnet18': 512,
        'resnet34': 512,
        'resnet50': 2048,
        'resnet101': 2048,
        'resnet152': 2048,
        'wide_resnet50_2': 2048,
        'wide_resnet101_2': 2048,
        'vit_tiny': 192,
        'vit_base': 768,
        'vit_large': 1024,
        'vit_huge': 1280
}

def cal_acc(loader, model, flag=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = model(inputs)
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()

    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal()/matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc
    else:
        return accuracy*100, mean_ent
    
def get_classnames(config):
    # the classnames file should be saved in the classnames folder
    dataset = {
        'mt': 'digits',
        'mm': 'digits',
        'us': 'digits',
        'sn': 'digits',
        'sd': 'digits',
        'rmt': 'digits',
        'cmt': 'digits-2',
        'cifar': 'cifar',
        'stl': 'stl',
        'visda': 'visda',
        'home': 'home_office',
        'domain_net': 'domain_net',
        'vlcs': 'vlcs',
        'pacs': 'pacs',
        'ti': 'ti'
    }
    for domain, filenames in dataset.items():
        if domain in config.domain_tgt:
            filename = filenames
            break
    if 'cmt' in config.domain_tgt:
        filename = 'digits-2'
    name_file = os.path.join('./classnames', filename + '.txt')
    classnames = []
    with open(name_file) as f:
        for line in f:
            line = line.strip()
            if line:
                classnames.extend([i for i in line.split(',')])
    f.close()
    
    assert len(classnames) == config.num_classes, f'The number of classnames is not correct, expect {config.num_classes}, got {len(classnames): classnames}'
    
    return classnames
