"""
Builds upon: https://github.com/LyWang12/CUPI-Domain
Corresponding paper: https://arxiv.org/pdf/2408.13161
"""

import torch.nn as nn
import torch
import torch.nn.functional as F
from termcolor import cprint
import utils.evaluators
import wandb
from torch.optim import lr_scheduler
import gc
import copy
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
# import models.ntl_vggnet
import time
import cv2
import os


class CUPI_Trainset(data.Dataset):
    def __init__(self, dataloaders_train):

        self.transform = dataloaders_train[0].dataset.transform 
        self.data_size = len(dataloaders_train[0].dataset)

        src_img = dataloaders_train[0].dataset.list_img
        src_label = dataloaders_train[0].dataset.list_label
        tgt_img = dataloaders_train[1].dataset.list_img
        tgt_label = dataloaders_train[1].dataset.list_label

        if len(src_img) != len(tgt_img):  # align the length of source and target, use the smaller one
            self.data_size = min(len(src_img), len(tgt_img))       
        
        ind = np.arange(self.data_size)
        ind = np.random.permutation(ind)
        
        self.list_img1 = src_img[:self.data_size]
        self.list_label1 = src_label[:self.data_size]

        self.list_img2 = tgt_img[:self.data_size]
        self.list_label2 = tgt_label[:self.data_size]

        self.list_img3 = tgt_img[ind]
        self.list_label3 = tgt_label[ind]
        
        # self.list_img1 = np.asarray(self.list_img1)
        # self.list_img1 = self.list_img1[ind]
        # self.list_img2 = np.asarray(self.list_img2)
        # self.list_img2 = self.list_img2[ind]
        # self.list_img3 = np.asarray(self.list_img3)
        # self.list_img3 = self.list_img3[ind]

        # self.list_label1 = np.asarray(self.list_label1)
        # self.list_label1 = self.list_label1[ind]
        # self.list_label2 = np.asarray(self.list_label2)
        # self.list_label2 = self.list_label2[ind]
        # self.list_label3 = np.asarray(self.list_label3)
        # self.list_label3 = self.list_label3[ind]

    def __getitem__(self, item):
        img1 = self.list_img1[item]
        img2 = self.list_img2[item]
        img3 = self.list_img3[item]
        label1 = self.list_label1[item]
        label2 = self.list_label2[item]
        label3 = self.list_label3[item]
        
        if isinstance(img1, str):
            img1 = cv2.imread(img1)
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img1 = cv2.resize(img1, (224, 224))
            img1 = np.array(img1)
            img2 = cv2.imread(img2)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            img2 = cv2.resize(img2, (224, 224))
            img2 = np.array(img2)
            img3 = cv2.imread(img3)
            img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
            img3 = cv2.resize(img3, (224, 224))
            img3 = np.array(img3)
        
        if isinstance(label1, (int, np.int32, np.int64)):
            label1 = np.eye(345)[label1]  # set to 345 for domain_net dataset
            label1 = np.array(label1)
            label2 = np.eye(345)[label2]
            label2 = np.array(label2)
            label3 = np.eye(345)[label3]
            label3 = np.array(label3)
        
        return self.transform(img1), torch.LongTensor(label1), self.transform(img2), torch.LongTensor(label2), self.transform(img3), torch.LongTensor(label3), item

    def __len__(self):
        return self.data_size


def calc_ins_mean_std(x, eps=1e-5):
    """extract feature map statistics"""
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = x.size()
    if len(size) == 4:  # for vgg, resnet
        N, C = size[:2]
        var = x.contiguous().view(N, C, -1).var(dim=2) + eps
        std = var.sqrt().view(N, C, 1, 1)
        mean = x.contiguous().view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    elif len(size) == 3:  # for vit
        N, C = size[0], size[2]
        var = x.contiguous().view(N, -1, C).var(dim=1)  + eps
        std = var.sqrt().view(N, 1, C)
        mean = x.contiguous().view(N, -1, C).mean(dim=1).view(N, 1, C)
         
    return mean, std


class CUPI(nn.Module):
    def __init__(self):
        super(CUPI, self).__init__()
    def forward(self, x):
        if self.training:
            if x.dim() == 4:  # for vgg, resnet
                batch_size, C = x.size()[0] // 3, x.size()[1]
                style_mean, style_std = calc_ins_mean_std(x[:batch_size])
                c_mean, c_std = calc_ins_mean_std(x[batch_size:2 * batch_size])
                conv_mean = nn.Conv2d(C, C, 1, bias=False).cuda()
                conv_std = nn.Conv2d(C, C, 1, bias=False).cuda()
                mean = torch.sigmoid(conv_mean(style_mean))
                std = torch.sigmoid(conv_std(style_std))
                x_a = (x[batch_size:2 * batch_size] - c_mean) / (c_std + 1e-6) * std + mean
                x = torch.cat((x[:batch_size], x_a, x[2 * batch_size:]), 0)
            elif x.dim() == 3:  # for vit
                batch_size, C = x.size()[0] // 3, x.size()[2]
                style_mean, style_std = calc_ins_mean_std(x[:batch_size])
                c_mean, c_std = calc_ins_mean_std(x[batch_size:2 * batch_size])
                conv_mean = nn.Conv1d(1, 1, 1, bias=False).cuda()
                conv_std = nn.Conv1d(1, 1, 1, bias=False).cuda()
                mean = torch.sigmoid(conv_mean(style_mean))
                std = torch.sigmoid(conv_std(style_std))
                x_a = (x[batch_size:2 * batch_size] - c_mean) / (c_std + 1e-6) * std + mean
                x = torch.cat((x[:batch_size], x_a, x[2 * batch_size:]), 0)
        return x
    

def Memory(config, dataloader, model):
    device = config.device
    TRAIN_SIZE = len(dataloader.dataset)
    if 'vgg' in config.teacher_network:
        memory_dict = {
            "memory_CUPI_features": torch.zeros(TRAIN_SIZE, 256).to(device),
            "memory_CUPI_labels": torch.zeros(TRAIN_SIZE).long().to(device),
            "memory_source_mean1": torch.zeros(TRAIN_SIZE, 64).to(device),
            "memory_source_std1": torch.zeros(TRAIN_SIZE, 64).to(device),
            "memory_source_mean2": torch.zeros(TRAIN_SIZE, 128).to(device),
            "memory_source_std2": torch.zeros(TRAIN_SIZE, 128).to(device),
            "memory_source_mean3": torch.zeros(TRAIN_SIZE, 256).to(device),
            "memory_source_std3": torch.zeros(TRAIN_SIZE, 256).to(device),
            "memory_source_mean4": torch.zeros(TRAIN_SIZE, 512).to(device),
            "memory_source_std4": torch.zeros(TRAIN_SIZE, 512).to(device),
            "memory_source_mean5": torch.zeros(TRAIN_SIZE, 512).to(device),
            "memory_source_std5": torch.zeros(TRAIN_SIZE, 512).to(device),
            "memory_source_labels": torch.zeros(TRAIN_SIZE).long().to(device)
        }
    elif 'vit' in config.teacher_network:
        memory_dict = {
            "memory_CUPI_features": torch.zeros(TRAIN_SIZE, 768).to(device),
            "memory_CUPI_labels": torch.zeros(TRAIN_SIZE).long().to(device),
            "memory_source_mean1": torch.zeros(TRAIN_SIZE, 768).to(device),
            "memory_source_std1": torch.zeros(TRAIN_SIZE, 768).to(device),
            "memory_source_mean2": torch.zeros(TRAIN_SIZE, 768).to(device),
            "memory_source_std2": torch.zeros(TRAIN_SIZE, 768).to(device),
            "memory_source_mean3": torch.zeros(TRAIN_SIZE, 768).to(device),
            "memory_source_std3": torch.zeros(TRAIN_SIZE, 768).to(device),
            "memory_source_mean4": torch.zeros(TRAIN_SIZE, 768).to(device),
            "memory_source_std4": torch.zeros(TRAIN_SIZE, 768).to(device),
            "memory_source_mean5": torch.zeros(TRAIN_SIZE, 768).to(device),
            "memory_source_std5": torch.zeros(TRAIN_SIZE, 768).to(device),
            "memory_source_labels": torch.zeros(TRAIN_SIZE).long().to(device)
        }
    with torch.no_grad():
        for i, (imgs, labels, imgc, labelc, _, _, idx) in enumerate(dataloader): # source, target, target
            imgs, labels, imgc, labelc = imgs.to(device), labels.to(device), imgc.to(device), labelc.to(device)
            imgs, labels, imgc, labelc =  imgs.float(), labels.float(), imgc.float(), labelc.float()
            # fs1, fc1, fs2, fc2, fs3, fc3, fs4, fc4, fs5, fc5, ps, pc, ys, yc = model(x=imgs, y=imgc, action='memory')
            if 'vgg' in config.teacher_network:
                fs1, fc1, fs2, fc2, fs3, fc3, fs4, fc4, fs5, fc5, ps, pc, ys, yc = forward_CUPI(
                    model, x=imgs, y=imgc, action='memory')
            elif 'vit' in config.teacher_network:
                fs1, fc1, fs2, fc2, fs3, fc3, fs4, fc4, fs5, fc5, ps, pc, ys, yc = forward_CUPI_vit(
                    model, x=imgs, y=imgc, action='memory')

            # source style
            mean1, std1 = calc_ins_mean_std(fs1)
            memory_dict["memory_source_mean1"][idx] = mean1.squeeze().detach()
            memory_dict["memory_source_std1"][idx] = std1.squeeze().detach()
            mean2, std2 = calc_ins_mean_std(fs2)
            memory_dict["memory_source_mean2"][idx] = mean2.squeeze().detach()
            memory_dict["memory_source_std2"][idx] = std2.squeeze().detach()
            mean3, std3 = calc_ins_mean_std(fs3)
            memory_dict["memory_source_mean3"][idx] = mean3.squeeze().detach()
            memory_dict["memory_source_std3"][idx] = std3.squeeze().detach()
            mean4, std4 = calc_ins_mean_std(fs4)
            memory_dict["memory_source_mean4"][idx] = mean4.squeeze().detach()
            memory_dict["memory_source_std4"][idx] = std4.squeeze().detach()
            mean5, std5 = calc_ins_mean_std(fs5)
            memory_dict["memory_source_mean5"][idx] = mean5.squeeze().detach()
            memory_dict["memory_source_std5"][idx] = std5.squeeze().detach()
            # CUPI prediction
            memory_dict["memory_CUPI_features"][idx] = pc
            memory_dict["memory_CUPI_labels"][idx] = torch.LongTensor([one_label.tolist().index(1) for one_label in labelc]).to(device)
            memory_dict["memory_source_labels"][idx] = torch.LongTensor([one_label.tolist().index(1) for one_label in labels]).to(device)

    print('Memory initial!')
    return memory_dict


class CalculateMean_(nn.Module):
    def __init__(self, config):
        super(CalculateMean_, self).__init__()
        self.config = config
        self.NUM_CLASSES = config.num_classes
        self.device = config.device

    def __call__(self, features, labels):
        N = features.size(0)
        C = self.NUM_CLASSES
        A = features.size(1)

        avg_CxA = torch.zeros(C, A).to(self.device)
        NxCxFeatures = features.view(N, 1, A).expand(N, C, A)

        onehot = torch.zeros(N, C).to(self.device)
        onehot.scatter_(1, labels.view(-1, 1), 1)
        NxCxA_onehot = onehot.view(N, C, 1).expand(N, C, A)

        Amount_CxA = NxCxA_onehot.sum(0)
        Amount_CxA[Amount_CxA == 0] = 1.0

        del onehot
        gc.collect()
        for c in range(self.NUM_CLASSES):
            c_temp = NxCxFeatures[:, c, :].mul(NxCxA_onehot[:, c, :])
            c_temp = torch.sum(c_temp, dim=0)
            avg_CxA[c] = c_temp / Amount_CxA[c]
        return avg_CxA.detach()
    

def Discrimination_loss(CalculateMean, memory_dict, ps, pc, pt, index, label1, label3):
    memory_dict["memory_CUPI_features"][index] = pc.detach()
    mean_CUPI = CalculateMean(memory_dict["memory_CUPI_features"], memory_dict["memory_CUPI_labels"])  # 10,256
    loss_d_sc = F.mse_loss(mean_CUPI[torch.LongTensor([one_label.tolist().index(1) for one_label in label1]).cuda()], ps)
    loss_d_tc = F.mse_loss(mean_CUPI[torch.LongTensor([one_label.tolist().index(1) for one_label in label3]).cuda()], pt)
    return loss_d_sc, loss_d_tc, memory_dict


def Style_loss(CalculateMean, memory_dict, fs1, ft1, fs2, ft2, fs3, ft3, fs4, ft4, fs5, ft5, index, label3):
    fs_mean1, fs_std1 = calc_ins_mean_std(fs1)
    memory_dict["memory_source_mean1"][index] = fs_mean1.squeeze().detach()
    memory_dict["memory_source_std1"][index] = fs_std1.squeeze().detach()
    mean_source1 = CalculateMean(memory_dict["memory_source_mean1"], memory_dict["memory_source_labels"])
    std_source1 = CalculateMean(memory_dict["memory_source_std1"], memory_dict["memory_source_labels"])
    ft_mean1, ft_std1 = calc_ins_mean_std(ft1)
    loss_s_mean_t_1 = F.mse_loss(mean_source1[torch.LongTensor([one_label.tolist().index(1) for one_label in label3]).cuda()], ft_mean1.squeeze())
    loss_s_std_t_1 = F.mse_loss(std_source1[torch.LongTensor([one_label.tolist().index(1) for one_label in label3]).cuda()], ft_std1.squeeze())

    fs_mean2, fs_std2 = calc_ins_mean_std(fs2)
    memory_dict["memory_source_mean2"][index] = fs_mean2.squeeze().detach()
    memory_dict["memory_source_std2"][index] = fs_std2.squeeze().detach()
    mean_source2 = CalculateMean(memory_dict["memory_source_mean2"], memory_dict["memory_source_labels"])
    std_source2 = CalculateMean(memory_dict["memory_source_std2"], memory_dict["memory_source_labels"])
    ft_mean2, ft_std2 = calc_ins_mean_std(ft2)
    loss_s_mean_t_2 = F.mse_loss(mean_source2[torch.LongTensor([one_label.tolist().index(1) for one_label in label3]).cuda()], ft_mean2.squeeze())
    loss_s_std_t_2 = F.mse_loss(std_source2[torch.LongTensor([one_label.tolist().index(1) for one_label in label3]).cuda()], ft_std2.squeeze())

    fs_mean3, fs_std3 = calc_ins_mean_std(fs3)
    memory_dict["memory_source_mean3"][index] = fs_mean3.squeeze().detach()
    memory_dict["memory_source_std3"][index] = fs_std3.squeeze().detach()
    mean_source3 = CalculateMean(memory_dict["memory_source_mean3"], memory_dict["memory_source_labels"])
    std_source3 = CalculateMean(memory_dict["memory_source_std3"], memory_dict["memory_source_labels"])
    ft_mean3, ft_std3 = calc_ins_mean_std(ft3)
    loss_s_mean_t_3 = F.mse_loss(mean_source3[torch.LongTensor([one_label.tolist().index(1) for one_label in label3]).cuda()], ft_mean3.squeeze())
    loss_s_std_t_3 = F.mse_loss(std_source3[torch.LongTensor([one_label.tolist().index(1) for one_label in label3]).cuda()], ft_std3.squeeze())

    fs_mean4, fs_std4 = calc_ins_mean_std(fs4)
    memory_dict["memory_source_mean4"][index] = fs_mean4.squeeze().detach()
    memory_dict["memory_source_std4"][index] = fs_std4.squeeze().detach()
    mean_source4 = CalculateMean(memory_dict["memory_source_mean4"], memory_dict["memory_source_labels"])
    std_source4 = CalculateMean(memory_dict["memory_source_std4"], memory_dict["memory_source_labels"])
    ft_mean4, ft_std4 = calc_ins_mean_std(ft4)
    loss_s_mean_t_4 = F.mse_loss(mean_source4[torch.LongTensor([one_label.tolist().index(1) for one_label in label3]).cuda()], ft_mean4.squeeze())
    loss_s_std_t_4 = F.mse_loss(std_source4[torch.LongTensor([one_label.tolist().index(1) for one_label in label3]).cuda()], ft_std4.squeeze())

    fs_mean5, fs_std5 = calc_ins_mean_std(fs5)
    memory_dict["memory_source_mean5"][index] = fs_mean5.squeeze().detach()
    memory_dict["memory_source_std5"][index] = fs_std5.squeeze().detach()
    mean_source5 = CalculateMean(memory_dict["memory_source_mean5"], memory_dict["memory_source_labels"])
    std_source5 = CalculateMean(memory_dict["memory_source_std5"], memory_dict["memory_source_labels"])
    ft_mean5, ft_std5 = calc_ins_mean_std(ft5)
    loss_s_mean_t_5 = F.mse_loss(mean_source5[torch.LongTensor([one_label.tolist().index(1) for one_label in label3]).cuda()], ft_mean5.squeeze())
    loss_s_std_t_5 = F.mse_loss(std_source5[torch.LongTensor([one_label.tolist().index(1) for one_label in label3]).cuda()], ft_std5.squeeze())

    loss_s_mean_t = loss_s_mean_t_1 + loss_s_mean_t_2 + loss_s_mean_t_3 + loss_s_mean_t_4 + loss_s_mean_t_5
    loss_s_std_t = loss_s_std_t_1 + loss_s_std_t_2 + loss_s_std_t_3 + loss_s_std_t_4 + loss_s_std_t_5

    return loss_s_mean_t, loss_s_std_t, memory_dict
    

def forward_CUPI(model, x, y=None, z=None, action=None):

    # to maintain the arch of NTL VGG, we use hook to get the bottleneck features
    features = {}
    def hook_fn(module, input, output):
        # return output
        features['bottleneck'] = output
    
    # handle = model.classifier1[0].register_forward_hook(hook_fn)
    handle = model.classifier1[2].register_forward_hook(hook_fn)

    if action == 'val':
        x = model.features(x)
        x = x.view(x.size(0), -1)
        # x = model.bottleneck(x)
        x = model.classifier1(x)
        return x

    elif action == 'memory':
        input = torch.cat((x, y), 0)
        input1 = model.chan_ex(model.features[:3](input))
        input2 = model.chan_ex(model.features[3:6](input1))
        input3 = model.chan_ex(model.features[6:11](input2))
        input4 = model.chan_ex(model.features[11:16](input3))
        input5 = model.chan_ex(model.features[16:](input4))
        fx1, fy1 = input1.chunk(2, dim=0)
        fx2, fy2 = input2.chunk(2, dim=0)
        fx3, fy3 = input3.chunk(2, dim=0)
        fx4, fy4 = input4.chunk(2, dim=0)
        fx5, fy5 = input5.chunk(2, dim=0)

        input5 = input5.view(input5.size(0), -1)
        # input5 = model.bottleneck(input5)
        px, py = input5.chunk(2, dim=0)

        x = model.classifier1(px)
        px = features['bottleneck']
        y = model.classifier1(py)
        py = features['bottleneck']

        return fx1, fy1, fx2, fy2, fx3, fy3, fx4, fy4, fx5, fy5, px, py, x, y

    elif action == 'train':
        input = torch.cat((x, y, z), 0)
        input1 = model.chan_ex(model.features[:3](input))
        input2 = model.chan_ex(model.features[3:6](input1))
        input3 = model.chan_ex(model.features[6:11](input2))
        input4 = model.chan_ex(model.features[11:16](input3))
        input5 = model.chan_ex(model.features[16:](input4))
        fx1, fy1, fz1 = input1.chunk(3, dim=0)
        fx2, fy2, fz2 = input2.chunk(3, dim=0)
        fx3, fy3, fz3 = input3.chunk(3, dim=0)
        fx4, fy4, fz4 = input4.chunk(3, dim=0)
        fx5, fy5, fz5 = input5.chunk(3, dim=0)

        input5 = input5.view(input5.size(0), -1)
        # input5 = model.bottleneck(input5)
        px, py, pz = input5.chunk(3, dim=0)

        x = model.classifier1(px)
        px = features['bottleneck']
        y = model.classifier1(py)
        py = features['bottleneck']
        z = model.classifier1(pz)
        pz = features['bottleneck']

        return fx1, fy1, fz1, fx2, fy2, fz2, fx3, fy3, fz3, fx4, fy4, fz4, fx5, fy5, fz5, px, py, pz, x, y, z

def forward_CUPI_vit(model, x, y=None, z=None, action=None):
    # Use hook to get the bottleneck features
    features = {}
    def hook_fn(module, input, output):
        normalized_output = model.norm(output)
        pooled_output = model.pool(normalized_output)
        features['bottleneck'] = pooled_output
    
    handle = model.blocks[-1].register_forward_hook(hook_fn)  # get the last block features
    
    if action == 'val':
        return model(x)
    
    elif action == 'memory':
        # Concatenate x and y
        input = torch.cat((x, y), 0)
        
        # Patch embedding and positional encoding
        input = model.patch_embed(input)
        input = model._pos_embed(input)
        input = model.patch_drop(input)
        input = model.norm_pre(input)
        
        # Transformer Blocks
        split = range(0, len(model.blocks), len(model.blocks) // 5)
        for i, block in enumerate(model.blocks):
            input = block(input)
            if i in split:
                input = model.chan_ex(input)
                if i == split[0]:
                    input1 = input
                elif i == split[1]:
                    input2 = input
                elif i == split[2]:
                    input3 = input
                elif i == split[3]:
                    input4 = input
                elif i == split[4]:
                    input5 = input
        fx1, fy1 = input1.chunk(2, dim=0)
        fx2, fy2 = input2.chunk(2, dim=0)
        fx3, fy3 = input3.chunk(2, dim=0)
        fx4, fy4 = input4.chunk(2, dim=0)
        fx5, fy5 = input5.chunk(2, dim=0)
        px, py = features['bottleneck'].chunk(2, dim=0)
        
        # Final classification head
        input = model.norm(input)
        input = model.forward_head(input)
        out_x, out_y = input.chunk(2, dim=0)
        
        return fx1, fy1, fx2, fy2, fx3, fy3, fx4, fy4, fx5, fy5, px, py, out_x, out_y
    
    elif action == 'train':
        # Concatenate x, y, and z
        input = torch.cat((x, y, z), 0)
        
        # Patch embedding and positional encoding
        input = model.patch_embed(input)
        input = model._pos_embed(input)
        input = model.patch_drop(input)
        input = model.norm_pre(input)
        
        # Transformer Blocks
        split = range(0, len(model.blocks), len(model.blocks) // 5)
        for i, block in enumerate(model.blocks):
            input = block(input)
            if i in split:
                input = model.chan_ex(input)
                if i == split[0]:
                    input1 = input
                elif i == split[1]:
                    input2 = input
                elif i == split[2]:
                    input3 = input
                elif i == split[3]:
                    input4 = input
                elif i == split[4]:
                    input5 = input
        fx1, fy1, fz1 = input1.chunk(3, dim=0)
        fx2, fy2, fz2 = input2.chunk(3, dim=0)
        fx3, fy3, fz3 = input3.chunk(3, dim=0)
        fx4, fy4, fz4 = input4.chunk(3, dim=0)
        fx5, fy5, fz5 = input5.chunk(3, dim=0)
        px, py, pz = features['bottleneck'].chunk(3, dim=0)
        
        # Final classification head
        input = model.norm(input)
        input = model.forward_head(input)
        out_x, out_y, out_z = input.chunk(3, dim=0)
        
        return fx1, fy1, fz1, fx2, fy2, fz2, fx3, fy3, fz3, fx4, fy4, fz4, fx5, fy5, fz5, px, py, pz, out_x, out_y, out_z


def train_tCUPI(config, dataloaders, valloaders, testloaders, model, datasets_name):
    
    cprint('Modify Net to CUPI Net', 'yellow')
    # model.classifier1 = nn.Sequential(
    #         nn.Linear(2048, 256),
    #         nn.ReLU(True),
    #         nn.Dropout(),
    #         nn.Linear(256, 256),
    #         nn.ReLU(True),
    #         nn.Dropout(),
    #         nn.Linear(256, config.num_classes),
    #     )
    # model.classifier1 = nn.Sequential(
    #         nn.Linear(2048, 256),
    #         # nn.ReLU(True),
    #         # nn.Dropout(),
    #         nn.Linear(256, 256),
    #         nn.ReLU(True),
    #         nn.Dropout(),
    #         nn.Linear(256, config.num_classes),
    #     )
    
    model.chan_ex = CUPI()
    model.to(config.device)
    CalculateMean = CalculateMean_(config).to(config.device)
    # modify the dataloader to CUPT

    cupi_trainfile = CUPI_Trainset(dataloaders)
    cupi_trainloaders = DataLoader(cupi_trainfile, batch_size=config.batch_size, 
                             shuffle=True, num_workers=config.num_workers, drop_last=True)

    model.eval()
    memory_dict = Memory(config, dataloader=cupi_trainloaders, model=model)
    model.train()

    if 'cmt' in config.domain_src:
        topk = (1, )  # 2-classification, use top1 acc
    else:
        topk = (1, 5)  # 10-classification, use top1 and top5 acc
    evaluators = [utils.evaluators.classification_evaluator(
        v, topk) for v in valloaders]
    evaluators_test = [utils.evaluators.classification_evaluator(
        v, topk) for v in testloaders]

    if not hasattr(config, 'CUPI_lr'):
        # update CUPI_lr
        config.update({'CUPI_lr': 0.0001}, allow_val_change=True)
        print('use default cifar-stl learning rate: 0.0001')
    lr = config.CUPI_lr

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    lambda1 = lambda epoch:0.999**epoch
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    criterion_KL = torch.nn.KLDivLoss()

    if config.teacher_network in ['vgg11', 'vgg13', 'vgg19', 'vgg11bn', 'vgg13bn', 'vgg19bn']:
        forward_func = forward_CUPI
    elif config.teacher_network in ['vit_tiny', 'vit_base', 'vit_large', 'vit_huge']:
        forward_func = forward_CUPI_vit
    # elif config.teacher_network in ['resnet18cmi', 'resnet18cmi_wobn', 'resnet34cmi_wobn', 'resnet34cmi', 'resnet50cmi', 'wide_resnet50_2cmi']:
        # forward_func = forward_CUTI_resnetcmi
    else:
        raise NotImplementedError

    device = config.device
    pretrain_epochs = 20
    for epoch in range(pretrain_epochs):
        model.train()
        for i, (img1, label1, img2, label2, img3, label3, index) in enumerate(cupi_trainloaders):
            img1, label1, img2, label2, img3, label3 = img1.to(device), label1.to(device), img2.to(device), label2.to(device), img3.to(device), label3.to(device)
            img1, label1, img2, label2, img3, label3 = img1.float(), label1.float(), img2.float(), label2.float(), img3.float(), label3.float()
            
            fs1, fc1, ft1, fs2, fc2, ft2, fs3, fc3, ft3, fs4, fc4, ft4, fs5, fc5, ft5, ps, pc, pt, ys, yc, yt = forward_func(
                model, x=img1, y=img2, z=img3, action='train')

            alpha = config.CUPI_alpha
            kl = config.CUPI_kl * (epoch + 1 / pretrain_epochs) ** 0.9

            ys = F.log_softmax(ys, dim=1)
            loss1 = criterion_KL(ys, label1)

            yc = F.log_softmax(yc, dim=1)
            loss2 = criterion_KL(yc, label2)
            loss2 = loss2 * alpha
            if loss2 > 1:
                loss2 = torch.clamp(loss2, 0, 1)

            yt = F.log_softmax(yt, dim=1)
            loss3 = criterion_KL(yt, label3)
            loss3 = loss3 * alpha
            if loss3 > 1:
                loss3 = torch.clamp(loss3, 0, 1)

            # discrimination loss
            loss_d_sc, loss_d_tc, memory_dict = Discrimination_loss(
                CalculateMean, memory_dict, ps, pc, pt, index, label1, label3)
            loss_d_sc = loss_d_sc * kl
            if loss_d_sc > 1:
                loss_d_sc = torch.clamp(loss_d_sc, 0, 1)

            loss_d_tc = loss_d_tc * kl
            if loss_d_tc > 1:
                loss_d_tc = torch.clamp(loss_d_tc, 0, 1)

            # style loss
            loss_s_mean_t, loss_s_std_t, memory_dict = Style_loss(
                CalculateMean, memory_dict, fs1, ft1, fs2, ft2, fs3, ft3, fs4, ft4, fs5, ft5, index, label3)
            loss_s_mean_t = loss_s_mean_t * kl
            if loss_s_mean_t > 1:
                loss_s_mean_t = torch.clamp(loss_s_mean_t, 0, 1)

            loss_s_std_t = loss_s_std_t * kl
            if loss_s_std_t > 1:
                loss_s_std_t = torch.clamp(loss_s_std_t, 0, 1)

            loss = loss1 - loss2 - loss3 - loss_d_sc + loss_d_tc - loss_s_mean_t - loss_s_std_t
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        model.eval()
        acc1s, acc5s = [], []
        for evaluator in evaluators:
            eval_results = evaluator(model, device=config.device)
            if topk == (1, 5):
                (acc1, acc5), val_celoss_ = eval_results['Acc'], eval_results['Loss']
            elif topk == (1, ):
                acc, val_celoss_ = eval_results['Acc'], eval_results['Loss']
                acc1, acc5 = acc[0], 0.0
            acc1s.append(acc1)
            acc5s.append(acc5)

        wandb_log_dict = {
            'pt_epoch': epoch,
            'pt_loss': loss.item(),
            'pt_Acc_val_src': acc1s[0],
            # 'Acc_src': acc1,
        }

        print('[CUPI Pretrain] | epoch %03d | train_loss: %.3f | src_val_acc1: %.1f, tgt_val_acc1: ' %
              (epoch, loss.item(), acc1s[0]), end='')
            # (epoch, loss.item(), acc1), end='')
        for dname, acc_tgt in zip(datasets_name[1:], acc1s[1:]):
        # for dname, acc_tgt in zip(datasets_name[1:], [acc2]):
            print(f'{dname}: {acc_tgt:.2f} ', end='')
            wandb_log_dict[f'pt_Acc_val_tgt_{dname}'] = acc_tgt
        tgt_mean = torch.mean(torch.tensor(acc1s[1:])).item()
        wandb_log_dict[f'pt_Acc_val_tgt_mean'] = tgt_mean
        print(f'| tgt_mean: {tgt_mean:.2f}')
        wandb.log(wandb_log_dict)
    
    test_acc1s, _ = utils.evaluators.eval_func(config, evaluators_test, model)

    # wandb.run.summary['final_valbest_Acc_ft_src'] = bestlogger.result()['src']
    # wandb.run.summary['final_valbest_Acc_ft_tgtmean'] = bestlogger.result()['tgt']
    wandb.run.summary['pt_Acc_test_src'] = test_acc1s[0]
    for dname, acc_tgt in zip(datasets_name[1:], test_acc1s[1:]):
        wandb.run.summary[f'pt_Acc_test_tgt_{dname}'] = acc_tgt
    test_acc1s_tgt_mean = torch.mean(torch.tensor(test_acc1s[1:])).item()
    wandb.run.summary[f'pt_Acc_test_tgt_mean'] = test_acc1s_tgt_mean
    
    return test_acc1s[0], test_acc1s_tgt_mean

