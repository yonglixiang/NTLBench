"""
Builds upon: https://github.com/tntek/source-free-domain-adaptation, https://github.com/tntek/source-free-domain-adaptation
Corresponding paper: https://arxiv.org/abs/2311.16510v3
"""
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb
import os
import torchvision.transforms as transforms
import torch.nn as nn
import torch.backends.cudnn as cudnn
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
from copy import deepcopy
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import numpy as np
import cv2

import utils
from utils.sfda_utils import SFDA_Trainset, op_copy, set_lr, lr_scheduler, MODEL_FEATURE_DIM, get_classnames
from utils.difo_utils import clip, prompt_tuning, IID_losses

def image_test(resize_size=256, crop_size=224, alexnet=False):
  if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  return  transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])

def image_test_50(resize_size=256, crop_size=224, alexnet=False):
    normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                std=[0.26862954, 0.26130258, 0.27577711])
                            
    return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(crop_size, interpolation=Image.BICUBIC),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            normalize
        ])

class DIFO_MIX_DATASET(Dataset):
    def __init__(self, confi_imag, confi_label, confi_dis, transform, mode='test'):
        self.transform = transform
        self.mode = mode
        self.image_list = confi_imag
        self.label_list = confi_label
        self.shot_predict_list = confi_dis

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img = self.image_list[idx]
        label = self.label_list[idx]
        
        # img preprocess: path(optional) -> img -> tensor -> numpy -> transform
        if isinstance(img, str):
            img = cv2.imread(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            img = np.array(img)
        if self.transform:
            img = self.transform(img)

        # label img preprocess: int(optional) -> one-hot
        if isinstance(label, (int, np.int32, np.int64)):
            label = np.eye(345)[label]  # set to 345 for domain_net dataset
            label = np.array(label)
            label = torch.LongTensor(label)
        
        # mix label
        pesu_label = self.shot_predict_list[idx]
            
        return img, label, pesu_label, idx

def clip_pre_text(cfg):
    """
    Prepare the text features for CLIP. Use the classnames as part of the prompt.
    """
    # get the classnames
    classnames = get_classnames(cfg)
    cfg.classname = classnames
    
    # add the prompt prefix, e.g., "A photo of a"
    prompt_prefix = cfg.DIFO_CTX_INIT.replace("_"," ")
    prompts = [prompt_prefix + " " + name + "." for name in classnames]
    
    # tokenize the prompts
    tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).cuda()
    
    return tokenized_prompts

def clip_text(model,text_features,inputs_test):
    """
    Compute the CLIP cls output for the text features and the image features.
    """
    with torch.no_grad():
        image_features = model.encode_image(inputs_test)
    logit_scale = model.logit_scale.data
    logit_scale = logit_scale.exp().cpu()
    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    logits = logit_scale * image_features @ text_features.t()
    return logits

def obtain_label(loader, model,text_inputs,text_features,clip_model):
    """
    Compute the:
    1. the logits of the model that using the clip_model as encoder
    2. the logits of the model that mixing the logits of the model and the clip_model
    3. the accuracy of the model that using the clip_model as encoder
    """
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            # get the inputs and labels
            # resize the inputs to align with the clip_model
            data = next(iter_test)
            inputs = data[0]
            inputs = inputs.cuda()
            labels = data[1]
            clip_inputs = F.interpolate(inputs, size=224, mode='bilinear') if inputs.size(-1) != 224 else inputs
            
            # get the outputs of the model and the clip_model
            outputs = model(inputs)
            if (text_features!=None):
                clip_score = clip_text(clip_model,text_features,clip_inputs)
            else :
                clip_score,_ = clip_model(clip_inputs, text_inputs)
            clip_score = clip_score.cpu()
            if start_test:
                all_output = outputs.float().cpu()
                all_clip_score = clip_score.float().cpu()
                all_label = labels.float().cpu()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
                all_clip_score = torch.cat((all_clip_score, clip_score.float()), 0)
                
    clip_all_output = nn.Softmax(dim=1)(all_clip_score).cpu()
    _, predict_clip = torch.max(clip_all_output, 1)
    _, all_label = torch.max(all_label, 1)
    accuracy_clip = torch.sum(torch.squeeze(predict_clip).float() == all_label).item() / float(all_label.size()[0])

    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    all_mix_output = (all_output+clip_all_output)/2

    confi_dis = all_mix_output.detach()
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    log_str = 'Accuracy = {:.2f}% -> CLIP_Accuracy  = {:.2f}%'.format(accuracy * 100, accuracy_clip * 100)
    # logging.info(log_str)
    # print(log_str)
    
    return confi_dis, clip_all_output

def DIFO(config, dataloader_train_srgt, dataloader_val, dataloader_test, 
         model, datasets_name):
    # shot train dataloader
    datafile_ta = SFDA_Trainset(dataloader_train_srgt[1], transform=None, mode='train')
    datafile_te = SFDA_Trainset(dataloader_train_srgt[1], transform=None, mode='train')
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
    evaluators_val = [utils.evaluators.classification_evaluator(
        v, topk) for v in dataloader_val]
    evaluators_test = [utils.evaluators.classification_evaluator(
        v, topk) for v in dataloader_test]
    
    # choose the model with the maximum Acc sum of source and target domain
    bestlogger = utils.evaluators.attack_ntl_logger_bestsum()
    
    # prepare the clip model
    clip_model, preprocess,_ = clip.load(config.DIFO_ARCH)
    clip_model.float()
    text_inputs = clip_pre_text(config)
    
    # optimize
    param_group = set_lr(model, config, only_f=True)
    optimizer = torch.optim.SGD(param_group)
    optimizer = op_copy(optimizer)
    
    # set up the training parameters
    num_sample = len(trainloader_ta.dataset)
    fea_bank = torch.randn(num_sample, MODEL_FEATURE_DIM[config.teacher_network])
    score_bank = torch.randn(num_sample, config.num_classes).cuda()

    # test the model before training
    model.eval()
    with torch.no_grad():
        iter_test = iter(trainloader_ta)
        for i in range(len(trainloader_ta)):
            data = next(iter_test)
            inputs = data[0]
            indx = data[-1]
            inputs = inputs.cuda()
            outputs = model(inputs)
            outputs = nn.Softmax(-1)(outputs)
            score_bank[indx] = outputs.detach().clone() 
    
    # Init the training parameters
    max_iter = config.surrogate_epochs * len(trainloader_ta)
    interval_iter = max_iter // config.surrogate_interval
    if interval_iter == 0:
        interval_iter = 1
    iter_num = 0
    text_features = None
    
    # start training
    epoch = 0
    model.train()
    while iter_num < max_iter:
        # Get the inputs
        try:
            inputs_test, _, tar_idx = next(iter_test)
        except:
            iter_test = iter(trainloader_ta)
            inputs_test, _, tar_idx = next(iter_test)
            epoch += 1

        if inputs_test.size(0) == 1:
            continue
        
        if iter_num % interval_iter == 0 and config.DIFO_CLS_PAR > 0:
            model.eval()
            confi_dis, clip_all_output = obtain_label(trainloader_te, model,text_inputs,text_features,clip_model)
            clip_all_output = clip_all_output.cuda()
            
            val_dataset = DIFO_MIX_DATASET(dataloader_train_srgt[1].dataset.list_img, dataloader_train_srgt[1].dataset.list_label, confi_dis, image_test(), mode='test')
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, drop_last=False)
            text_features = prompt_tuning.prompt_main(config, val_loader)
            
            config.load = 'prompt_model.pt'
            model.train()
        
        # Epoch update
        iter_num += 1
        lr_scheduler(config, optimizer, iter_num=iter_num, max_iter=max_iter)
        
        # Forward, get the output
        inputs_test = inputs_test.cuda()
        outputs_test = model(inputs_test)
        softmax_out = nn.Softmax(-1)(outputs_test)
        
        ln_sam = softmax_out.shape[0]
        data = np.random.exponential(scale=0.1, size=ln_sam)
        data = np.expand_dims(data, axis=1)
        data = torch.from_numpy(data)
        
        K = softmax_out.size(1)
        _, predict = torch.max(score_bank[tar_idx], 1)
        _, clip_predict = torch.max(clip_all_output[tar_idx], 1)
        predict_one = np.eye(K)[predict.cpu()]
        clip_one = np.eye(K)[clip_predict.cpu()]

        data = data.numpy()
        predict_mix = data*predict_one + (1-data)*clip_one
        predict_mix = torch.from_numpy(predict_mix).cuda()
        
        if config.DIFO_CLS_PAR > 0:
            targets = predict_mix
            loss_soft = (- targets * outputs_test).sum(dim=1)
            classifier_loss = loss_soft.mean()
            classifier_loss *= config.DIFO_CLS_PAR
        else:
            classifier_loss = torch.tensor(0.0).cuda()
        
        iic_loss = IID_losses.IID_loss(softmax_out, clip_all_output[tar_idx])
        classifier_loss = classifier_loss + config.DIFO_IIC_PAR * iic_loss

        msoftmax = softmax_out.mean(dim=0)

        gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + config.DIFO_EPSILON))
        classifier_loss = classifier_loss - config.DIFO_GENT_PAR * gentropy_loss
        with torch.no_grad():
            score_bank[tar_idx] = softmax_out.detach().clone()

        optimizer.zero_grad()
        classifier_loss.backward()
        optimizer.step()
        
        if iter_num % interval_iter == 0 or iter_num == max_iter:
            model.eval()
            
            # evaluate the model
            if topk == (1, 5):
                acc1s, _ = utils.evaluators.eval_func(config, evaluators_val, model)
            elif topk == (1, ):
                acc = utils.evaluators.eval_func(config, evaluators_val, model)
                acc1s = acc[0]
            tgt_mean = torch.mean(torch.tensor(acc1s[1:])).item()
            if epoch == 0 or bestlogger.log(acc1s[0], tgt_mean):
                if topk == (1, 5):
                    test_acc1s, _ = utils.evaluators.eval_func(config, evaluators_test, model)
                elif topk == (1, ):
                    test_acc = utils.evaluators.eval_func(config, evaluators_test, model)
                    test_acc1s = test_acc[0]

            wandb_log_dict = {
                'iter_ft': iter_num,
                'epoch_ft': epoch,
                'loss_ft': classifier_loss.item(),
                'Acc_ft_src': acc1s[0]}

            # print validation
            print('[SFDA-DIFO] | epoch %03d | iter %03d | train_loss: %.3f | src_val_acc1: %.1f, tgt_val_acc1: ' %
                (epoch, iter_num, classifier_loss.item(), acc1s[0]), end='')
            for dname, acc_tgt in zip(datasets_name[1:], acc1s[1:]):
                print(f'{dname}: {acc_tgt:.2f} ', end='')
                wandb_log_dict[f'Acc_ft_tgt_{dname}'] = acc_tgt
            print('')
            wandb.log(wandb_log_dict)
            
            model.train()
    
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
