from copy import deepcopy
from PIL import Image
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torch.nn as nn
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from .imagnet_prompts import imagenet_classes

from .IID_losses import IID_loss
from .clip.custom_clip import get_coop


def image_test(resize_size=256, crop_size=224, alexnet=False):
  if not alexnet:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  return  transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])

def image_test_50(resize_size=256, crop_size=224, alexnet=False):
    normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                std=[0.26862954, 0.26130258, 0.27577711])
                            
    return transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(crop_size, interpolation=Image.BICUBIC),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            normalize
        ])

def test_time_tuning(model, inputs,pesu_label, optimizer, cfg):
    for j in range(cfg.DIFO_TTA_STEPS):
        with torch.cuda.amp.autocast():
            output,_ = model(inputs) 
            pesu_label = pesu_label.cuda()
            output = nn.Softmax(dim=1)(output)
            loss = IID_loss(output, pesu_label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return 

def prompt_main(cfg, val_loader):
    # This codebase has only been tested under the single GPU setting
    # assert int(cfg.GPU_ID) is not None
    assert 'cuda' in cfg.device
    text_features = main_worker(cfg, val_loader)
    text_features = text_features.detach()
    return text_features

def main_worker(cfg, val_loader):
    # if cfg.SETTING.DATASET in domain_datasets:
    #     cfg.domain_name = cfg.domain[cfg.SETTING.T]
    #     classnames = cfg.classname
    cfg.domain_name = cfg.domain_tgt
    classnames = cfg.classname
    imagenet_classnames = imagenet_classes

    # model = get_coop(cfg.DIFO_ARCH, cfg.SETTING.DATASET, int(cfg.GPU_ID), cfg.DIFO_N_CTX, cfg.DIFO_CTX_INIT)
    model = get_coop(cfg.DIFO_ARCH, imagenet_classnames, 0, cfg.DIFO_N_CTX, cfg.DIFO_CTX_INIT)
    model = model.cuda()

    # if cfg.DIFO_LOAD is not None:
    #     pretrained_ctx = torch.load(cfg.DIFO_LOAD)['ctx']
    #     assert pretrained_ctx.size()[0] == cfg.DIFO_N_CTX
    #     with torch.no_grad():
    #         model.prompt_learner.ctx.copy_(pretrained_ctx)
    #         model.prompt_learner.ctx_init_state = pretrained_ctx

    for name, param in model.named_parameters():
        if "prompt_learner" not in name:
            param.requires_grad_(False)

    trainable_param = model.prompt_learner.parameters()
    if 'RN' in cfg.DIFO_ARCH :
        prompt_lr = cfg.surrogate_lr*0.1
        data_transform = image_test_50()
    else :
        data_transform = image_test()
        prompt_lr = cfg.surrogate_lr

    optimizer = torch.optim.SGD(trainable_param, prompt_lr,weight_decay=5e-4,momentum=0.9,nesterov=False)
    optim_state = deepcopy(optimizer.state_dict())
    cudnn.benchmark = True
    set_id = 'sfuda'
    model.reset_classnames(classnames, cfg.DIFO_ARCH)
 
    text_features = test_time_adapt_eval(val_loader, model, optimizer, optim_state, cfg)
    return text_features

def test_time_adapt_eval(val_loader, model, optimizer, optim_state, cfg):
    with torch.no_grad():
        model.train()
    iter_test = iter(val_loader)
    max_iter = len(val_loader)
    iter_num = 0
    while iter_num < max_iter:
        try:
            images, target,pesu_label,_ = next(iter_test)
        except:
            iter_test = iter(val_loader)
            images, target,pesu_label,_ = next(iter_test)

        if len(images.size()) > 4:
            assert images.size()[0] == 1
            images = images.squeeze(0)
        images = images.cuda()
        target = target.cuda()
        
        if cfg.DIFO_TTA_STEPS > 0:
            with torch.no_grad():
                model.train()
        optimizer.load_state_dict(optim_state)
        test_time_tuning(model,images,pesu_label, optimizer, cfg)
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                model.eval()
                _,text_features = model(images)
        iter_num = iter_num + 1
    torch.save(model.prompt_learner.state_dict(),cfg.DIFO_LOAD)
    return text_features