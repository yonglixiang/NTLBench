"""
Builds upon: https://github.com/LyWang12/CUTI-Domain
Corresponding paper: https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_Model_Barrier_A_Compact_Un-Transferable_Isolation_Domain_for_Model_Intellectual_CVPR_2023_paper.pdf
"""

import torch.nn as nn
import torch
import torch.nn.functional as F
from termcolor import cprint
import utils.evaluators
import wandb
from torch.optim import lr_scheduler
    

def calc_ins_mean_std(x, eps=1e-5):
        """extract feature map statistics"""
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


class CUTI(nn.Module):
    def __init__(self):
        super(CUTI, self).__init__()
    def forward(self, x):
        if self.training:
            if x.dim() == 4:  # for vgg, resnet
                batch_size, C = x.size()[0]//2, x.size()[1]
                style_mean, style_std = calc_ins_mean_std(x[:batch_size])
                conv_mean = nn.Conv2d(C, C, 1, bias=False).cuda()
                conv_std = nn.Conv2d(C, C, 1, bias=False).cuda()
                mean = torch.sigmoid(conv_mean(style_mean))
                std = torch.sigmoid(conv_std(style_std))
                x_a = x[batch_size:]*std+mean
                x = torch.cat((x[:batch_size], x_a), 0)
            elif x.dim() == 3:  # for vit
                batch_size, C = x.size()[0] // 2, x.size()[2]
                style_mean, style_std = calc_ins_mean_std(x[:batch_size])
                conv_mean = nn.Conv1d(1, 1, 1, bias=False).cuda()
                conv_std = nn.Conv1d(1, 1, 1, bias=False).cuda()
                mean = torch.sigmoid(conv_mean(style_mean))
                std = torch.sigmoid(conv_std(style_std))
                x_a = (x[batch_size:]) * std + mean
                x = torch.cat((x[:batch_size], x_a), 0)
        return x

def forward_CUTI(model, x, y=None, choice=0):
    if y == None:
        x = model.features(x)
        x = x.view(x.size(0), -1)
        x = model.classifier1(x)
        return x
    elif choice % 2 == 0:
        input = torch.cat((x, y), 0)
        input = model.features(input)
        x, y = input.chunk(2, dim=0)

        x = x.view(x.size(0), -1)
        x = model.classifier1(x)

        y = y.view(y.size(0), -1)
        y = model.classifier1(y)

        return x, y

    elif choice % 2 == 1:
        input = torch.cat((x, y), 0)
        input = model.chan_ex(model.features[:3](input))
        input = model.chan_ex(model.features[3:6](input))
        input = model.chan_ex(model.features[6:11](input))
        input = model.chan_ex(model.features[11:16](input))
        input = model.chan_ex(model.features[16:](input))

        x, y = input.chunk(2, dim=0)

        x = x.view(x.size(0), -1)
        x = model.classifier1(x)

        y = y.view(y.size(0), -1)
        y = model.classifier1(y)

        return x, y
    

def forward_CUTI_resnet(model, x, y=None, choice=0):
    bs = x.shape[0]
    if y == None:
        return model(x)
    elif choice % 2 == 0:
        input = torch.cat((x, y), 0)
        output = model(input)
        x = output[:bs, :]
        y = output[bs:, :]
        return x, y
    elif choice % 2 == 1:
        input = torch.cat((x, y), 0)

        out = model.conv1(input)
        out = model.bn1(out)
        out = F.relu(out)
        # CUTI
        out = model.chan_ex(out)

        out = model.layer1(out)
        # CUTI
        out = model.chan_ex(out)

        out = model.layer2(out)
        # CUTI
        out = model.chan_ex(out)

        out = model.layer3(out)
        # CUTI
        out = model.chan_ex(out)

        out = model.layer4(out)
        # CUTI
        out = model.chan_ex(out)

        out = F.adaptive_avg_pool2d(out, (1,1))
        feature = out.view(out.size(0), -1)
        # output = model.linear(feature)
        output = model.fc(feature)
        
        x = output[:bs, :]
        y = output[bs:, :]
        return x, y
    
def forward_CUTI_vit(model, x, y=None, choice=0):
    if y == None:
        return model(x)
    
    elif choice % 2 == 0:
        input = torch.cat((x, y), 0)
        output = model(input)
        x, y = output.chunk(2, dim=0)
        return x, y
    
    elif choice % 2 == 1:
        # Concatenate x and y
        input = torch.cat((x, y), 0)
        
        # Patch embedding and positional encoding
        input = model.patch_embed(input)
        input = model._pos_embed(input)
        input = model.patch_drop(input)
        input = model.norm_pre(input)
        
        # Transformer Blocks
        for block in model.blocks:
            input = block(input)  # extract features
            input = model.chan_ex(input)  # style transfer
        input = model.norm(input)
       
        # CLS
        out = model.forward_head(input)
        out_x, out_y = out.chunk(2, dim=0)
        
        return out_x, out_y
        
def train_tCUTI(config, dataloaders, valloaders, testloaders, model, datasets_name):
    if 'cmt' in config.domain_src:
        topk = (1, )  # 2-classification, use top1 acc
    else:
        topk = (1, 5)  # 10-classification, use top1 and top5 acc
    
    evaluators = [utils.evaluators.classification_evaluator(
        v, topk) for v in valloaders]
    evaluators_test = [utils.evaluators.classification_evaluator(
        v, topk) for v in testloaders]
    
    model.chan_ex = CUTI()
    lr = 0.0001
    # lr = 0.00001
    print('use default cifar-stl learning rate')

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    lambda1 = lambda epoch:0.999**epoch
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    criterion_KL = torch.nn.KLDivLoss()

    if config.teacher_network in ['vgg11', 'vgg13', 'vgg19', 'vgg11bn', 'vgg13bn', 'vgg19bn']:
        forward_func = forward_CUTI
    elif config.teacher_network in ['resnet18', 'resnet34', 'resnet50', 'wide_resnet50_2']:
        forward_func = forward_CUTI_resnet
    elif config.teacher_network in ['vit_tiny', 'vit_small', 'vit_base', 'vit_large']:
        forward_func = forward_CUTI_vit
    else:
        raise NotImplementedError

    device = config.device
    # cnt = 0
    for epoch in range(config.pretrain_epochs):
        model.train()
        for i, zipped in enumerate(zip(dataloaders[0], dataloaders[1])):
            img1 = zipped[0][0].to(device).float()
            label1 = zipped[0][1].to(device).float()
            img2 = zipped[1][0].to(device).float()
            label2 = zipped[1][1].to(device).float()
            
            out1, out2 = forward_func(model, img1, img2, i)

            out1 = F.log_softmax(out1, dim=1)
            loss1 = criterion_KL(out1, label1)

            out2 = F.log_softmax(out2, dim=1)
            loss2 = criterion_KL(out2, label2)

            # set important parameters
            if 'CUTI_alpha' in dict(config).keys():
                alpha = config.CUTI_alpha
            else:
                alpha = 0.1
                # alpha = 0.5
                # print('use default paras: alpha and beta')

            loss2 = loss2 * alpha
            if loss2 > 1:
                loss2 = torch.clamp(loss2, 0, 1)

            loss = loss1 - loss2
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
        
        model.eval()
        # acc1 = validate_class(valloaders[0], model, epoch, num_class=config.num_classes)
        # acc2 = validate_class(valloaders[1], model, epoch, num_class=config.num_classes)
        # print("epoch = {:02d}, acc_s = {:.3f}, acc_t = {:.3f}".format(epoch, acc1, acc2) + '\n')

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

        print('[CUTI Pretrain] | epoch %03d | train_loss: %.3f | src_val_acc1: %.1f, tgt_val_acc1: ' %
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

