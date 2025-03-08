"""
Builds upon: https://github.com/conditionWang/NTL
Corresponding paper: https://arxiv.org/pdf/2106.06916
"""

import torch.nn as nn
import torch
import torch.nn.functional as F
from termcolor import cprint
import utils.evaluators
import wandb
from torch.optim import lr_scheduler
from utils.utils import model_dict_cmi


def train_tntl(config, dataloaders, valloaders, testloaders, model, datasets_name):
    if 'cmt' in config.domain_src:
        topk = (1, )  # 2-classification, use top1 acc
    else:
        topk = (1, 5)  # 10-classification, use top1 and top5 acc
    
    evaluators = [utils.evaluators.classification_evaluator(
        v, topk) for v in valloaders]
    evaluators_test = [utils.evaluators.classification_evaluator(
        v, topk) for v in testloaders]
    
    # lr = 0.0001
    lr = 0.0001
    print('use default cifar-stl learning rate')
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    lambda1 = lambda epoch:0.999**epoch
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    #criterion = torch.nn.CrossEntropyLoss()
    criterion_KL = torch.nn.KLDivLoss()

    device = config.device
    cnt = 0
    for epoch in range(config.pretrain_epochs):
        model.train()
        for i, zipped in enumerate(zip(dataloaders[0], dataloaders[1])):
            img1 = zipped[0][0].to(device).float()
            label1 = zipped[0][1].to(device).float()
            img2 = zipped[1][0].to(device).float()
            label2 = zipped[1][1].to(device).float()
            
            if config.teacher_network in model_dict_cmi.keys():
                # out1, fe1 = model(img1, return_features=True)
                # out2, fe2 = model(img2, return_features=True)
                if epoch == 0 and i == 0:
                    print('NTL combine training')
                out, feat = model(torch.cat((img1, img2), 0), return_features=True)
                out1, out2 = out.chunk(2, dim=0)
                fe1, fe2 = feat.chunk(2, dim=0)
            else:
                # out1, out2, fe1, fe2 = model(img1, img2)
                if config.teacher_network in ['vgg11', 'vgg13', 'vgg19']:
                    if epoch == 0 and i == 0:
                        print('NTL seperate training for vgg without bn')
                    out1, out2, fe1, fe2 = model(img1, img2)
                else:
                    if epoch == 0 and i == 0:
                        print('NTL combine training')
                    out, feat = model.forward_f(torch.cat((img1, img2), 0))
                    out1, out2 = out.chunk(2, dim=0)
                    fe1, fe2 = feat.chunk(2, dim=0)

            out1 = F.log_softmax(out1,dim=1)
            loss1 = criterion_KL(out1, label1)
            #loss = criterion(out, label.squeeze())

            out2 = F.log_softmax(out2,dim=1)
            loss2 = criterion_KL(out2, label2)#?change to 0.01 when different dataset, 0.1 on watermark

            # set important parameters
            if 'NTL_alpha' in dict(config).keys() and 'NTL_beta' in dict(config).keys():
                alpha = config.NTL_alpha
                beta = config.NTL_beta
            else:
                alpha = 0.1
                beta = 0.1
                if epoch == 0 and i == 0:
                    print('use default paras: alpha and beta')

            mmd_loss = MMD_loss()(fe1.view(fe1.size(0), -1), fe2.view(fe2.size(0), -1)) * beta
            loss2 = loss2 * alpha
            if loss2 > 1:
                loss2 = torch.clamp(loss2, 0, 1)#0.01
            if mmd_loss > 1:
                mmd_loss_1 = torch.clamp(mmd_loss, 0, 1)
            else:
                mmd_loss_1 = mmd_loss
            
            loss = loss1 - loss2 * mmd_loss_1
            # loss = loss1
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            cnt += 1
            # if i % 100 == 0:
            #     print('Epoch:{0},Frame:{1}, train_loss {2}'.format(
            #         epoch, cnt*config.batch_size, loss.item()))
            #     print('mmd loss: ', mmd_loss.item())
        
        model.eval()
        acc1s, acc5s, val_celoss = [], [], []
        for evaluator in evaluators:
            eval_results = evaluator(model, device=config.device)
            if topk == (1, 5):
                (acc1, acc5), val_celoss_ = eval_results['Acc'], eval_results['Loss']
            elif topk == (1, ):
                acc, val_celoss_ = eval_results['Acc'], eval_results['Loss']
                acc1, acc5 = acc[0], 0.0
            acc1s.append(acc1)
            acc5s.append(acc5)
            val_celoss.append(val_celoss_)
        # scheduler.step()
        # is_best = acc1 > best_acc1
        # best_acc1 = max(acc1, best_acc1)

        wandb_log_dict = {
            'pt_epoch': epoch,
            'pt_loss': loss.item(),
            'pt_mmd_loss': mmd_loss.item(),
            'pt_Acc_val_src': acc1s[0],
        }
        ########## insert log of mddloss 

        print('[NTL Pretrain] | epoch %03d | train_loss: %.3f, mmd_loss: %.3f | src_val_acc1: %.1f, tgt_val_acc1: ' %
              (epoch, loss.item(), mmd_loss.item(), acc1s[0]), end='')
        for dname, acc_tgt in zip(datasets_name[1:], acc1s[1:]):
            print(f'{dname}: {acc_tgt:.2f} ', end='')
            wandb_log_dict[f'pt_Acc_val_tgt_{dname}'] = acc_tgt
        tgt_mean = torch.mean(torch.tensor(acc1s[1:])).item()
        wandb_log_dict[f'pt_Acc_val_tgt_mean'] = tgt_mean
        print(f'| tgt_mean: {tgt_mean:.2f}')
        
        wandb.log(wandb_log_dict)

    # test 
    test_acc1s, _ = utils.evaluators.eval_func(config, evaluators_test, model)

    # wandb.run.summary['final_valbest_Acc_ft_src'] = bestlogger.result()['src']
    # wandb.run.summary['final_valbest_Acc_ft_tgtmean'] = bestlogger.result()['tgt']
    wandb.run.summary['pt_Acc_test_src'] = test_acc1s[0]
    for dname, acc_tgt in zip(datasets_name[1:], test_acc1s[1:]):
        wandb.run.summary[f'pt_Acc_test_tgt_{dname}'] = acc_tgt
    test_acc1s_tgt_mean = torch.mean(torch.tensor(test_acc1s[1:])).item()
    wandb.run.summary[f'pt_Acc_test_tgt_mean'] = test_acc1s_tgt_mean

    return test_acc1s[0], test_acc1s_tgt_mean    



class MMD_loss(nn.Module):
    def __init__(self, kernel_mul = 2.0, kernel_num = 5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        return
    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2) 
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY -YX)
        return loss
    