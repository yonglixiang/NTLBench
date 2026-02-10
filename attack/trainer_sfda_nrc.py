"""
Builds upon: https://github.com/Albert0147/NRC_SFDA, https://github.com/tntek/source-free-domain-adaptation
Corresponding paper: https://proceedings.neurips.cc/paper_files/paper/2021/file/f5deaeeae1538fb6c45901d524ee2f98-Paper.pdf
"""

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb

import utils
from utils.sfda_utils import SFDA_Trainset, op_copy, set_lr, lr_scheduler, MODEL_FEATURE_DIM
from utils.sfda_loss import Entropy, Entropywt, grl_hook, CDAN, DANN

def NRC(config, dataloader_train_srgt, dataloader_val, dataloader_test, 
         model, datasets_name):
    
    # shot train dataloader
    datafile_ta = SFDA_Trainset(dataloader_train_srgt[1])
    datafile_te = SFDA_Trainset(dataloader_train_srgt[1])
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
    
    # optimize
    param_group = set_lr(model, config)
    optimizer = torch.optim.SGD(param_group, 
                                momentum=config.surrogate_momentum,
                                weight_decay=config.surrogate_weight_decay,
                                nesterov=config.surrogate_NESTEROV
                                )
    optimizer = op_copy(optimizer)
    
    # set up the training parameters
    acc_init = 0
    start = True
    num_sample = len(trainloader_ta.dataset)
    fea_bank = torch.randn(num_sample, MODEL_FEATURE_DIM[config.teacher_network])
    score_bank = torch.randn(num_sample, config.num_classes).cuda()

    
    max_iter = config.surrogate_epochs * len(trainloader_ta)
    interval_iter = max_iter // config.surrogate_interval
    if interval_iter == 0:
        interval_iter = 1
    iter_num = 0
    epoch = -1
    
    # test the model before training
    model.eval()
    with torch.no_grad():
        iter_test = iter(trainloader_ta)
        for i in range(len(trainloader_ta)):
            data = next(iter_test)
            inputs = data[0]
            indx = data[-1]
            #labels = data[1]
            inputs = inputs.cuda()
            if 'cmi' in config.teacher_network:
                outputs, _ = model(inputs)
            else:
                outputs, features = model.forward_f(inputs)
            
            features_norm = F.normalize(features)
            outputs = nn.Softmax(-1)(outputs)
            fea_bank[indx] = features_norm.detach().clone().cpu()
            score_bank[indx] = outputs.detach().clone() #.cpu()
            #all_label = torch.cat((all_label, labels.float()), 0)
        #fea_bank = fea_bank.detach().cpu().numpy()
        #score_bank = score_bank.detach()
    
    config.NRC_K = 5
    config.NRC_KK = 4
    
    # start training
    model.train()
    while iter_num < max_iter:
        if 'office-31' not in config.domain_tgt:
            if iter_num > 0.5*max_iter:
                config.NRC_K = 5
                config.NRC_KK = 4
        
        # Get the inputs
        try:
            inputs_test, _, tar_idx = next(iter_test)
        except:
            iter_test = iter(trainloader_ta)
            inputs_test, _, tar_idx = next(iter_test)
            epoch += 1

        if inputs_test.size(0) == 1:
            continue
        
        # Epoch update
        inputs_test = inputs_test.cuda()
        iter_num += 1
        if 'office-31' in config.domain_tgt:
            lr_scheduler(config, optimizer, iter_num=iter_num, max_iter=max_iter)
        
        # Forward, get the output and features
        inputs_target = inputs_test.cuda()
        if 'cmi' in config.teacher_network:
            output, features_test = model(inputs_target)
        else:
            output, features_test = model.forward_f(inputs_target)
        softmax_out = nn.Softmax(dim=1)(output)
        output_re = softmax_out.unsqueeze(1)  # batch x 1 x num_class

        with torch.no_grad():
            output_f_norm = F.normalize(features_test)
            output_f_ = output_f_norm.cpu().detach().clone()

            fea_bank[tar_idx] = output_f_.detach().clone().cpu()
            score_bank[tar_idx] = softmax_out.detach().clone()

            
            distance = output_f_ @ fea_bank.T
            _, idx_near = torch.topk(distance,
                                     dim=-1,
                                     largest=True,
                                     k=config.NRC_K + 1)
            idx_near = idx_near[:, 1:]  #batch x K
            score_near = score_bank[idx_near]  #batch x K x C
            #score_near=score_near.permute(0,2,1)

            fea_near = fea_bank[idx_near]  #batch x K x num_dim
            fea_bank_re = fea_bank.unsqueeze(0).expand(fea_near.shape[0], -1,
                                                       -1)  # batch x n x dim
            distance_ = torch.bmm(fea_near,
                                  fea_bank_re.permute(0, 2,
                                                      1))  # batch x K x n
            _, idx_near_near = torch.topk(
                distance_, dim=-1, largest=True,
                k=config.NRC_KK + 1)  # M near neighbors for each of above K ones
            idx_near_near = idx_near_near[:, :, 1:]  # batch x K x M
            tar_idx_ = tar_idx.unsqueeze(-1).unsqueeze(-1)
            match = (idx_near_near == tar_idx_).sum(-1).float()  # batch x K
            weight = torch.where(
                match > 0., match,
                torch.ones_like(match).fill_(0.1))  # batch x K

            weight_kk = weight.unsqueeze(-1).expand(-1, -1, config.NRC_KK)  # batch x K x M

            
            #weight_kk[idx_near_near == tar_idx_] = 0

            score_near_kk = score_bank[idx_near_near]  # batch x K x M x C
            #print(weight_kk.shape)
            weight_kk = weight_kk.contiguous().view(weight_kk.shape[0],
                                                    -1)  # batch x KM
            weight_kk = weight_kk.fill_(0.1)
            score_near_kk = score_near_kk.contiguous().view(
                score_near_kk.shape[0], -1, config.num_classes)  # batch x KM x C

        # nn of nn
        output_re = softmax_out.unsqueeze(1).expand(-1, config.NRC_K * config.NRC_KK,
                                                    -1)  # batch x KM x C
        const = torch.mean(
            (F.kl_div(output_re, score_near_kk, reduction='none').sum(-1) *
             weight_kk.cuda()).sum(1))
        loss = torch.mean(const)  #* 0.5

        # nn
        softmax_out_un = softmax_out.unsqueeze(1).expand(-1, config.NRC_K,
                                                         -1)  # batch x K x C
        
        loss += torch.mean(
            (F.kl_div(softmax_out_un, score_near, reduction='none').sum(-1) *
             weight.cuda()).sum(1))  #

        msoftmax = softmax_out.mean(dim=0)
        im_div = torch.sum(msoftmax * torch.log(msoftmax + 1e-5))
        loss += im_div  

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if iter_num % interval_iter == 0 or iter_num == max_iter:
            model.eval()
            
            # validation
            if topk == (1, 5):
                acc1s, _ = utils.evaluators.eval_func(config, evaluators_val, model)
            elif topk == (1, ):
                acc = utils.evaluators.eval_func(config, evaluators_val, model)
                acc1s = acc[0]
            tgt_mean = torch.mean(torch.tensor(acc1s[1:])).item()
            # if best on validation, test 
            if epoch == 0 or bestlogger.log(acc1s[0], tgt_mean):
                if topk == (1, 5):
                    test_acc1s, _ = utils.evaluators.eval_func(config, evaluators_test, model)
                elif topk == (1, ):
                    test_acc = utils.evaluators.eval_func(config, evaluators_test, model)
                    test_acc1s = test_acc[0]

            wandb_log_dict = {
                'iter_ft': iter_num,
                'epoch_ft': epoch,
                'loss_ft': loss.item(),
                'Acc_ft_src': acc1s[0]}

            # print validation
            print('[SFDA-NRC] | epoch %03d | iter %03d | train_loss: %.3f | src_val_acc1: %.1f, tgt_val_acc1: ' %
                (epoch, iter_num, loss.item(), acc1s[0]), end='')
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
