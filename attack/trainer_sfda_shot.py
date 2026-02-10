"""
Builds upon: https://github.com/tim-learn/SHOT, https://github.com/tntek/source-free-domain-adaptation
Corresponding paper: http://proceedings.mlr.press/v119/liang20a/liang20a.pdf
"""

import torch.nn as nn
import torch
import utils
import wandb
import numpy as np
from torch.utils.data import DataLoader
from scipy.spatial.distance import cdist

from utils.sfda_utils import SFDA_Trainset, op_copy, set_lr, lr_scheduler
from utils.sfda_loss import Entropy


def obtain_label(loader, model, config):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            labels = torch.argmax(labels, dim=1)

            inputs = inputs.cuda()
            
            if 'cmi' in config.teacher_network:
                outputs, feas = model(inputs, return_features=True)
            else:
                outputs, feas = model.forward_f(inputs)

            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    ent = torch.sum(-all_output * torch.log(all_output + config.SHOT_EPSILON), dim=1)
    unknown_weight = 1 - ent / np.log(config.num_classes)
    _, predict = torch.max(all_output, 1)
    _, all_label = torch.max(all_label, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    if config.SHOT_DISTANCE == 'cosine':
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()

    for _ in range(2):
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        cls_count = np.eye(K)[predict].sum(axis=0)
        labelset = np.where(cls_count>config.SHOT_THRESHOLD)
        labelset = labelset[0]

        dd = cdist(all_fea, initc[labelset], config.SHOT_DISTANCE)
        pred_label = dd.argmin(axis=1)
        predict = labelset[pred_label]

        aff = np.eye(K)[predict]

    acc = np.sum(predict == all_label.float().numpy()) / len(all_fea)
    log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)
    # print(log_str)
    # logging.info(log_str)
    return predict.astype('int')


def SHOT(config, dataloader_train_srgt, dataloader_val, dataloader_test, 
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
    optimizer = torch.optim.SGD(param_group)
    optimizer = op_copy(optimizer)
    
    max_iter = config.surrogate_epochs * len(trainloader_ta)
    interval_iter = max_iter // config.surrogate_interval
    if interval_iter == 0:
        interval_iter = 1
    iter_num = 0
    epoch = -1
    
    model.train()
    while iter_num < max_iter:
        try:
            inputs_test, _, tar_idx = next(iter_test)
        except:
            iter_test = iter(trainloader_ta)
            inputs_test, _, tar_idx = next(iter_test)
            epoch += 1

        if inputs_test.size(0) == 1:
            continue

        if iter_num % interval_iter == 0 and config.SHOT_CLS_PAR > 0:
            model.eval()
            mem_label = obtain_label(trainloader_te, model, config)
            mem_label = torch.from_numpy(mem_label).to(config.device)
            model.train()

        inputs_test = inputs_test.cuda()

        iter_num += 1
        lr_scheduler(config, optimizer, iter_num=iter_num, max_iter=max_iter)

        if 'cmi' in config.teacher_network:
            outputs_test, features_test = model(inputs_test, return_features=True)
        else:
            outputs_test, features_test = model.forward_f(inputs_test)

        if config.SHOT_CLS_PAR > 0:
            pred = mem_label[tar_idx]
            classifier_loss = nn.CrossEntropyLoss()(outputs_test, pred)
            classifier_loss *= config.SHOT_CLS_PAR
            if iter_num < interval_iter and 'visda' in config.domain_tgt:
                classifier_loss *= 0
        else:
            classifier_loss = torch.tensor(0.0).cuda()

        if config.SHOT_ENT:
            softmax_out = nn.Softmax(dim=1)(outputs_test)
            entropy_loss = torch.mean(Entropy(softmax_out))
            if config.SHOT_GENT:
                msoftmax = softmax_out.mean(dim=0)
                gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + config.SHOT_EPSILON))
                entropy_loss -= gentropy_loss
            im_loss = entropy_loss * config.SHOT_ENT_PAR
            classifier_loss += im_loss

        optimizer.zero_grad()
        classifier_loss.backward()
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
            if iter_num == 0 or bestlogger.log(acc1s[0], tgt_mean):
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
            print('[SFDA-SHOT] | epoch %03d | iter %03d | train_loss: %.3f | src_val_acc1: %.1f, tgt_val_acc1: ' %
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
