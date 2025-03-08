import torch.nn as nn
import torch
import torch.nn.functional as F
from termcolor import cprint
import utils.evaluators
import wandb
from torch.optim import lr_scheduler
import utils.hntl_loss_func as hntl_loss_func
import models.hntl_vggnet as hntl_vggnet
import models.hntl_vit as hntl_vit


def train_tHNTL(config, dataloaders, valloaders, testloaders, model, datasets_name):

    device = config.device
    if 'vgg' in config.teacher_network:
        model_c, model_s, model_dec = hntl_vggnet.creat_disentangle_model(config)
    elif 'vit' in config.teacher_network:
        model_c, model_s, model_dec = hntl_vit.creat_disentangle_model(config)
    model_c = model_c.to(device)
    model_s = model_s.to(device)
    model_dec = model_dec.to(device)
    cprint('training disentanglement (model_c, model_s, model_dec):', 'yellow')
    print(f'\tlr: {config.HNTL_Disen_lr}, epoch: {config.HNTL_Disen_epochs}')
    print('training KD (model):', 'yellow')
    print(f'\tlr: {config.HNTL_KD_lr}, epoch: {config.HNTL_KD_epochs}')

    # stage 1: disentanglement
    train_HNTL_disentangle(config, dataloaders, valloaders, model_c, model_s, model_dec, datasets_name)
    # stage 1: kd
    train_HNTL_KD(config, dataloaders, valloaders, model_c, model_s, model, datasets_name)

    # test
    if 'cmt' in config.domain_src:
        topk = (1, )  # 2-classification, use top1 acc
    else:
        topk = (1, 5)  # 10-classification, use top1 and top5 acc
    evaluators_test = [utils.evaluators.classification_evaluator(
        v, topk) for v in testloaders]
    
    if len(testloaders) > 2: raise NotImplementedError
    
    test_acc1s, _ = utils.evaluators.eval_func(config, evaluators_test, model)

    wandb.run.summary['pt_Acc_test_src'] = test_acc1s[0]
    for dname, acc_tgt in zip(datasets_name[1:], test_acc1s[1:]):
        wandb.run.summary[f'pt_Acc_test_tgt_{dname}'] = acc_tgt
    test_acc1s_tgt_mean = torch.mean(torch.tensor(test_acc1s[1:])).item()
    wandb.run.summary[f'pt_Acc_test_tgt_mean'] = test_acc1s_tgt_mean
    
    return test_acc1s[0], test_acc1s_tgt_mean


def train_HNTL_disentangle(config, dataloaders, valloaders, model_c, model_s, model_dec, datasets_name):
    optimizer_c = torch.optim.SGD(model_c.parameters(),
                                lr=config.HNTL_Disen_lr,
                                momentum=config.HNTL_Disen_momentum,
                                weight_decay=config.HNTL_Disen_weight_decay)
    optimizer_s = torch.optim.SGD(model_s.parameters(),
                                lr=config.HNTL_Disen_lr,
                                momentum=config.HNTL_Disen_momentum,
                                weight_decay=config.HNTL_Disen_weight_decay)
    optimizer_dec = torch.optim.SGD(model_dec.parameters(),
                                lr=config.HNTL_Disen_lr,
                                momentum=config.HNTL_Disen_momentum,
                                weight_decay=config.HNTL_Disen_weight_decay)
    
    if 'cmt' in config.domain_src:
        topk = (1, )  # 2-classification, use top1 acc
    else:
        topk = (1, 5)  # 10-classification, use top1 and top5 acc
    evaluators = [utils.evaluators.classification_evaluator(
        v, topk) for v in valloaders]
    best_acc1 = 0
    for epoch in range(config.HNTL_Disen_epochs):
        model_c.train()
        model_s.train()
        for i, zipped in enumerate(zip(dataloaders[0], dataloaders[1])):
            imgs_src = zipped[0][0]
            labels_src = zipped[0][1]
            imgs_tgt = zipped[1][0]
            labels_tgt = zipped[1][1]

            imgs = torch.cat([imgs_src, imgs_tgt], dim=0).to(config.device)
            labels_c = torch.cat([labels_src, labels_tgt], dim=0).to(config.device)
            labels_c = torch.argmax(labels_c, dim=1)
            labels_s = torch.zeros_like(labels_c).to(config.device)
            labels_s[config.batch_size:] = 1


            outputs_c = model_c(imgs)
            outputs_s = model_s(imgs)
            x_rec = model_dec(outputs_c['f_rp'], outputs_s['f_rp'])

            loss_cls_c = hntl_loss_func.cal_content_classification(
                outputs_c['pred'], labels_c)
            loss_vae_c = config.HNTL_lambda_vae * hntl_loss_func.cal_vae_loss(
                outputs_c['f_mu'], outputs_c['f_logvar'])

            loss_cls_s = hntl_loss_func.cal_style_classification(
                outputs_s['pred'], labels_s)
            loss_vae_s = config.HNTL_lambda_vae * hntl_loss_func.cal_vae_loss(
                outputs_s['f_mu'], outputs_s['f_logvar'])

            loss_rec = config.HNTL_rec_coef * hntl_loss_func.compute_rec(x_rec, imgs)

            # do not add disentangle loss
            loss_disent_c = torch.zeros(1).to(config.device).mean()
            loss_disent_s = torch.zeros(1).to(config.device).mean()

            loss_c = loss_cls_c + loss_vae_c + loss_disent_c
            loss_s = loss_cls_s + loss_vae_s + loss_disent_s
            loss_rec = loss_rec
            loss = loss_c + loss_s + loss_rec
            
            optimizer_c.zero_grad()
            optimizer_s.zero_grad()
            optimizer_dec.zero_grad()
            loss.backward()
            optimizer_c.step()
            optimizer_s.step()
            optimizer_dec.step()

        model_c.eval()
        model_s.eval()

        test_results = hntl_loss_func.eval_disentangle(config, model_c, model_s, valloaders)

        print(f'[HNTL Pretrain S1:Disentangle]', end='')
        print(' | epoch %03d' % (epoch))
        print('\tContent: loss_c: %.3f, loss_cls_c: %.3f, loss_rec_c: %.3f, loss_vae_c: %.3f, loss_disent_c: %.3f' % (
            loss_c.item(), loss_cls_c.item(), loss_rec.item(), loss_vae_c.item(), loss_disent_c.item()))
        print('\tStyle:   loss_s: %.3f, loss_cls_s: %.3f, loss_rec_s: %.3f, loss_vae_s: %.3f, loss_disent_s: %.3f' % (
            loss_s.item(), loss_cls_s.item(), loss_rec.item(), loss_vae_s.item(), loss_disent_s.item()))
        # print('\tContent -> Label: src_val_acc1: %.1f, tgt_val_acc1: %.1f' % (
        #     acc_content['acc1s'][0], acc_content['acc1s'][1]))
        print('\tContent -> Label: src_val_acc1: %.1f, tgt_val_acc1: %.1f' % (
            test_results['content_2_label_src'][0], test_results['content_2_label_tgt'][0]))
        print('\tStyle -> Label:   src_val_acc1: %.1f, tgt_val_acc1: %.1f' % (
            test_results['style_2_label_src'][0], test_results['style_2_label_tgt'][0]))
        print('\tContent -> Domain: %.1f, Style -> Domain: %.1f\n' % (
            test_results['content_2_domain'][0], test_results['style_2_domain'][0]))
        wandb.log({
            'pt_disen_epoch': epoch,
            'pt_loss_c': loss_c.item(),
            'pt_loss_s': loss_s.item(),
            'pt_loss_cls_c': loss_cls_c.item(),
            'pt_loss_rec_c': loss_rec.item(),
            'pt_loss_vae_c': loss_vae_c.item(),
            'pt_loss_cls_s': loss_cls_s.item(),
            'pt_loss_rec_s': loss_rec.item(),
            'pt_loss_vae_s': loss_vae_s.item(),
            'pt_Acc_Disen_C2L_src': test_results['content_2_label_src'][0],
            'pt_Acc_Disen_C2L_tgt': test_results['content_2_label_tgt'][0],
            'pt_Acc_Disen_S2L_src': test_results['style_2_label_src'][0],
            'pt_Acc_Disen_S2L_tgt': test_results['style_2_label_tgt'][0],
            'pt_Acc_Disen_C2D': test_results['content_2_domain'][0],
            'pt_Acc_Disen_S2D': test_results['style_2_domain'][0]
            # 'val_acc1_src': acc1s[0],
            # 'val_acc1_tgt': acc1s[1],
        })
    pass


def train_HNTL_KD(config, dataloaders, valloaders, model_c, model_s, model_student, datasets_name):
    optimizer = torch.optim.SGD(model_student.parameters(),
                                lr=config.HNTL_KD_lr,
                                momentum=config.HNTL_KD_momentum,
                                weight_decay=config.HNTL_KD_weight_decay)
    # optimizer = torch.optim.Adam(model_student.parameters(),
    #                             lr=config.KD_lr,
    #                             weight_decay=config.KD_weight_decay)

    if 'cmt' in config.domain_src:
        topk = (1, )  # 2-classification, use top1 acc
    else:
        topk = (1, 5)  # 10-classification, use top1 and top5 acc
    evaluators = [utils.evaluators.classification_evaluator(
        v, topk) for v in valloaders]
    # best_acc1 = 0

    model_c.eval()
    model_s.eval()
    for para in model_c.parameters():
        para.requires_grad = False
    for para in model_s.parameters():
        para.requires_grad = False

    for epoch in range(config.HNTL_KD_epochs):
        model_student.train()

        for i, zipped in enumerate(zip(dataloaders[0], dataloaders[1])):
            imgs_src = zipped[0][0]
            labels_src = zipped[0][1]
            imgs_tgt = zipped[1][0]
            labels_tgt = zipped[1][1]

            imgs = torch.cat([imgs_src, imgs_tgt], dim=0).to(config.device)
            labels_c = torch.cat([labels_src, labels_tgt], dim=0).to(config.device)
            labels_c = torch.argmax(labels_c, dim=1)
            labels_s = torch.zeros_like(labels_c).to(config.device)
            labels_s[config.batch_size:] = 1

            with torch.no_grad():
                outputs_c = model_c.forward_full(imgs)
                outputs_s = model_s.forward_full(imgs)
            outputs = model_student(imgs)
            
            loss_src_c = F.mse_loss(outputs[:config.batch_size, :], 
                                    outputs_c['pred'][:config.batch_size, :])
            loss_tgt_s = F.mse_loss(outputs[config.batch_size:, :], 
                                    model_c.pred(outputs_s['f_rp'])[config.batch_size:, :])
            
            loss = loss_src_c + loss_tgt_s
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model_student.eval()

        # test_results = hntl_loss_func.test_disentangle_KD(
        #     config, model_c, model_s, model_student, valloaders)
        acc1s, acc5s = [], []
        for evaluator in evaluators:
            eval_results = evaluator(model_student, device=config.device)
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
            'pt_loss_kd_src_c': loss_src_c.item(),
            'pt_loss_kd_tgt_s': loss_tgt_s.item(),
            'pt_Acc_val_src': acc1s[0],
            # 'Acc_src': acc1,
        }

        print('[HNTL Pretrain S2:KD] | epoch %03d | KD loss: %.3f | src_val_acc1: %.1f, tgt_val_acc1: ' %
              (epoch, loss.item(), acc1s[0]), end='')
            # (epoch, loss.item(), acc1), end='')
        for dname, acc_tgt in zip(datasets_name[1:], acc1s[1:]):
            print(f'{dname}: {acc_tgt:.2f} ', end='')
            wandb_log_dict[f'pt_Acc_val_tgt_{dname}'] = acc_tgt
        tgt_mean = torch.mean(torch.tensor(acc1s[1:])).item()
        wandb_log_dict[f'pt_Acc_val_tgt_mean'] = tgt_mean
        print(f'| tgt_mean: {tgt_mean:.2f}')
        wandb.log(wandb_log_dict)