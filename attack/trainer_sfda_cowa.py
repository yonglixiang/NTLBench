"""
Builds upon: https://github.com/Jhyun17/CoWA-JMDS, https://github.com/tntek/source-free-domain-adaptation
Corresponding paper: https://proceedings.mlr.press/v162/lee22c/lee22c.pdf
"""

import torch.nn as nn
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from torchvision import transforms
from tqdm import tqdm
import pickle
import math, os, wandb

import utils
from utils.sfda_utils import SFDA_Trainset, op_copy, set_lr, lr_scheduler
from utils.sfda_loss import Entropy, Entropywt, grl_hook, CDAN, DANN

def gmm(cfg,all_fea, pi, mu, all_output):    
    Cov = []
    dist = []
    log_probs = []
    
    for i in range(len(mu)):
        temp = all_fea - mu[i]
        predi = all_output[:,i].unsqueeze(dim=-1)
        sum_predi = predi.sum() 
        if sum_predi < 1e-5:
            sum_predi = 1e-5
        Covi = torch.matmul(temp.t(), temp * predi.expand_as(temp)) / sum_predi + cfg.COWA_EPSILON * torch.eye(temp.shape[1]).cuda()
        try:
            chol = torch.linalg.cholesky(Covi)
        except RuntimeError:
            Covi += cfg.COWA_EPSILON * torch.eye(temp.shape[1]).cuda() * 100
            try:
                chol = torch.linalg.cholesky(Covi)
            except RuntimeError:
                try:
                    eigenvalues, eigenvectors = torch.linalg.eigh(Covi)
                    min_eig = torch.max(eigenvalues) * 1e-6
                    eigenvalues = torch.clamp(eigenvalues, min=min_eig)
                    sqrt_eigenvalues = torch.sqrt(eigenvalues)
                    chol = torch.matmul(eigenvectors, torch.diag(sqrt_eigenvalues))
                except RuntimeError as e:
                    print(f"Eigendecomposition failed for component {i}, using identity matrix")
                    exit(1)
                    chol = torch.eye(temp.shape[1]).cuda() * cfg.COWA_EPSILON * 100
                    
        
        chol_inv = torch.inverse(chol)
        Covi_inv = torch.matmul(chol_inv.t(), chol_inv)
        logdet = torch.logdet(Covi)
        mah_dist = (torch.matmul(temp, Covi_inv) * temp).sum(dim=1)
        log_prob = -0.5*(Covi.shape[0] * np.log(2*math.pi) + logdet + mah_dist) + torch.log(pi)[i]
        Cov.append(Covi)
        log_probs.append(log_prob)
        dist.append(mah_dist)
    Cov = torch.stack(Cov, dim=0)
    dist = torch.stack(dist, dim=0).t()
    log_probs = torch.stack(log_probs, dim=0).t()
    zz = log_probs - torch.logsumexp(log_probs, dim=1, keepdim=True).expand_as(log_probs)
    gamma = torch.exp(zz)
    
    return zz, gamma

def evaluation(loader, model, cfg, cnt):
    start_test = True
    iter_test = iter(loader)
    for _ in tqdm(range(len(loader))):
        data = next(iter_test)
        inputs = data[0]
        labels = data[1].cuda()
        labels = torch.argmax(labels, dim=1)
        inputs = inputs.cuda()
        
        if 'cmi' in cfg.teacher_network:
            outputs, feas = model(inputs, return_features=True)
        else:
            outputs, feas = model.forward_f(inputs)
        if start_test:
            all_fea = feas.float()
            all_output = outputs.float()
            all_label = labels.float()
            start_test = False
        else:
            all_fea = torch.cat((all_fea, feas.float()), 0)
            all_output = torch.cat((all_output, outputs.float()), 0)
            all_label = torch.cat((all_label, labels.float()), 0)
            
    _, predict = torch.max(all_output, 1)
    _, all_label = torch.max(all_label, 1)
    accuracy_return = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(Entropy(nn.Softmax(dim=1)(all_output))).data.item()

    if 'visda' in cfg.domain_tgt:
        matrix = confusion_matrix(all_label.cpu().numpy(), torch.squeeze(predict).float().cpu().numpy())
        acc_return = matrix.diagonal()/matrix.sum(axis=1) * 100
        aacc = acc_return.mean()
        aa = [str(np.round(i, 2)) for i in acc_return]
        acc_return = ' '.join(aa)

    all_output_logit = all_output
    all_output = nn.Softmax(dim=1)(all_output)
    all_fea_orig = all_fea
    ent = torch.sum(-all_output * torch.log(all_output + cfg.COWA_EPSILON2), dim=1)
    unknown_weight = 1 - ent / np.log(cfg.num_classes)

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    if cfg.COWA_DISTANCE == 'cosine':
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    all_fea = all_fea.float()
    K = all_output.shape[1]
    aff = all_output.float()
    initc = torch.matmul(aff.t(), (all_fea))
    initc = initc / (1e-8 + aff.sum(dim=0)[:,None])

    if cfg.COWA_PICKLE and (cnt==0):
        data = {
            'all_fea': all_fea,
            'all_output': all_output,
            'all_label': all_label,
            'all_fea_orig': all_fea_orig,
        }
        filename = os.path.join(cfg.COWA_OUTPUT_DIR, 'data_{}'.format(cfg.domain_tgt) + '.pickle')
        try:
            os.makedirs(cfg.COWA_OUTPUT_DIR)
        except:
            pass
        with open(filename, 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        print('data_{}.pickle finished\n'.format(cfg.domain_tgt))
        
        
    ############################## Gaussian Mixture Modeling #############################

    uniform = torch.ones(len(all_fea),cfg.num_classes)/cfg.num_classes
    uniform = uniform.cuda()

    pi = all_output.sum(dim=0)
    mu = torch.matmul(all_output.t(), (all_fea))
    mu = mu / pi.unsqueeze(dim=-1).expand_as(mu)

    zz, gamma = gmm(cfg,(all_fea), pi, mu, uniform)
    pred_label = gamma.argmax(dim=1)
    
    for round in range(1):
        pi = gamma.sum(dim=0)
        mu = torch.matmul(gamma.t(), (all_fea))
        mu = mu / pi.unsqueeze(dim=-1).expand_as(mu)

        zz, gamma = gmm(cfg,(all_fea), pi, mu, gamma)
        pred_label = gamma.argmax(axis=1)
            
    aff = gamma
    
    acc = (pred_label==all_label).float().mean()
    log_str = 'soft_pseudo_label_Accuracy = {:.2f}%'.format(acc * 100) + '\n'
    # logging.info(log_str)
    # print(log_str)
    # cfg.out_file.write(log_str + '\n')
    # cfg.out_file.flush()

    log_str = 'Model Prediction : Accuracy = {:.2f}%'.format(accuracy * 100) + '\n'

    if 'visda' in cfg.domain_tgt:
        log_str += 'VISDA classwise accuracy : {:.2f}%\n{}'.format(aacc, acc_return) + '\n'
        
    log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)
    # print(log_str)
    
    ############################## Computing JMDS score #############################

    sort_zz = zz.sort(dim=1, descending=True)[0]
    zz_sub = sort_zz[:,0] - sort_zz[:,1]
    
    LPG = zz_sub / zz_sub.max()

    if cfg.COWA_COEFF=='JMDS':
        PPL = all_output.gather(1, pred_label.unsqueeze(dim=1)).squeeze()
        JMDS = (LPG * PPL)
    elif cfg.COWA_COEFF=='PPL':
        JMDS = all_output.gather(1, pred_label.unsqueeze(dim=1)).squeeze()
    elif cfg.COWA_COEFF=='NO':
        JMDS=torch.ones_like(LPG)
    else:
        JMDS = LPG

    sample_weight = JMDS

    if cfg.domain_tgt == 'visda':
        return aff, sample_weight, aacc/100
    return aff, sample_weight, accuracy

def KLLoss(input_, target_, coeff, cfg):
    softmax = nn.Softmax(dim=1)(input_)
    kl_loss = (- target_ * torch.log(softmax + cfg.COWA_EPSILON2)).sum(dim=1)
    kl_loss *= coeff
    return kl_loss.mean(dim=0)

def mixup(x, c_batch, t_batch, model, cfg):
    # weight mixup
    if cfg.COWA_ALPHA==0:
        outputs = model(x)
        return KLLoss(outputs, t_batch, c_batch, cfg)
    lam = (torch.from_numpy(np.random.beta(cfg.COWA_ALPHA, cfg.COWA_ALPHA, [len(x)]))).float().cuda()
    t_batch = t_batch.cpu()
    t_batch = torch.eye(cfg.num_classes)[t_batch.argmax(dim=1)].cuda()
    shuffle_idx = torch.randperm(len(x))
    mixed_x = (lam * x.permute(1,2,3,0) + (1 - lam) * x[shuffle_idx].permute(1,2,3,0)).permute(3,0,1,2)
    mixed_c = lam * c_batch + (1 - lam) * c_batch[shuffle_idx]
    mixed_t = (lam * t_batch.permute(1,0) + (1 - lam) * t_batch[shuffle_idx].permute(1,0)).permute(1,0)
    mixed_x, mixed_c, mixed_t = map(torch.autograd.Variable, (mixed_x, mixed_c, mixed_t))
    mixed_outputs = model(mixed_x)
    return KLLoss(mixed_outputs, mixed_t, mixed_c, cfg)


def CoWA(config, dataloader_train_srgt, dataloader_val, dataloader_test, 
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
    
    # set the augmentation
    resize_size = 64  # 256
    crop_size = 64  # 224
    augment1 = transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
    ])
    
    # set up the training parameters
    max_iter = config.surrogate_epochs * len(trainloader_ta)
    interval_iter = max_iter // config.surrogate_interval
    if interval_iter == 0:
        interval_iter = 1
    
    iter_num = 0
    epoch = -1
    epochs = []
    accuracies = []
    
    # test the model before training
    model.eval()
    with torch.no_grad():
        # Compute JMDS score at offline & evaluation
        soft_pseudo_label, coeff, accuracy = evaluation(
            trainloader_te, model, config, epoch)
        epochs.append(epoch)
        accuracies.append(np.round(accuracy*100, 2))
    
    
    # start training
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
        inputs_test = inputs_test.cuda()

        # Get the soft pseudo label
        iter_num += 1
        lr_scheduler(config, optimizer, iter_num=iter_num, max_iter=max_iter)
        pred = soft_pseudo_label[tar_idx]
        pred_label = pred.argmax(dim=1)
        
        
        coeff, pred = map(torch.autograd.Variable, (coeff, pred))
        images1 = torch.autograd.Variable(augment1(inputs_test))
        images1 = images1.cuda()
        coeff = coeff.cuda()
        pred = pred.cuda()
        pred_label = pred_label.cuda()
        
        CoWA_loss = mixup(images1, coeff[tar_idx], pred, model, config)
        
        # For warm up the start.
        if iter_num < config.COWA_WARM * interval_iter + 1:
            CoWA_loss *= 1e-6
            
        optimizer.zero_grad()
        CoWA_loss.backward()
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
                'loss_ft': CoWA_loss.item(),
                'Acc_ft_src': acc1s[0]}

            # print validation
            print('[SFDA-CoWA] | epoch %03d | iter %03d | train_loss: %.3f | src_val_acc1: %.1f, tgt_val_acc1: ' %
                (epoch, iter_num, CoWA_loss.item(), acc1s[0]), end='')
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
