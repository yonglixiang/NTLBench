"""
Builds upon: https://github.com/ChiangE/Sophon
Corresponding paper: https://arxiv.org/pdf/2404.12699
"""

from torch import nn, optim
import numpy as np
import torch
import torch.nn.functional as F
from termcolor import cprint
import utils.evaluators
from utils.load_utils import load_bn, save_bn
import wandb
from torch.optim import lr_scheduler
import learn2learn as l2l
import copy


def accuracy(predictions, targets):
    with torch.no_grad():
        predictions = predictions.argmax(dim=1)
        targets = targets.argmax(dim=1)
    return (predictions == targets).sum().float() / targets.size(0)

def initialize(config, model):
    if 'vgg' in config.teacher_network:
        last_layer = model.classifier1[-1]
    elif 'vit' in config.teacher_network:
        last_layer = model.forward_head
    
    torch.nn.init.xavier_uniform_(last_layer.weight)
    if last_layer.bias is not None:
        torch.nn.init.zeros_(last_layer.bias)
    return model

def inverse_fast_adapt_multibatch(batches, learner, loss, shots, ways, config):
    # Adapt the model
    learner = initialize(config, learner)
    test_loss = 0
    test_accuracy = 0
    total_test = 0
    for index,batch in enumerate(batches):
        data, labels = batch
        data, labels = data.to(config.device).float(), labels.to(config.device).float()
        adaptation_indices = np.zeros(data.size(0), dtype=bool)
        # adaptation_indices[np.arange(shots*ways)] = True
        adaptation_indices[np.random.choice(np.arange(data.size(0)), shots*ways, replace=False)] = True
        evaluation_indices = torch.from_numpy(~adaptation_indices)
        adaptation_indices = torch.from_numpy(adaptation_indices)
        adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
        evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]
        current_test = evaluation_data.shape[0]
        # print(current_test)
        total_test += current_test
        adaptation_error = loss(learner(adaptation_data), adaptation_labels)
        if index == 0:
            current_grads = learner.adapt(adaptation_error,None) 
        else:
            last_grads = current_grads
            current_grads = learner.adapt(adaptation_error,last_grads) 
        predictions = learner(evaluation_data)
        evaluation_error = loss(1-predictions, evaluation_labels)  # inverse loss, aim to minimize
        evaluation_accuracy = accuracy(predictions, evaluation_labels)
        test_loss += evaluation_error*current_test
        test_accuracy += evaluation_accuracy*current_test
    return test_loss*1.0/total_test, test_accuracy*1.0/total_test

def kl_fast_adapt_multibatch(batches, learner, loss, shots, ways, config):
    # Adapt the model
    test_loss = 0
    test_accuracy = 0
    total_test = 0
    for index,batch in enumerate(batches):
        data, labels = batch
        data, labels = data.to(config.device).float(), labels.to(config.device).float()
        adaptation_indices = np.zeros(data.size(0), dtype=bool)
        # adaptation_indices[np.arange(shots*ways)] = True
        adaptation_indices[np.random.choice(np.arange(data.size(0)), shots*ways, replace=False)] = True
        evaluation_indices = torch.from_numpy(~adaptation_indices)
        adaptation_indices = torch.from_numpy(adaptation_indices)
        adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
        evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]
        current_test = evaluation_data.shape[0]
        # print(current_test)
        total_test += current_test
        adaptation_error = loss(learner(adaptation_data), adaptation_labels)
        if index == 0:
            current_grads = learner.adapt(adaptation_error,None) 
        else:
            last_grads = current_grads
            current_grads = learner.adapt(adaptation_error,last_grads) 
        # Evaluate the adapted model
        predictions = learner(evaluation_data)
        normalized_preds = torch.nn.functional.softmax(predictions, dim=1).cuda()
        target_preds = 0.1 * torch.ones((predictions.shape[0], predictions.shape[1])).cuda()
        evaluation_error = F.kl_div(torch.log(normalized_preds), target_preds, reduction='batchmean')  # KL uniform loss, aim to minimize
        evaluation_accuracy = accuracy(predictions, evaluation_labels)
        test_loss += evaluation_error*current_test
        test_accuracy += evaluation_accuracy*current_test
    return test_loss*1.0/total_test, test_accuracy*1.0/total_test 


def train_sophon(config, dataloaders, valloaders, testloaders, model, datasets_name):
    # setup evaluator
    if 'cmt' in config.domain_src:
        topk = (1, )  # 2-classification, use top1 acc
    else:
        topk = (1, 5)  # 10-classification, use top1 and top5 acc
    evaluators = [utils.evaluators.classification_evaluator(
        v, topk) for v in valloaders]
    evaluators_test = [utils.evaluators.classification_evaluator(
        v, topk) for v in testloaders]
    
    # init
    ways = config.SOPHON_ways
    shots = int(config.batch_size * 0.9 / ways)
    device = config.device
    
    src_iter = iter(dataloaders[0])
    tgt_iter = iter(dataloaders[1])
    
    maml = l2l.algorithms.MAML(model, lr=config.SOPHON_fast_lr, first_order=True)
    maml_opt = optim.Adam(maml.parameters(), config.SOPHON_alpha * config.SOPHON_lr)
    
    criterion = nn.CrossEntropyLoss(reduction='mean')
    natural_optimizer = optim.Adam(maml.parameters(), config.SOPHON_beta*config.SOPHON_lr)
    if 'bn' in config.teacher_network:
        mean0, var0 = save_bn(copy.deepcopy(model))
        
    # define maml training function
    funcs = {
        'inverse_loss':  inverse_fast_adapt_multibatch,
        'kl_loss': kl_fast_adapt_multibatch
    }
    fast_adapt_multibatch = funcs[config.SOPHON_maml_loss_func]

    # Pretrain
    for epoch in range(config.pretrain_epochs):
        model.train()
        
        # Fine-tuning suppression in target domain
        maml_losses = []
        maml_accs = []
        for ml in range(config.SOPHON_ml_loop):
            # Get particial target domain for this maml epoch
            batches = []
            for _ in range(config.SOPHON_adaptation_steps):
                try:
                    batches.append(next(tgt_iter))
                except StopIteration:
                    # reset iter
                    tgt_iter = iter(dataloaders[1])
                    batches.append(next(tgt_iter))
            
            # Apply maml adaptation and update via error
            maml_opt.zero_grad()
            learner = maml.clone()
            means, vars  = save_bn(model)
            evaluation_error, evaluation_accuracy = fast_adapt_multibatch(batches,
                                                                        learner,
                                                                        criterion,
                                                                        shots,
                                                                        ways,
                                                                        config)
            model.zero_grad()
            evaluation_error.backward()
            nn.utils.clip_grad_norm_(maml.module.parameters(), max_norm=0.5, norm_type=2)
            maml_opt.step()
            
            # Log
            maml_losses.append(evaluation_error.item())
            maml_accs.append(evaluation_accuracy.item())
            # Load bn
            model = load_bn(model, means, vars)
            

        
        # Normal training reinforcement in source domain
        nl_losses = []
        nl_accs = []
        for nl in range(config.SOPHON_nl_loop):
            torch.cuda.empty_cache()
            
            try:
                img, label = next(src_iter)
                img, label = img.to(device).float(), label.to(device).float()
            except StopIteration:
                src_iter = iter(dataloaders[0])
                img, label = next(src_iter)
                img, label = img.to(device).float(), label.to(device).float()
            
            natural_optimizer.zero_grad()
            loss = criterion(model(img), label) 
            loss.backward()
            natural_optimizer.step()
            
            # log
            nl_losses.append(loss.item())
            
        
        # Test and Log
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
        wandb_log_dict = {
            'pt_epoch': epoch,
            'pt_FTS_avg_loss': torch.mean(torch.tensor(maml_losses)).item(),
            'pt_FTS_avg_acc': torch.mean(torch.tensor(maml_accs)).item(),
            'pt_NTR_avg_loss': torch.mean(torch.tensor(nl_losses)).item(),
            'pt_FTS_last_loss': maml_losses[-1],
            'pt_FTS_last_acc': maml_accs[-1],
            'pt_NTR_last_loss': nl_losses[-1],
            'pt_Acc_val_src': acc1s[0],
        }
        
        print('[SOPHON Pretrain] | epoch %03d | FTS_loss: %.3f, FTS_acc: %.3f, NTR_loss: %.3f | src_val_acc1: %.1f, tgt_val_acc1: ' %
              (epoch, maml_losses[-1], maml_accs[-1], nl_losses[-1], acc1s[0]), end='')
        for dname, acc_tgt in zip(datasets_name[1:], acc1s[1:]):
            print(f'{dname}: {acc_tgt:.2f} ', end='')
            wandb_log_dict[f'pt_Acc_val_tgt_{dname}'] = acc_tgt
        tgt_mean = torch.mean(torch.tensor(acc1s[1:])).item()
        wandb_log_dict[f'pt_Acc_val_tgt_mean'] = tgt_mean
        print(f'| tgt_mean: {tgt_mean:.2f}')
        
        wandb.log(wandb_log_dict)
    
    # test
    test_acc1s, _ = utils.evaluators.eval_func(config, evaluators_test, model)

    wandb.run.summary['pt_Acc_test_src'] = test_acc1s[0]
    for dname, acc_tgt in zip(datasets_name[1:], test_acc1s[1:]):
        wandb.run.summary[f'pt_Acc_test_tgt_{dname}'] = acc_tgt
    test_acc1s_tgt_mean = torch.mean(torch.tensor(test_acc1s[1:])).item()
    wandb.run.summary[f'pt_Acc_test_tgt_mean'] = test_acc1s_tgt_mean
    
    return test_acc1s[0], test_acc1s_tgt_mean

