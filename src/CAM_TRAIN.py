import datetime
from distutils.version import LooseVersion
import math
import os
import os.path as osp
import shutil

import numpy as np
import matplotlib.pyplot as plt
#import skimage.io
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import tqdm 
import sys
#import torchfcn

class CAM_Trainer(object):

    def __init__(self, cuda, model, optimizer,
                 train_loader, val_loader, out, max_iter, n_class, 
                 size_average=False, interval_validate=None):
        self.cuda = cuda

        self.model = model
        self.optim = optimizer

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.timestamp_start = \
            datetime.datetime.now()
        self.size_average = size_average

        if interval_validate is None:
            self.interval_validate = len(self.train_loader)
        else:
            self.interval_validate = interval_validate

        self.out = out
        if not osp.exists(self.out):
            os.makedirs(self.out)

        self.log_headers = [
            'epoch',
            'iteration',
            'train/loss',
            'valid/loss',
            'elapsed_time',
        ]
        if not osp.exists(osp.join(self.out, 'log.csv')):
            with open(osp.join(self.out, 'log.csv'), 'w') as f:
                f.write(','.join(self.log_headers) + '\n')

        self.epoch = 0
        self.loss = 0
        self.iteration = 0
        self.max_iter = max_iter
        self.best_loss = 1000000
        self.n_class = n_class
        
    def validate(self):
        training = self.model.training
        self.model.eval()

        n_class = self.n_class 

        val_loss = 0
        visualizations = []
        label_trues, label_preds = [], []
        for batch_idx, (data, target) in tqdm.notebook.tqdm(
                enumerate(self.val_loader), total=len(self.val_loader),
                desc='Valid iteration=%d' % self.iteration, leave=False):
            
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            if target.cpu().numpy()[0, 0] == 1:
                continue
            data, target = Variable(data), Variable(target).to(dtype=torch.long) # target data type 이슈
            with torch.no_grad():
                score, feature, sigmoid_score = self.model(data)
            loss = self.loss_fcn(score, target)
            
            loss_data = loss.data.item()
            if np.isnan(loss_data):
                raise ValueError('loss is nan while validating')
            val_loss += loss_data / len(data)

            imgs = data.data.cpu()
            lbl_pred = sigmoid_score.data.cpu().numpy()
            lbl_true = target.data.cpu().numpy()
            
            print (lbl_true[0])
            print (lbl_pred[0])
            plt.figure(figsize=(10, 6))
            plt.plot(lbl_true[0], label='GT')
            plt.plot(lbl_pred[0], label='Pred')
            plt.legend()
            plt.show()
            
            for img, lt, lp in zip(imgs, lbl_true, lbl_pred):
                label_trues.append(lt)
                label_preds.append(lp)
            break    


        val_loss /= len(self.val_loader)

        with open(osp.join(self.out, 'log.csv'), 'a') as f:
            elapsed_time = (
                datetime.datetime.now() -
                self.timestamp_start).total_seconds()
            log = [self.epoch, self.iteration] + [''] * 5 + \
                  [val_loss] + [elapsed_time]
            log = map(str, log)
            f.write(','.join(log) + '\n')

        
        is_best = val_loss < self.best_loss
        if is_best:
            self.best_loss = val_loss
        torch.save({
            'epoch': self.epoch,
            'iteration': self.iteration,
            'arch': self.model.__class__.__name__,
            'optim_state_dict': self.optim.state_dict(),
            'model_state_dict': self.model.state_dict(),
            'best_loss': self.best_loss,
        }, osp.join(self.out, 'checkpoint.pth.tar'))
#         if is_best:
        shutil.copy(osp.join(self.out, 'checkpoint.pth.tar'),
                        osp.join(self.out, 'model_best.pth.tar'))

        if training:
            self.model.train()

    def train_epoch(self):
        self.model.train()
        iteration=0
        n_class = self.n_class
            # for jupyter notebook (tqdm.notebook.tqdm)
        for batch_idx, (data, target) in tqdm.notebook.tqdm(
                enumerate(self.train_loader), total=len(self.train_loader),
                desc='Train epoch={}'.format(self.epoch), leave=False):
#             if np.random.randint(10)>3:
#                 if target.cpu().numpy()[0, 0] == 1:
#                     continue
            #teration = batch_idx + self.epoch * len(self.train_loader)
            #f self.iteration != 0 and (iteration - 1) != self.iteration:
            #   continue  # for resuming
            self.iteration = iteration
        
            if self.iteration % self.interval_validate == 0:
                self.validate()

            #assert self.model.training

            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target).to(dtype=torch.long) # target data type 이슈
            self.optim.zero_grad()
            score, feature, _ = self.model(data)
            loss = self.loss_fcn(score, target)
#             loss /= len(data)
#             loss = loss
            loss_data = loss.data.item()
            sys.stdout.write('\r iteration : {0}, loss : {1:>20s}'.format(iteration, str(np.round(loss_data, decimals=4))))
            if np.isnan(loss_data):
                raise ValueError('loss is nan while training')
            loss.backward()
            self.optim.step()

            metrics = []
            lbl_pred = score.data.cpu().numpy()
            lbl_true = target.data.cpu().numpy()
            iteration+=1

            with open(osp.join(self.out, 'log.csv'), 'a') as f:
                elapsed_time = (
                    datetime.datetime.now() -
                    self.timestamp_start).total_seconds()
                log = [self.epoch, self.iteration] + [loss_data] + \
                    [''] * 5 + [elapsed_time]
                log = map(str, log)
                f.write(','.join(log) + '\n')

            if self.iteration >= self.max_iter:
                break

    def train(self):
        max_epoch = int(math.ceil(1. * self.max_iter / len(self.train_loader)))

        for epoch in tqdm.notebook.trange(self.epoch, max_epoch,
                                 desc='Train'):
            self.epoch = epoch
            self.train_epoch()
            if self.iteration >= self.max_iter:
                break
                
                
    # utils

    def loss_fcn(self, input, target):
        n,_ = input.shape
#weight = torch.FloatTensor([0.1]+[0.9]*(self.n_class-1)).cuda(), reduction = 'none')
        loss_all = torch.nn.MultiLabelSoftMarginLoss(reduction = 'mean')(input, target)
        loss_target = torch.nn.MultiLabelSoftMarginLoss(reduction = 'mean')(input*target, target)
#         print (loss)
        
        return loss_all+loss_target