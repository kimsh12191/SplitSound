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

class Trainer(object):

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
            'train/acc',
            'train/acc_cls',
            'train/mean_iu',
            'train/fwavacc',
            'valid/loss',
            'valid/acc',
            'valid/acc_cls',
            'valid/mean_iu',
            'valid/fwavacc',
            'elapsed_time',
        ]
        if not osp.exists(osp.join(self.out, 'log.csv')):
            with open(osp.join(self.out, 'log.csv'), 'w') as f:
                f.write(','.join(self.log_headers) + '\n')

        self.epoch = 0
        self.loss = 0
        self.iteration = 0
        self.max_iter = max_iter
        self.best_mean_iu = 0
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
            data, target = Variable(data), Variable(target).to(dtype=torch.long) # target data type 이슈
            with torch.no_grad():
                score = self.model(data)
            loss = self.cross_entropy2d(score, target,
                                  size_average=self.size_average)
            loss_data = loss.data.item()
            if np.isnan(loss_data):
                raise ValueError('loss is nan while validating')
            val_loss += loss_data / len(data)

            imgs = data.data.cpu()
            lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
            lbl_true = target.data.cpu()
            fig = plt.figure(figsize=(10, 12))
            fig.add_subplot(1, 3, 1)
            plt.imshow(imgs.numpy()[0, 0, :, :])
            fig.add_subplot(1, 3, 2)
            plt.imshow(lbl_pred[0, :, :])
            fig.add_subplot(1, 3, 3)
            plt.imshow(lbl_true.numpy()[0, :, :])
            plt.show()
            for img, lt, lp in zip(imgs, lbl_true, lbl_pred):
                label_trues.append(lt)
                label_preds.append(lp)
            break    
        metrics = self.label_accuracy_score(
            label_trues, label_preds, n_class)


        val_loss /= len(self.val_loader)

        with open(osp.join(self.out, 'log.csv'), 'a') as f:
            elapsed_time = (
                datetime.datetime.now() -
                self.timestamp_start).total_seconds()
            log = [self.epoch, self.iteration] + [''] * 5 + \
                  [val_loss] + list(metrics) + [elapsed_time]
            log = map(str, log)
            f.write(','.join(log) + '\n')

        mean_iu = metrics[2]
        is_best = mean_iu > self.best_mean_iu
        if is_best:
            self.best_mean_iu = mean_iu
        torch.save({
            'epoch': self.epoch,
            'iteration': self.iteration,
            'arch': self.model.__class__.__name__,
            'optim_state_dict': self.optim.state_dict(),
            'model_state_dict': self.model.state_dict(),
            'best_mean_iu': self.best_mean_iu,
        }, osp.join(self.out, 'checkpoint.pth.tar'))
        if is_best:
            shutil.copy(osp.join(self.out, 'checkpoint.pth.tar'),
                        osp.join(self.out, 'model_best.pth.tar'))

        if training:
            self.model.train()

    def train_epoch(self):
        self.model.train()
        
        n_class = self.n_class
            # for jupyter notebook (tqdm.notebook.tqdm)
        for batch_idx, (data, target) in tqdm.notebook.tqdm(
                enumerate(self.train_loader), total=len(self.train_loader),
                desc='Train epoch={}'.format(self.epoch), leave=False):
            iteration = batch_idx + self.epoch * len(self.train_loader)
            if self.iteration != 0 and (iteration - 1) != self.iteration:
                continue  # for resuming
            self.iteration = iteration
        
            if self.iteration % self.interval_validate == 0:
                self.validate()

            #assert self.model.training

            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target).to(dtype=torch.long) # target data type 이슈
            self.optim.zero_grad()
            score = self.model(data)
            
            loss = self.cross_entropy2d(score, target,
                                   size_average=self.size_average)
            loss /= len(data)
            loss_data = loss.data.item()
            sys.stdout.write('\r iteration : {0}, loss : {1:>20s}'.format(iteration, str(np.round(loss_data, decimals=4))))
            if np.isnan(loss_data):
                raise ValueError('loss is nan while training')
            loss.backward()
            self.optim.step()

            metrics = []
            lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
            lbl_true = target.data.cpu().numpy()
            acc, acc_cls, mean_iu, fwavacc = \
                self.label_accuracy_score(
                    lbl_true, lbl_pred, n_class=n_class)
            metrics.append((acc, acc_cls, mean_iu, fwavacc))
            metrics = np.mean(metrics, axis=0)

            with open(osp.join(self.out, 'log.csv'), 'a') as f:
                elapsed_time = (
                    datetime.datetime.now() -
                    self.timestamp_start).total_seconds()
                log = [self.epoch, self.iteration] + [loss_data] + \
                    metrics.tolist() + [''] * 5 + [elapsed_time]
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
    def cross_entropy2d(self, input, target, weight=None, size_average=True):
        # input: (n, c, h, w), 
        # target: (n, h, w), 0~n_class number
        n, c, h, w = input.size()
        # log_p: (n, c, h, w)

        # >=0.3
        log_p = F.log_softmax(input, dim=1)
        # log_p: (n*h*w, c)
        log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()

        log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0] 
        # target: (n*h*w,)
        mask = target >= 0
        target = target[mask]
        loss = F.nll_loss(log_p.view(-1, c), target.view(-1), weight=weight, reduction='sum') # The negative log likelihood loss.
        if size_average:
            loss /= mask.data.sum()
        return loss


    # utils for validation
    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * np.int32(label_true[mask]) +
            label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
        return hist


    def label_accuracy_score(self, label_trues, label_preds, n_class):
        """Returns accuracy score evaluation result.
          - overall accuracy
          - mean accuracy
          - mean IU
          - fwavacc
        """
        hist = np.zeros((n_class, n_class))
        for lt, lp in zip(label_trues, label_preds):
            hist += self._fast_hist(lt.flatten(), lp.flatten(), n_class)
        acc = np.diag(hist).sum() / hist.sum()
        with np.errstate(divide='ignore', invalid='ignore'):
            acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        with np.errstate(divide='ignore', invalid='ignore'):
            iu = np.diag(hist) / (
                hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
            )
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        return acc, acc_cls, mean_iu, fwavacc