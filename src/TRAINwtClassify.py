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

class FCNTrainerClassify(object):

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
            if np.random.randint(10)>5:
                if np.mean(target.cpu().numpy()[0, 0]) >= 1:
                    continue
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target).to(dtype=torch.long) # target data type 이슈
            with torch.no_grad():
                score, score_reconst, score_softmax = self.model(data)
#             loss = self.cross_entropy2d(score, target,
            class_loss, reconst_loss = self.loss_fcn(score, score_reconst, target, data,
                                  size_average=self.size_average)
            loss = class_loss+reconst_loss
            loss_data = loss.data.item()
            if np.isnan(loss_data):
                raise ValueError('loss is nan while validating')
#             val_loss += loss_data / len(data)

            imgs = data.data.cpu()
            lbl_pred = score_softmax.data.cpu().numpy()
            lbl_true = target.data.cpu()
            fig = plt.figure(figsize=(10, 12))
            n, c, w, h = lbl_pred.shape
            score = torch.nn.Sigmoid()(score)
            print (score[0])
            print (lbl_true[0, :, 0, 0])
            pos_loc = list(np.where(lbl_true[0, :, 0, 0]==1)[0])
            neg_loc = list(np.where(lbl_true[0, :, 0, 0]==0)[0])
            fig.add_subplot(3, len([pos_loc[0], neg_loc[0]])+1, 1)
            plt.imshow(imgs.numpy()[0, 0, :, :])
            for i, label_loc in enumerate([pos_loc[0], neg_loc[0]]):
                fig.add_subplot(3, len([pos_loc[0], neg_loc[0]])+1, i+2)
                plt.title(np.round(score[0, label_loc].cpu().numpy(), 4))
                plt.imshow(lbl_pred[0, label_loc, :, :])
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
        
        n_class = self.n_class
            # for jupyter notebook (tqdm.notebook.tqdm)
        for batch_idx, (data, target) in tqdm.notebook.tqdm(
                enumerate(self.train_loader), total=len(self.train_loader),
                desc='Train epoch={}'.format(self.epoch), leave=False):
#             if np.random.randint(10)>3:
#                 if np.mean(target.cpu().numpy()[0, 0]) >= 1:
#                     continue
#             iteration = batch_idx + self.epoch * len(self.train_loader)
#             if self.iteration != 0 and (iteration - 1) != self.iteration:
#                 continue  # for resuming
#             self.iteration = iteration
        
            if self.iteration % self.interval_validate == 0:
                self.validate()

            #assert self.model.training

            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target).to(dtype=torch.long) # target data type 이슈
            self.optim.zero_grad()
            score, score_reconst, score_softmax = self.model(data)
            
            class_loss, reconst_loss = self.loss_fcn(score, score_reconst, target, data,
                                   size_average=self.size_average)
#             reconst_loss = self.reconst_loss(score, data, target,
#                                    size_average=self.size_average)
            l2_regul = self.regularizer(self.model)
            loss = 0.01*l2_regul+class_loss+reconst_loss
            loss_data = loss.data.item()
            class_loss_data = class_loss.data.item()
            reconst_loss_data = reconst_loss.data.item()
            sys.stdout.write('\r iteration : {0}, class_loss : {1:>20s}, reconst_loss : {2:>20s}'.format(self.iteration, str(np.round(class_loss_data, decimals=4)), str(np.round(reconst_loss_data, decimals=4))))
            if np.isnan(loss_data):
                raise ValueError('loss is nan while training')
            loss.backward()
            self.optim.step()

            metrics = []
            lbl_pred = score.data.cpu().numpy()
            lbl_true = target.data.cpu().numpy()
            

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
            self.iteration += 1
    def train(self):
        max_epoch = int(math.ceil(1. * self.max_iter / len(self.train_loader)))
        

        for epoch in tqdm.notebook.trange(self.epoch, max_epoch,
                                 desc='Train'):
            self.epoch = epoch
            self.train_epoch()
            if self.iteration >= self.max_iter:
                break
                
                
    # utils
    def loss_fcn(self, input_class, input_reconst, target, target_reconst, size_average=True):
        # input data type 안맞으면 문제생김
        # expected dtype Double but got dtype Float
        n, c, h, w = input_reconst.size()
        target = target.transpose(1, 2).transpose(2, 3).contiguous()
        input_reconst = input_reconst.transpose(1, 2).transpose(2, 3).contiguous()
        input_class_reshape = input_class.view(n, -1, c).to(torch.float32)
        input_reconst_reshape = input_reconst.view(n, -1, c).to(torch.float32)
        target_reshape = target.view(n, -1, c).to(torch.float32)
        target_reconst = target_reconst.transpose(1, 2).transpose(2, 3).contiguous()
        target_reconst_reshape = target_reconst.view(n, -1, 1).to(torch.float32)
        loss = torch.nn.MultiLabelSoftMarginLoss(reduction = 'sum')((input_class).to(torch.float32), target_reshape[:, 0, :].view(n, c))
        lossKL = torch.nn.KLDivLoss(reduction = 'batchmean')((torch.nn.Sigmoid()(input_reconst_reshape)).log(), target_reshape)
#         lossKL = torch.nn.MultiLabelSoftMarginLoss(reduction = 'sum')(input_reconst_reshape.view(-1, c), target_reshape.view(-1, c))
        return loss , lossKL/(320*160)


    def regularizer(self, model):
        l2_norm = 0
        for param in model.parameters():
            l2_norm += torch.sum(torch.pow(param, 2))
        return l2_norm