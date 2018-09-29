import numpy as np
import os, time, copy, random
from glob import glob
import sys
from torchvision import models, transforms, datasets
import torch.optim as optim
import torch.nn as nn
import torch
from Layer.Loss import ProxyStatic
from Utils.Reader import ImageReader
from Utils.Utils import to_one_hot
from Utils import logging
import losses
import tqdm

PHASE = ['tra','val']


##################################################
# step -1: Predefined function
##################################################
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    temp = target.view(1, -1).expand_as(pred)
    temp = temp.cuda()
    correct = pred.eq(temp)

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class learn():
    def __init__(self, dst, RGBmean, RGBstdv, data_dict, inter_distance = 4,loss = 'softmax',gpuid = 'cuda:0',
                 num_epochs=200, init_lr=0.0001, decay=0.1, batch_size=64,
                 imgsize=256, num_workers=16, print_freq = 10, save_step = 10,
                 salency = 'none', scale = 128 , pool_type = 'max_avg'):
        self.dst = dst # save dir
        self.gpuid = gpuid
        self.loss = loss
            
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.decay_time = [False,False]
        self.init_lr = init_lr
        self.decay_rate = decay
        self.num_epochs = num_epochs

        self.data_dict = data_dict
        
        self.imgsize = imgsize
        self.RGBmean = RGBmean
        self.RGBstdv = RGBstdv
        
        self.record = []
        self.epoch = 0
        self.print_freq = print_freq
        self.save_step = save_step
        self.loss = loss
        if not self.setsys(): print('system error'); return
        self.losses = AverageMeter()
        self.top1 = AverageMeter()
        self.top5 = AverageMeter()
        #sys.stdout = logging.Logger(os.path.join(dst, 'log.txt'))
        print('init_lr : {}'.format(init_lr))
        print('image size: {}'.format(imgsize))
        print('batch size: {}'.format(batch_size))
        print('num workers: {}'.format(num_workers))

        ##model parameter
        self.saliency = salency
        self.scale = scale
        self.pool_type = pool_type
        self.inter_distance = inter_distance


    def run(self):
        self.loadData()
        self.setModel()
        self.opt(self.num_epochs)
        return
    

    ##################################################
    # step 0: System check and predefine function
    ##################################################
    def setsys(self):
        if not torch.cuda.is_available(): print('No GPU detected'); return False
        if not os.path.exists(self.dst): os.makedirs(self.dst)
        self.device = torch.device(self.gpuid)
        return True



    
    ##################################################
    # step 1: Loading Data
    ##################################################
    def loadData(self):
        
        self.data_transforms = transforms.Compose([transforms.Resize(int(self.imgsize*1.1)),
                                                   transforms.RandomRotation(10),
                                                   transforms.RandomCrop(self.imgsize),
                                                   transforms.RandomHorizontalFlip(),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize(self.RGBmean, self.RGBstdv)])
        

        self.dsets = ImageReader(self.data_dict, self.data_transforms)
        self.classSize = len(self.data_dict)
        self.class_label = torch.Tensor(np.array(range(self.classSize)))
        print('output size: {}'.format(self.classSize))

        return
    
    ##################################################
    # step 2: Set Model
    ##################################################
    def setModel(self):
        print('Setting model')
        from models import resnet_50
        self.model = resnet_50.resnet_50(pretrained=True,num_class =self.classSize,
                                         saliency=self.saliency,pool_type = self.pool_type,
                                         is_train = True,scale = self.scale)
        self.model = self.model.to(self.gpuid)
        self.criterion = losses.create(self.loss).to(self.gpuid)
        self.center_criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.init_lr,
                              momentum=0.9, weight_decay=0.00005)
        # self.optimizer = optim.SGD(self.model.parameters(), lr=self.init_lr, momentum=0.9)
        return
    
    def lr_scheduler(self, epoch):
        if epoch>=0.5*self.num_epochs and not self.decay_time[0]: 
            self.decay_time[0] = True
            lr = self.init_lr*self.decay_rate
            print('LR is set to {}'.format(lr))
            for param_group in self.optimizer.param_groups: param_group['lr'] = lr
        if epoch>=0.8*self.num_epochs and not self.decay_time[1]: 
            self.decay_time[1] = True
            lr = self.init_lr*self.decay_rate*self.decay_rate
            print('LR is set to {}'.format(lr))
            for param_group in self.optimizer.param_groups: param_group['lr'] = lr
        return
            
    ##################################################
    # step 3: Learning
    ##################################################
    def tra(self):
        # Set model to training mode
        self.model.train(True)
            
        dataLoader = torch.utils.data.DataLoader(self.dsets, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

        
        # iterate batch
        for i,data in enumerate(dataLoader):
            self.optimizer.zero_grad()
            
            with torch.set_grad_enabled(True):
                inputs_bt, labels_bt = data # <FloatTensor> <LongTensor>
                #labels_bt = labels_bt.to(torch.long)
                inputs_var = torch.autograd.Variable(inputs_bt).cuda()
                labels_var = torch.autograd.Variable(labels_bt).cuda()
                center_labels_var = torch.autograd.Variable(self.class_label.to(torch.long)).cuda()

                fvec, class_weight = self.model(inputs_var)
                #on_hot vector
                labels_var_one_hot = to_one_hot(labels_var,n_dims=self.classSize)
                # inter_class_distance
                fvec = fvec - self.inter_distance * labels_var_one_hot.cuda()
                #intra_class_distance
                loss = self.criterion(fvec, labels_var)
                center_loss = self.criterion(torch.mm(class_weight,torch.t(class_weight)),center_labels_var)
                #center_loss = self.center_criterion(torch.mm(class_weight,torch.t(class_weight)),torch.eye(self.classSize).cuda())
                total_loss = 0.5 * center_loss + loss
                prec1, prec5 = accuracy(fvec.data, labels_bt, topk=(1, 5))
                self.losses.update(total_loss.data[0], inputs_bt.size(0))
                self.top1.update(prec1[0], inputs_bt.size(0))
                self.top5.update(prec5[0], inputs_bt.size(0))
                total_loss.backward()
                self.optimizer.step()
            if i % self.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    self.epoch, i, len(dataLoader), loss=self.losses, top1=self.top1, top5=self.top5))
        return
    def opt(self, num_epochs):
        # recording time and epoch acc and best result
        for epoch in range(num_epochs):
            self.epoch =epoch
            self.lr_scheduler(epoch)
            
            self.tra()
            # deep copy the model
            if epoch % self.save_step == 0:
                torch.save(self.model, os.path.join(self.dst, '%d_model.pth' % epoch))
    
    

    