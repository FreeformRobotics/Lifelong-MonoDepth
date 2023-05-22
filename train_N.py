import argparse
import shutil
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import loaddata_nyu
import loaddata_kitti
import util
import numpy as np
from models import modules, net, resnet
import pdb
import copy
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description='Single-line LiDAR Completion')
parser.add_argument('--epochs', default=20, type=int, help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, help='weight decay (default: 1e-4)')
parser.add_argument('--num_tasks', '--number of tasks/domains', default=2, type=float)


def main():
    global args
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cudnn.benchmark = True

    encoder = modules.E_resnet(resnet.resnet34(pretrained=True))
    backbone = net.backbone(encoder, num_features=512, block_channel=[64, 128, 256, 512])
    
    #############define the model
    model = net.model_ll(backbone,num_tasks=1, block_channel=[64, 128, 256, 512])
    model.to(device)
    print('Number of G parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
    
    batch_size = 8

    train_loader_nyu = loaddata_nyu.getTrainingData(batch_size)
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)
        train(model, train_loader_nyu, optimizer)
    
    save_checkpoint({ 'state_dict': model.state_dict()},filename='N.pth.tar')

def train(net, train_loader_nyu, optimizer):
    net.train()

    for i, sample_batched in enumerate(train_loader_nyu):
        image = sample_batched['image'].cuda()
        depth = sample_batched['depth'].cuda()
        
        optimizer.zero_grad()
        out, _ = net(image)

        pred, um = out[0][0], out[0][1]
        pred = torch.nn.functional.upsample(pred, size=[depth.size(2),depth.size(3)], mode='bilinear', align_corners=True)
        um = torch.nn.functional.upsample(um, size=[depth.size(2),depth.size(3)], mode='bilinear', align_corners=True)
        
        mask = (depth > 0)
        pred = pred[mask]
        depth = depth[mask]
        um = um[mask]
        
        loss_d = (torch.exp(-um) * (pred/depth.median()-depth/depth.median())**2 + 2*um).mean()
        
        loss_d.backward()
        optimizer.step()

        if i % 2000 == 0:
            print(i, loss_d.item())
            print('mae',(pred-depth).abs().mean().item())
            print(i,um.mean().item())
 


def adjust_learning_rate(optimizer, epoch):
    lr = args.lr * (0.5 ** (epoch // 5))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class AverageMeter(object):

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


def save_checkpoint(state, filename):
    # """Saves checkpoint to disk"""
    directory = "runs/" 
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)

if __name__ == '__main__':
    main()
