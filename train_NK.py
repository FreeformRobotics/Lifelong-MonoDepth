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
    
    #############load the old model
    model_ = net.model_ll(backbone,num_tasks=1, block_channel=[64, 128, 256, 512])
    checkpoint = torch.load("./runs/N.pth.tar")
    model_.load_state_dict(checkpoint['state_dict'])
    
    ###########define the new model
    model = net.model_ll(copy.deepcopy(model_),num_tasks=args.num_tasks, block_channel=[64, 128, 256, 512])
    model_.to(device)
    model.to(device)
    
    print('Number of G parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
    
    batch_size = 8

    ##############load new training data on the new domain
    train_loader_kitti = loaddata_kitti.getTrainingData(batch_size)
    ##############load 500 replay data on the old domain
    replay_nyu = loaddata_nyu.getTrainingData(batch_size, csv_file='./datasets/replay_nyu.csv')

    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)
        train(model, model_, train_loader_kitti, replay_nyu, optimizer)
    
    save_checkpoint({ 'state_dict': model.state_dict()},filename='NK.pth.tar')

def train(net, net_, train_loader_kitti, replay_nyu, optimizer):
    net.train()
    net_.eval()
    replay_nyu_iter = iter(replay_nyu)
    
    for i, sample_batched in enumerate(train_loader_kitti):
        image = sample_batched['image'].cuda()
        depth = sample_batched['depth'].cuda()
        
        optimizer.zero_grad()
        out, _ = net(image)
        out_, _ = net_(image)
        
        ##############calculate distillation loss for task1
        pred_task1, um_task1 = out[0][0], out[0][1]
        pred_task1_, um_task1_ = out_[0][0], out_[0][1]
        
        loss_task1 = ((pred_task1/pred_task1_.median()-pred_task1_/pred_task1_.median()).abs() + (um_task1/um_task1_.median()-um_task1_/um_task1_.median()).abs()).mean()

        ##############calculate new loss for task2
        pred_task2, um_task2 = out[1][0], out[1][1]
                
        pred_task2 = torch.nn.functional.upsample(pred_task2, size=[depth.size(2),depth.size(3)], mode='bilinear', align_corners=True)
        um_task2 = torch.nn.functional.upsample(um_task2, size=[depth.size(2),depth.size(3)], mode='bilinear', align_corners=True)
        
        mask = (depth > 0)
        pred_task2 = pred_task2[mask]
        depth = depth[mask]
        um_task2 = um_task2[mask]
        
        loss_task2 = (torch.exp(-um_task2) * (pred_task2/depth.median()-depth/depth.median())**2 + 2*um_task2).mean()        
        ##############replay loss
        try:
            replay_nyu_batch =  replay_nyu_iter.next()
            replay_nyu_img, replay_nyu_depth = replay_nyu_batch['image'].cuda(), replay_nyu_batch['depth'].cuda() 
            replay_out, _ = net(replay_nyu_img)
           
            replay_pred_task1, replay_um_task1 = replay_out[0][0], replay_out[0][1]                        
            replay_pred_task1 = torch.nn.functional.upsample(replay_pred_task1, size=[replay_nyu_depth.size(2),replay_nyu_depth.size(3)], mode='bilinear', align_corners=True)
            replay_um_task1 = torch.nn.functional.upsample(replay_um_task1, size=[replay_nyu_depth.size(2),replay_nyu_depth.size(3)], mode='bilinear', align_corners=True)
            
            mask2 = (replay_nyu_depth > 0)
            replay_pred_task1 = replay_pred_task1[mask2]
            replay_nyu_depth = replay_nyu_depth[mask2]
            replay_um_task1 = replay_um_task1[mask2]
        
            loss_replay_task1 = (torch.exp(-replay_um_task1) * (replay_pred_task1/replay_nyu_depth.median()-replay_nyu_depth/replay_nyu_depth.median())**2 + 2*replay_um_task1).mean()
            
            loss_d = loss_task1  * 10 +  loss_replay_task1 * 10 + loss_task2

        except StopIteration:
            pass
            
            loss_d = loss_task1  * 10  + loss_task2
            
        loss_d.backward()
        optimizer.step()

        if i % 2000 == 0:
            print(i, loss_task1.item(), loss_task2.item())
            print('mae',(pred_task2-depth).abs().mean().item())
            print(i,um_task2.mean().item())
    
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
