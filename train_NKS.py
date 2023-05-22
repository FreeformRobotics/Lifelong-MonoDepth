import argparse
import shutil
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import loaddata_scannet
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
parser.add_argument('--num_tasks', '--number of tasks/domains', default=3, type=float)



def main():
    global args
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cudnn.benchmark = True

    encoder = modules.E_resnet(resnet.resnet34(pretrained=True))
    backbone = net.backbone(encoder, num_features=512, block_channel=[64, 128, 256, 512])

    #############load the old model
    model_ = net.model_ll(backbone,num_tasks=2, block_channel=[64, 128, 256, 512])
    checkpoint = torch.load("./runs/NK.pth.tar")
    model_.load_state_dict(checkpoint['state_dict'])

    ###########define the new model
    model = net.model_ll(copy.deepcopy(model_),num_tasks=args.num_tasks, block_channel=[64, 128, 256, 512])
    model_.to(device)
    model.to(device)
    
    print('Number of G parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
    
    batch_size = 4

    cudnn.benchmark = True

    pdb.set_trace()
    train_loader_scans = loaddata_scannet.getTrainingData(batch_size)
    replay_kitti = loaddata_kitti2.getTrainingData3(batch_size, csv_file='./datasets/replay_kitti.csv')
    replay_nyu = loaddata_nyu.getTrainingData(batch_size, csv_file='./datasets/replay_nyu.csv')

    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer_T, epoch)
        train(model, model_, train_loader_scans, replay_nyu, replay_kitti, optimizer)
    
    save_checkpoint({ 'state_dict': model.state_dict()},filename='NKS.pth.tar') 

def train(net, net_, train_loader_scans, replay_nyu, replay_kitti, optimizer):
    net.train()
    net_.eval()
    replay_nyu_iter = iter(replay_nyu)
    replay_kitti_iter = iter(replay_kitti)

    for i, sample_batched in enumerate(train_loader_scans):
        image = sample_batched['image'].cuda()
        depth = sample_batched['depth'].cuda()
                
        optimizer.zero_grad()
        out, _ = net(image)
        out_, _ = net_(image)
        
        ##############calculate distillation loss for old tasks
        pred_task1, um_task1 = out[0][0], out[0][1]
        pred_task1_, um_task1_ = out_[0][0], out_[0][1]

        pred_task2, um_task2 = out[1][0], out[1][1]
        pred_task2_, um_task2_ = out_[1][0], out_[1][1]
        
        loss_task1 = ((pred_task1/pred_task1_.median()-pred_task1_/pred_task1_.median()).abs() + (um_task1/um_task1_.median()-um_task1_/um_task1_.median()).abs()).mean()
        loss_task2 = ((pred_task2/pred_task2_.median()-pred_task2_/pred_task2_.median()).abs() + (um_task2/um_task2_.median()-um_task2_/um_task2_.median()).abs()).mean()

        ##############calculate new loss for the new task
        pred_task3, um_task3 = out[2][0], out[2][1]
                
        pred_task3 = torch.nn.functional.upsample(pred_task3, size=[depth.size(2),depth.size(3)], mode='bilinear', align_corners=True)
        um_task3 = torch.nn.functional.upsample(um_task3, size=[depth.size(2),depth.size(3)], mode='bilinear', align_corners=True)
        
        mask = (depth > 0)
        pred_task3 = pred_task3[mask]
        depth = depth[mask]
        um_task3 = um_task3[mask]
        
        loss_task3 = (torch.exp(-um_task3) * (pred_task3/depth.median()-depth/depth.median())**2 + 2*um_task3).mean()

        try:
            #####################################################replay nyu
            replay_nyu_batch =  replay_nyu_iter.next()
            replay_nyu_img, replay_nyu_depth = replay_nyu_batch['image'].cuda(), replay_nyu_batch['depth'].cuda() 
            replay_out_nyu, _ = net(replay_nyu_img)
           
            replay_pred_task1, replay_um_task1 = replay_out_nyu[0][0], replay_out_nyu[0][1]                        
            replay_pred_task1 = torch.nn.functional.upsample(replay_pred_task1, size=[replay_nyu_depth.size(2),replay_nyu_depth.size(3)], mode='bilinear', align_corners=True)
            replay_um_task1 = torch.nn.functional.upsample(replay_um_task1, size=[replay_nyu_depth.size(2),replay_nyu_depth.size(3)], mode='bilinear', align_corners=True)
            
            mask2 = (replay_nyu_depth > 0)
            replay_pred_task1 = replay_pred_task1[mask2]
            replay_nyu_depth = replay_nyu_depth[mask2]
            replay_um_task1 = replay_um_task1[mask2]
        
            loss_replay_task1 = (torch.exp(-replay_um_task1) * (replay_pred_task1/replay_nyu_depth.median()-replay_nyu_depth/replay_nyu_depth.median())**2 + 2*replay_um_task1).mean()

            #####################################################replay kitti
            replay_kitti_batch =  replay_kitti_iter.next()
            replay_kitti_img, replay_kitti_depth = replay_kitti_batch['image'].cuda(), replay_kitti_batch['depth'].cuda() 
            replay_out_kitti, _ = net(replay_kitti_img)
           
            replay_pred_task2, replay_um_task2 = replay_out_kitti[1][0], replay_out_kitti[1][1]                        
            replay_pred_task2 = torch.nn.functional.upsample(replay_pred_task2, size=[replay_kitti_depth.size(2),replay_kitti_depth.size(3)], mode='bilinear', align_corners=True)
            replay_um_task2 = torch.nn.functional.upsample(replay_um_task2, size=[replay_kitti_depth.size(2),replay_kitti_depth.size(3)], mode='bilinear', align_corners=True)
            
            mask3 = (replay_kitti_depth > 0)
            replay_pred_task2 = replay_pred_task2[mask3]
            replay_kitti_depth = replay_kitti_depth[mask3]
            replay_um_task2 = replay_um_task2[mask3]
        
            loss_replay_task2 = (torch.exp(-replay_um_task2) * (replay_pred_task2/replay_kitti_depth.median()-replay_kitti_depth/replay_kitti_depth.median())**2 + 2*replay_um_task2).mean()

            loss_d = loss_task1 * 10 + loss_task2 * 100 + loss_replay_task1 * 10 + loss_replay_task2 * 100  + loss_task3
        except StopIteration:
            pass
            loss_d = loss_task1 * 10 + loss_task2 * 10 + loss_task3
        
        loss_d.backward()
        optimizer.step()

        if i % 2000 == 0:
            print(i, loss_task1.item(), loss_task2.item(),  loss_task3.item())
            print('mae',(pred_task3-depth).abs().mean().item())
            print(i,um_task3.mean().item())

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
