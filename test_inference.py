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
parser.add_argument('--num_tasks', '--number of tasks/domains', default=2, type=float)

def main():
    global args
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    encoder = modules.E_resnet(resnet.resnet34(pretrained=True))
    backbone = net.backbone(encoder, num_features=512, block_channel=[64, 128, 256, 512])
    
    model = net.model_ll(backbone,num_tasks=args.num_tasks, block_channel=[64, 128, 256, 512])
    model.to(device)
    ############load the trained models named in learning order, e.g., KN means learn on KITTI first and NYU-v2 second.
    checkpoint = torch.load("./runs/NKS.pth.tar")
    model.load_state_dict(checkpoint['state_dict'])
    
    print('Number of G parameters: {}'.format(sum([p.data.nelement() for p in net_t.parameters()])))
    
    batch_size = 1

    cudnn.benchmark = True

    replay_kitti = loaddata_kitti.getTrainingData(batch_size, csv_file='./datasets/replay_kitti.csv')
    replay_nyu = loaddata_nyu.getTrainingData(batch_size, csv_file='./datasets/replay_nyu.csv')
    replay_scans = loaddata_scannet.getTrainingData(batch_size, csv_file='./datasets/replay_scannet.csv')

    test_loader_nyu = loaddata_nyu.getTestingData(batch_size)
    test_loader_kitti = loaddata_kitti.getTestingData(batch_size)
    test_loader_scans = loaddata_scannet.getTestingData(batch_size)

    feas_nyu = test_feas(replay_nyu, net_t, batch_size, task=0)  
    feas_kitti = test_feas(replay_kitti, net_t, batch_size, task=1)
    feas_scans = test_feas(replay_scans, net_t, batch_size, task=2)
    
    feas_nyu = feas_nyu.view(1,64,114,152)
    feas_kitti = feas_kitti.view(1,64,160,240)
    feas_scans = feas_scans.view(1,64,114,152)

    feas_nyu2 = torch.nn.functional.upsample(feas_nyu, size=[176,608], mode='bilinear', align_corners=True)
    feas_kitti2 = torch.nn.functional.upsample(feas_kitti, size=[176,608], mode='bilinear', align_corners=True)
    feas_kitti3 = torch.nn.functional.upsample(feas_kitti, size=[114,152], mode='bilinear', align_corners=True)
    feas_scans2 = torch.nn.functional.upsample(feas_scans, size=[176,608], mode='bilinear', align_corners=True)
    
    test(test_loader_nyu, net_t, feas_kitti3, feas_nyu, feas_scans, task=0)
    test(test_loader_kitti, net_t, feas_kitti2, feas_nyu2, feas_scans2, task=1)
    test(test_loader_scans, net_t, feas_kitti3, feas_nyu, feas_scans, task=2)



def test_feas(test_loader, net, batchsize, task):
    net.eval()
    if(task==1):        

        mean_feas= torch.zeros(batchsize,64,160,240).cuda()
    else:
        mean_feas= torch.zeros(batchsize,64,114,152).cuda()
    with torch.no_grad():
        for i, sample_batched in enumerate(test_loader):
            image, depth = sample_batched['image'], sample_batched['depth']

            depth = depth.cuda()
            image = image.cuda()
            
            out, features = net(image)
            mean_feas += features
            if(i==499):
                break

    mean_feas = mean_feas/499
    return mean_feas.mean(0)


def test(test_loader, net, feas_kitti, feas_nyu, feas_scans, task):
    net.eval()

    totalNumber = 0

    errorSum = {'MSE': 0, 'RMSE': 0, 'ABS_REL': 0, 'LG10': 0,
                'MAE': 0,  'DELTA1': 0, 'DELTA2': 0, 'DELTA3': 0}
    
    count1 = 0
    count2 = 0
    count3 = 0

    end = time.time()                
    with torch.no_grad():
        for i, sample_batched in enumerate(test_loader):
            image, depth = sample_batched['image'], sample_batched['depth']

            depth = depth.cuda()
            image = image.cuda()

            out, feas = net(image)

            d1 = ((feas-feas_nyu)**2).mean().data.cpu().numpy()       
            d2 = ((feas-feas_kitti)**2).mean().data.cpu().numpy()    
            d3 = ((feas-feas_scans)**2).mean().data.cpu().numpy()

            idex = np.argmin([d1,d2,d3])
  
            if(idex == 0):
                count1 += 1
            elif(idex == 1):
                count2 += 1
            elif(idex == 2):
                count3 += 1

            output, um = out[idex][0], out[idex][1]    
            output = torch.nn.functional.upsample(output, size=[depth.size(2),depth.size(3)], mode='bilinear', align_corners=True)

            mask = (depth > 0)
            depth = depth[mask]
            output = output[mask]
            
            batchSize = depth.size(0)
            totalNumber = totalNumber + batchSize
            errors = util.evaluateError(output, depth)
            errorSum = util.addErrors(errorSum, errors, batchSize)
            averageError = util.averageErrors(errorSum, totalNumber)

        end2 = time.time()
        total_time = (end2-end)
        average_time = total_time/totalNumber
        print(total_time,average_time)
        print(averageError)

        print('task',count1,count2,count3)



def denormalize(image_tensor, use_fp16=False):
    '''
    convert floats back to input
    '''
    if use_fp16:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float16)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float16)
    else:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[:, c] = torch.clamp(image_tensor[:, c] * s + m, 0, 1)

    return image_tensor

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




if __name__ == '__main__':
    main()
