import argparse
import shutil
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import loaddata_nyu
import loaddata_kitti
# import loaddata_scannet
import util
import numpy as np

from models import modules, net, resnet

import pdb
import copy
import matplotlib
import matplotlib.image
matplotlib.rcParams['image.cmap'] = 'jet'


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
    checkpoint = torch.load("./runs/KN.pth.tar")
    model.load_state_dict(checkpoint['state_dict'])
    
    print('Number of G parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    cudnn.benchmark = True

    batch_size = 1
    test_loader_nyu = loaddata_nyu.getTestingData(batch_size)
    test_loader_kitti = loaddata_kitti.getTestingData(batch_size)
    # test_loader_scans = loaddata_scannet.getTestingData(batch_size)

    ###########specify task order
    test(test_loader_kitti, model, task=0)
    test(test_loader_nyu, model, task=1)
    # test(test_loader_scans, net_t, task=2)
        
def test(test_loader, net, task):
    net.eval()

    totalNumber = 0

    errorSum = {'MSE': 0, 'RMSE': 0, 'ABS_REL': 0, 'LG10': 0,
                'MAE': 0,  'DELTA1': 0, 'DELTA2': 0, 'DELTA3': 0}

    with torch.no_grad():
        for i, sample_batched in enumerate(test_loader):
            image, depth = sample_batched['image'], sample_batched['depth']

            depth = depth.cuda()
            image = image.cuda()

            out, _ = net(image)
            output, um = out[task][0], out[task][1]
            output = torch.nn.functional.upsample(output, size=[depth.size(2),depth.size(3)], mode='bilinear', align_corners=True)

            mask = (depth > 0)
            depth = depth[mask]
            output = output[mask]
            
            batchSize = depth.size(0)
            totalNumber = totalNumber + batchSize
            errors = util.evaluateError(output, depth)
            errorSum = util.addErrors(errorSum, errors, batchSize)
            averageError = util.averageErrors(errorSum, totalNumber)

        print(averageError)



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
