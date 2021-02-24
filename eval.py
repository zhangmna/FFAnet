import os
import sys
import cv2
import argparse
import numpy as np
import math
import torch
from torch import nn
from torch.nn import MSELoss
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import settings
from dataset import TestDataset
from model import RESCAN 
from cal_ssim import SSIM

logger = settings.logger
torch.cuda.manual_seed_all(66)
torch.manual_seed(66)
# torch.cuda.set_device(settings.device_id)


def ensure_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
def PSNR(img1, img2):
    b,_,_,_=img1.shape
    #mse=0
    #for i in range(b):
    img1=np.clip(img1,0,255)
    img2=np.clip(img2,0,255)
    mse = np.mean((img1/ 255. - img2/ 255.) ** 2)#+mse
    if mse == 0:
        return 100
    #mse=mse/b
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))        
class Session:
    def __init__(self):
        self.log_dir = settings.log_dir
        self.model_dir = settings.model_dir
        ensure_dir(settings.log_dir)
        ensure_dir(settings.model_dir)
        logger.info('set log dir as %s' % settings.log_dir)
        logger.info('set model dir as %s' % settings.model_dir)

        # self.net = RESCAN().cuda()
        if len(settings.device_id) >1:
            self.net = nn.DataParallel(RESCAN()).cuda()
        else:
            self.net = RESCAN().cuda() 

        self.crit = MSELoss().cuda()
        self.ssim = SSIM().cuda()
        self.dataloaders = {}

    def get_dataloader(self, dataset_name):
        dataset = TestDataset(dataset_name)
        if not dataset_name in self.dataloaders:
            self.dataloaders[dataset_name] = \
                    DataLoader(dataset, batch_size=1, 
                            shuffle=False, num_workers=1, drop_last=False)
        return self.dataloaders[dataset_name]

    def load_checkpoints(self, name):
        ckp_path = os.path.join(self.model_dir, name)
        try:
            obj = torch.load(ckp_path)
            logger.info('Load checkpoint %s' % ckp_path)
        except FileNotFoundError:
            logger.info('No checkpoint %s!!' % ckp_path)
            return
        self.net.load_state_dict(obj['net'])

    def inf_batch(self, name, batch):
        O, B = batch['O'].cuda(), batch['B'].cuda()
        O, B = Variable(O, requires_grad=False), Variable(B, requires_grad=False)
        R = O - B

        with torch.no_grad():
            O_Rs = self.net(O)
        loss_list = [self.crit(O_R, B) for O_R in O_Rs]
        ssim_list = [self.ssim(O_R, B) for O_R in O_Rs]
        psnr=PSNR(O_Rs[0].data.cpu().numpy()*255, B.data.cpu().numpy()*255)
        losses = {
            'loss%d' % i: loss.item()
            for i, loss in enumerate(loss_list)
        }
        ssimes = {
            'ssim%d' % i: ssim.item()
            for i, ssim in enumerate(ssim_list)
        }
        losses.update(ssimes)

        return losses,psnr


def run_test(ckp_name):
    sess = Session()
    sess.net.eval()
    sess.load_checkpoints(ckp_name)
    dt = sess.get_dataloader('test')
    psnr_all=0
    all_num = 0
    all_losses = {}
    for i, batch in enumerate(dt):
        losses,psnr= sess.inf_batch('test', batch)
        psnr_all=psnr_all+psnr
        batch_size = batch['O'].size(0)
        all_num += batch_size
        for key, val in losses.items():
            if i == 0:
                all_losses[key] = 0.
            all_losses[key] += val * batch_size
            logger.info('batch %d mse %s: %f' % (i, key, val))

    for key, val in all_losses.items():
        logger.info('total mse %s: %f' % (key, val / all_num))
    #psnr=sum(psnr_all)
    #print(psnr)
    print('psnr_ll:%8f'%(psnr_all/all_num))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='latest')

    args = parser.parse_args(sys.argv[1:])
    run_test(args.model)

