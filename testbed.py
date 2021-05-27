
import os
import sys
import time
import torch
import torch.utils.data as uData
from networks import UNetD
from datasets.DenoisingDatasets import LDCTTrain
from math import ceil
from utils import *
import torchvision.utils as vutils
import warnings
import matplotlib.pyplot as plt
from scipy import fftpack

from scipy.optimize import curve_fit

# filter warnings
warnings.simplefilter('ignore', Warning, lineno=0)

# default dtype
torch.set_default_dtype(torch.float32)

_C = 3
_modes = ['train']


def func(x, a1, b1, a2, b2, c):
    return a1 * x**b1 + a2 * x**b2 + c

def fit_dct_curve():
    curve_1p = np.load('figs/03301_diag.npy')
    xdata = np.linspace(0, 1, curve_1p.shape[0])
    popt, pcov = curve_fit(func, xdata, np.log(curve_1p))
    plt.plot(xdata, curve_1p, 'b-')
    plt.plot(xdata, np.exp(func(xdata, *popt)), 'r-')
    plt.yscale("log")
    plt.show()
    print(*popt)
    np.save('figs/coeffs_03301.npy', popt)





def from4kto400(input):
    out = input * 4096 - 1024
    return ((out + 160)/400).clamp_(0.0, 1.0)

def experiments(datasets):
    batch_size = {'train':16}
    data_loader = {phase:uData.DataLoader(datasets[phase], batch_size=batch_size[phase],
                                          shuffle=True, num_workers=4, pin_memory=True) for phase in _modes}
    num_data = {phase:len(datasets[phase]) for phase in _modes}
    num_iter_epoch = {phase: ceil(num_data[phase] / batch_size[phase]) for phase in _modes}


    # test stage
    tic = time.time()
    phase = 'train'
    abssum_noisy = []
    last = len(data_loader[phase]) - 1
    print(str(last))
    for ii, data in enumerate(data_loader[phase]):
        im_noisy, im_gt = [x.cuda() for x in data]

        # im_denoise.clamp_(0.0, 1.0) ####
        ########################################
        # HU_min = -160
        # HU_range = 400
        # im_denoise_ = (im_denoise - HU_min)/HU_range
        # im_gt_ = (im_gt - HU_min)/HU_range
        # im_denoise_.clamp_(0.0, 1.0)
        # im_gt_.clamp_(0.0, 1.0)
        # # # # # # # # # # # # # # # #
        # ([16,3,128,128])
        dct_factor = 2 * im_gt.shape[2] * im_gt.shape[3]
        if ii is 0:
            abssum_noisy = torch.sum(torch.abs(im_noisy), (0, 1))
        else:
            abssum_noisy += torch.sum(torch.abs(im_noisy), (0, 1))

        if ii % 100 == 0 or ii is last-1:
            avg_noisy = (abssum_noisy / (ii+1) / 16)
            avg_noisy = avg_noisy.cpu().numpy()
            a00, a01, a02, a03 = avg_noisy[0,0], avg_noisy[0,-1], avg_noisy[-1,0], avg_noisy[-1,-1]
            print('{:05d} || {:.4f} | {:.4f} | {:.4f} | {:.4f}'.format(ii+1, a00, a01, a02, a03))
            plt.imshow(np.log(avg_noisy))
            plt.title('dctavg(log(max)={:.3f}), ii={:d}'.format(np.log(a00), ii+1))
            plt.savefig('figs/{:05d}_img.png'.format(ii+1)); plt.clf()
            plt.plot(np.diag(avg_noisy)); plt.title('dctvalue, ii={:d}'.format(ii+1)); plt.yscale("log")
            plt.savefig('figs/{:05d}_plot.png'.format(ii+1)); plt.clf()
            plt.plot()
            np.save('figs/{:05d}_diag.npy'.format(ii+1), np.diag(avg_noisy))
            np.save('figs/avg_of_{:d}.npy'.format((ii+1)*16), avg_noisy)

            if ii is last-1:
                print('saving global average...')
                np.save('figs/avg_of_{:d}.npy'.format(last*16), avg_noisy)


    print('-'*50)

    toc = time.time()
    print('Test done!')

def dct_property_check_experiment():

    datasets = {'train':LDCTTrain(h5_file='',
                                  length=5000*16*4,
                                  pch_size=128,
                                  mask=False)}
    experiments(datasets)

if __name__ == '__main__':
    # dct_property_check_experiment()
    fit_dct_curve()
