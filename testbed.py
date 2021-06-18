
import os
import sys
import time
import torch
import torch.utils.data as uData
from networks import UNetD
from datasets.DenoisingDatasets import LDCTTrain, LDCTTest512
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
    curve_1p = 512*np.load('figs/test512/00301_diag.npy')
    xdata = np.linspace(0, 1, curve_1p.shape[0])
    popt, pcov = curve_fit(func, xdata, np.log(curve_1p))
    plt.plot(xdata, curve_1p, 'b-')
    plt.plot(xdata, np.exp(func(xdata, *popt)), 'r-')
    plt.yscale("log")
    plt.show()
    print(*popt)
    np.save('figs/test512/coeffs_00301.npy', popt)


def compare_dct_curve():
    curve_1p = 512*np.load('figs/test512/00301_diag.npy')
    xdata = np.linspace(0, 2, curve_1p.shape[0])
    curve_2p = np.load('figs/03301_diag.npy')
    xdata2 = np.linspace(0, 1, curve_2p.shape[0])
    plt.plot(xdata, curve_1p, 'b-')
    plt.plot(xdata2, curve_2p, 'g-')
    plt.yscale("log")
    plt.show()



def from4kto400(input):
    out = input * 4096 - 1024
    return ((out + 160)/400).clamp_(0.0, 1.0)

def experiments(datasets, pth):
    batch_size = {'train':4}
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

        if ii % 100 == 0 or ii is last-2:
            avg_noisy = (abssum_noisy / (ii+1) / 16)
            avg_noisy = avg_noisy.cpu().numpy()
            a00, a01, a02, a03 = avg_noisy[0,0], avg_noisy[0,-1], avg_noisy[-1,0], avg_noisy[-1,-1]
            print('{:05d} || {:.6f} | {:.6f} | {:.6f} | {:.6f}'.format(ii+1, a00, a01, a02, a03))
            plt.imshow(np.log(avg_noisy))
            plt.title('dctavg(log(max)={:.3f}), ii={:d}'.format(np.log(a00), ii+1))
            plt.savefig(pth+'/{:05d}_img.png'.format(ii+1)); plt.clf()
            plt.plot(np.diag(avg_noisy)); plt.title('dctvalue, ii={:d}'.format(ii+1)); plt.yscale("log")
            plt.savefig(pth+'/{:05d}_plot.png'.format(ii+1)); plt.clf()
            plt.plot()
            np.save(pth+'/{:05d}_diag.npy'.format(ii+1), np.diag(avg_noisy))
            np.save(pth+'/avg_of_{:d}.npy'.format((ii+1)*16), avg_noisy)

            if ii is last-2:
                print('saving global average...')
                np.save(pth+'/avg_of_{:d}.npy'.format(last*16), avg_noisy)


    print('-'*50)

    toc = time.time()
    print('Test done!')


def dct_property_check_experiment():

    # datasets = {'train':LDCTTrain(h5_file='',
    #                               length=5000*16*4,
    #                               pch_size=128,
    #                               mask=False)}
    datasets = {'train':LDCTTest512('')}
    experiments(datasets, pth='figs/test512')


def dct_stat_per_size():

    # import image 128x128
    c1 = np.load('../mocomed/dataset/DANet_CT/128_LDCT_val/val_LDCT_03_999.npy')
    # import image 512x512
    c5 = np.load('../mocomed/dataset/DANet_CT/512_LDCT_test/test_LDCT_09_99.npy')
    # noise 128
    u1 = np.random.rand(128, 128, 3)
    # noise 512
    u5 = np.random.rand(512, 512, 3)
    # gauss 128
    n1 = np.clip(np.random.normal(.25, .1, (128, 128, 3)), 0, 1)
    # gauss 512
    n5 = np.clip(np.random.normal(.25, .1, (512, 512, 3)), 0, 1)

    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)
    ax1.imshow(c1); ax4.imshow(c5)
    ax2.imshow(u1); ax5.imshow(u5)
    ax3.imshow(n1); ax6.imshow(n5)
    plt.show()

    c1d = fftpack.dctn(c1, axes=(0,1)) / 2 / 128
    c5d = fftpack.dctn(c5, axes=(0,1)) / 2 / 512
    u1d = fftpack.dctn(c1, axes=(0,1)) / 2 / 128
    u5d = fftpack.dctn(c5, axes=(0,1)) / 2 / 512
    n1d = fftpack.dctn(n1, axes=(0,1)) / 2 / 128
    n5d = fftpack.dctn(n5, axes=(0,1)) / 2 / 512

    q = [0, .01, .25, .5, .75, .99, 1]
    print(str(np.quantile(np.log(c1d), q)))
    print(str(np.quantile(c5d, q)))
    print(str(np.quantile(u1d, q)))
    print(str(np.quantile(u5d, q)))
    print(str(np.quantile(n1d, q)))
    print(str(np.quantile(np.log(n5d), q)))

    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)
    ax1.imshow(np.log(np.abs(c1d[:,:,1]))); ax1.set_title(str(np.max(c1d))+" - "+str(np.min(np.abs(c1d))))
    ax4.imshow(np.log(np.abs(c5d[:,:,1]))); ax2.set_title(str(np.max(c5d))+" - "+str(np.min(np.abs(c5d))))
    ax2.imshow(np.log(np.abs(u1d[:,:,1]))); ax3.set_title(str(np.max(u1d))+" - "+str(np.min(np.abs(u1d))))
    ax5.imshow(np.log(np.abs(u5d[:,:,1]))); ax4.set_title(str(np.max(u5d))+" - "+str(np.min(np.abs(u5d))))
    ax3.imshow(np.log(np.abs(n1d[:,:,1]))); ax5.set_title(str(np.max(n1d))+" - "+str(np.min(np.abs(n1d))))
    ax6.imshow(np.log(np.abs(n5d[:,:,1]))); ax6.set_title(str(np.max(n5d))+" - "+str(np.min(np.abs(n5d))))
    plt.show()


if __name__ == '__main__':
    # dct_property_check_experiment()
    # fit_dct_curve()
    compare_dct_curve()
    # dct_stat_per_size()
