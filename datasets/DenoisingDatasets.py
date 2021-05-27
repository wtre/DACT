#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2019-09-02 15:51:11

import sys
import torch
import h5py as h5
import random
import cv2
import os
import numpy as np
import torch.utils.data as uData
from glob import glob
from skimage import img_as_float32 as img_as_float
from .data_tools import random_augmentation
from . import BaseDataSetH5, BaseDataSetFolder
from scipy import fftpack


# Benchmardk Datasets: and SIDD
class BenchmarkTrain(BaseDataSetH5):
    def __init__(self, h5_file, length, pch_size=128, mask=False):
        super(BenchmarkTrain, self).__init__(h5_file, length)
        self.pch_size = pch_size
        self.mask = mask

    def __getitem__(self, index):
        num_images = self.num_images
        ind_im = random.randint(0, num_images-1)

        with h5.File(self.h5_path, 'r') as h5_file:
            imgs_sets = h5_file[self.keys[ind_im]]
            im_gt, im_noisy = self.crop_patch(imgs_sets)
        im_gt = img_as_float(im_gt)
        im_noisy = img_as_float(im_noisy)

        # data augmentation
        im_gt, im_noisy = random_augmentation(im_gt, im_noisy)

        im_gt = torch.from_numpy(im_gt.transpose((2, 0, 1)))
        im_noisy = torch.from_numpy(im_noisy.transpose((2, 0, 1)))

        if self.mask:
            return im_noisy, im_gt, torch.ones((1,1,1), dtype=torch.float32)
        else:
            return im_noisy, im_gt

class BenchmarkTest(BaseDataSetH5):
    def __getitem__(self, index):
        with h5.File(self.h5_path, 'r') as h5_file:
            imgs_sets = h5_file[self.keys[index]]
            C2 = imgs_sets.shape[2]
            C = int(C2/2)
            im_noisy = np.array(imgs_sets[:, :, :C])
            im_gt = np.array(imgs_sets[:, :, C:])
        im_gt = img_as_float(im_gt)
        im_noisy = img_as_float(im_noisy)

        im_gt = torch.from_numpy(im_gt.transpose((2, 0, 1)))
        im_noisy = torch.from_numpy(im_noisy.transpose((2, 0, 1)))

        return im_noisy, im_gt


def weight_matrix_func(x, a1, b1, a2, b2, c):
    if torch.is_tensor(x):
        return torch.exp(a1 * x**b1 + a2 * x**b2 + c)
    else:
        return np.exp(a1 * x**b1 + a2 * x**b2 + c)


def create_wmat(pch_size):
    coeffs = np.load('data/coeffs_03301.npy')
    wmat = torch.linspace(0, 0.5, pch_size)
    wmat = wmat.repeat(pch_size, 1)
    wmat = wmat + wmat.transpose(0, 1)
    wmat = weight_matrix_func(wmat, *coeffs)
    return wmat.repeat(3, 1, 1)


# Custom Datasets: Mayo
# TODO: [doing now] swap the following two function into the mayo ones
class LDCTTrain(BaseDataSetH5):
    def __init__(self, h5_file, length, pch_size=128, mask=False, if_dct_norm=True, domain='CT'):
        # TODO: h5_file and mask is not working!
        self.files_x = sorted(glob(os.path.join('../mocomed/dataset/DANet_CT/128_LDCT_train', 'train_LDCT_*.npy')))
        self.files_y = sorted(glob(os.path.join('../mocomed/dataset/DANet_CT/128_NDCT_train', 'train_NDCT_*.npy')))
        self.mask = mask
        self.if_dct_norm = if_dct_norm
        self.domain=domain
        self.p = pch_size
        if if_dct_norm:
            self.wmat = create_wmat(pch_size)

    def __getitem__(self, index):
        # for DCT we decided to multiply sqrt(2/N) for both dct and idct.
        x2 = np.load(self.files_x[index])
        x = fftpack.dctn(x2, axes=(0,1))
        x = torch.from_numpy(x).float().permute((2,0,1)) / (2*self.p)
        # x2 = torch.from_numpy(x2).float().permute((2,0,1))
        y2 = np.load(self.files_y[index])
        y = fftpack.dctn(y2, axes=(0,1))
        y = torch.from_numpy(y).float().permute((2,0,1)) / (2*self.p)
        # y2 = torch.from_numpy(y2).float().permute((2,0,1))

        if self.if_dct_norm:
            x *= self.wmat
            y *= self.wmat
        return x, y

    def __len__(self):
        return len(self.files_x)


class LDCTTest(BaseDataSetH5):
    def __init__(self, index, if_dct_norm=False):
        self.files_x = sorted(glob(os.path.join('../mocomed/dataset/DANet_CT/128_LDCT_val', 'val_LDCT_*.npy')))
        self.files_y = sorted(glob(os.path.join('../mocomed/dataset/DANet_CT/128_NDCT_val', 'val_NDCT_*.npy')))
        self.if_dct_norm = if_dct_norm
        if if_dct_norm:
            self.wmat = create_wmat(128)

    def __getitem__(self, index):
        x2 = np.load(self.files_x[index])
        dct_factor = 2 * x2.shape[0] * x2.shape[1]
        x = fftpack.dctn(x2, axes=(0,1))
        x = torch.from_numpy(x).float().permute((2,0,1)) / dct_factor
        # x2 = torch.from_numpy(x2).float().permute((2,0,1))
        y2 = np.load(self.files_y[index])
        y = fftpack.dctn(y2, axes=(0,1))
        y = torch.from_numpy(y).float().permute((2,0,1)) / dct_factor
        # y2 = torch.from_numpy(y2).float().permute((2,0,1))

        if self.if_dct_norm:
            x *= self.wmat
            y *= self.wmat
        return x, y#, x2, y2

    def __len__(self):
        return len(self.files_x)


class LDCTTest512(BaseDataSetH5):
    def __init__(self, index, if_dct_norm=False):
        self.files_x = sorted(glob(os.path.join('../mocomed/dataset/DANet_CT/512_LDCT_test', 'test_LDCT_*.npy')))
        self.files_y = sorted(glob(os.path.join('../mocomed/dataset/DANet_CT/512_NDCT_test', 'test_NDCT_*.npy')))
        self.if_dct_norm = if_dct_norm

    def __getitem__(self, index):
        x2 = np.load(self.files_x[index])
        dct_factor = 2 * x2.shape[0] * x2.shape[1]
        x = fftpack.dctn(x2, axes=(0,1))
        x = torch.from_numpy(x).float().permute((2,0,1)) / dct_factor
        # x2 = torch.from_numpy(x2).float().permute((2,0,1))
        y2 = np.load(self.files_y[index])
        y = fftpack.dctn(y2, axes=(0,1))
        y = torch.from_numpy(y).float().permute((2,0,1)) / dct_factor
        # y2 = torch.from_numpy(y2).float().permute((2,0,1))

        if self.if_dct_norm:
            wmat = create_wmat(y.shape[0])
            x *= wmat
            y *= wmat
        return x, y#, x2, y2


    def __len__(self):
        return len(self.files_x)


class FakeTrain(BaseDataSetFolder):
    def __init__(self, path_list, length, pch_size=128):
        super(FakeTrain, self).__init__(path_list, pch_size, length)

    def __getitem__(self, index):
        num_images = self.num_images
        ind_im = random.randint(0, num_images-1)

        im_gt = img_as_float(cv2.imread(self.path_list[ind_im], 1)[:, :, ::-1])
        im_gt = self.crop_patch(im_gt)

        # data augmentation
        im_gt = random_augmentation(im_gt)[0]

        im_gt = torch.from_numpy(im_gt.transpose((2, 0, 1)))

        return im_gt, im_gt, torch.zeros((1,1,1), dtype=torch.float32)

class PolyuTrain(BaseDataSetFolder):
    def __init__(self, path_list, length, pch_size=128, mask=False):
        super(PolyuTrain, self).__init__(path_list, pch_size, length)
        self.mask = mask

    def __getitem__(self, index):
        num_images = self.num_images
        ind_im = random.randint(0, num_images-1)

        path_noisy = self.path_list[ind_im]
        head, tail = os.path.split(path_noisy)
        path_gt = os.path.join(head, tail.replace('real', 'mean'))
        im_noisy = img_as_float(cv2.imread(path_noisy, 1)[:, :, ::-1])
        im_gt = img_as_float(cv2.imread(path_gt, 1)[:, :, ::-1])
        im_noisy, im_gt = self.crop_patch(im_noisy, im_gt)

        # data augmentation
        im_gt, im_noisy = random_augmentation(im_gt, im_noisy)

        im_gt = torch.from_numpy(im_gt.transpose((2, 0, 1)))
        im_noisy = torch.from_numpy(im_noisy.transpose((2, 0, 1)))

        if self.mask:
            return im_noisy, im_gt, torch.ones((1,1,1), dtype=torch.float32)
        else:
            return im_noisy, im_gt

    def crop_patch(self, im_noisy, im_gt):
        pch_size = self.pch_size
        H, W, _ = im_noisy.shape
        ind_H = random.randint(0, H-pch_size)
        ind_W = random.randint(0, W-pch_size)
        im_pch_noisy = im_noisy[ind_H:ind_H+pch_size, ind_W:ind_W+pch_size,]
        im_pch_gt = im_gt[ind_H:ind_H+pch_size, ind_W:ind_W+pch_size,]
        return im_pch_noisy, im_pch_gt
