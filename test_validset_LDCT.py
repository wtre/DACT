
import os
import sys
import time
import torch
import torch.utils.data as uData
from networks import UNetD
from datasets.DenoisingDatasets import LDCTTest
from math import ceil
from utils import *
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
import warnings
from pathlib import Path
import commentjson as json
from scipy import fftpack
# sys.path.insert(0, './datasets')
# from DenoisingDatasets import create_wmat

# filter warnings
warnings.simplefilter('ignore', Warning, lineno=0)

# default dtype
torch.set_default_dtype(torch.float32)

_C = 3
_modes = ['val']


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


def load_wmat():
    wmat = torch.from_numpy(np.load('data/avg_of_52816.npy'))
    return wmat.repeat(3, 1, 1)


def from4kto400(input):
    out = input * 4096 - 1024
    return ((out + 160)/400).clamp_(0.0, 1.0)


def test_all(net, datasets, args):
    batch_size = {'val':args['batch_size']}
    data_loader = {phase:uData.DataLoader(datasets[phase], batch_size=batch_size[phase],
                                          shuffle=False, num_workers=args['num_workers'], pin_memory=True) for phase in _modes}
    num_data = {phase:len(datasets[phase]) for phase in _modes}
    num_iter_epoch = {phase: ceil(num_data[phase] / batch_size[phase]) for phase in _modes}
    step_img = args['step_img'] if False else {x:0 for x in _modes}
    writer = SummaryWriter(str(Path(args['log_dir'])))

    mae_epoch = {'val':0}

    # wmat_base = create_wmat(128)
    wmat_base = load_wmat()

    # test stage
    tic = time.time()
    net['D'].eval()
    psnr_per_epoch = ssim_per_epoch = 0
    phase = 'val'
    for ii, data in enumerate(data_loader[phase]):
        im_noisy, im_gt = [x.cuda() for x in data]
        with torch.set_grad_enabled(False):
            im_denoise = im_noisy - net['D'](im_noisy)

        mae_iter = F.l1_loss(im_denoise, im_gt)
        # im_denoise.clamp_(0.0, 1.0) ####
        ########################################
        # HU_min = -160
        # HU_range = 400
        # im_denoise_ = (im_denoise - HU_min)/HU_range
        # im_gt_ = (im_gt - HU_min)/HU_range
        # im_denoise_.clamp_(0.0, 1.0)
        # im_gt_.clamp_(0.0, 1.0)
        wmat = wmat_base.repeat(im_gt.shape[0], 1, 1, 1).numpy()
        # # # # # # # # # # # # # # # #
        dct_factor = 2 * im_gt.shape[2] #* im_gt.shape[3]
        im_gt_np = im_gt.cpu().numpy()
        im_denoise_np = im_denoise.cpu().numpy()
        im_noisy_np = im_noisy.cpu().numpy()
        im_gt = torch.from_numpy(fftpack.idctn(im_gt_np*wmat, axes=(3,2))) / dct_factor
        im_denoise = torch.from_numpy(fftpack.idctn(im_denoise_np*wmat, axes=(3,2))) / dct_factor
        im_noisy = torch.from_numpy(fftpack.idctn(im_noisy_np*wmat, axes=(3,2))) / dct_factor
        # # # # # # # # # # # # # # # #
        im_gt_ = from4kto400(im_gt)
        im_denoise_ = from4kto400(im_denoise)
        im_noisy_ = from4kto400(im_noisy)
        ########################################
        mae_epoch[phase] += mae_iter
        psnr_iter = batch_PSNR(im_denoise_, im_gt_)
        psnr_per_epoch += psnr_iter
        ssim_iter = batch_SSIM(im_denoise_, im_gt_)
        ssim_per_epoch += ssim_iter
        # print statistics every log_interval mini_batches
        if (ii+1) % 25 == 0:
            log_str = 'val:{:0>3d}/{:0>3d}, mae={:.2e}, psnr={:4.2f}, ssim={:5.4f}'
            print(log_str.format(ii+1, num_iter_epoch[phase], mae_iter, psnr_iter, ssim_iter))
            # tensorboard summary
            x1 = vutils.make_grid(im_denoise[:, 1, :, :].unsqueeze(1).repeat(1, 3, 1, 1), normalize=True, scale_each=True)
            writer.add_image(phase+' Denoised images', x1, step_img[phase])
            x2 = vutils.make_grid(im_gt[:, 1, :, :].unsqueeze(1).repeat(1, 3, 1, 1), normalize=True, scale_each=True)
            writer.add_image(phase+' GroundTruth', x2, step_img[phase])
            x3 = vutils.make_grid(im_gt_[:, 1, :, :].unsqueeze(1).repeat(1, 3, 1, 1), normalize=True, scale_each=False, range=(0,1))
            writer.add_image(phase+' GroundTruth but must be [-160, 240]', x3, step_img[phase])
            x4 = vutils.make_grid(im_gt_[:, 1, :, :].unsqueeze(1).repeat(1, 3, 1, 1), normalize=True, scale_each=False, range=(.25,.75))
            writer.add_image(phase+' GroundTruth test on [-60, 140]', x4, step_img[phase])
            x5 = vutils.make_grid(im_noisy[:, 1, :, :].unsqueeze(1).repeat(1, 3, 1, 1), normalize=True, scale_each=True)
            writer.add_image(phase+' Noisy Image', x5, step_img[phase])
            x6 = vutils.make_grid(im_noisy_, normalize=False, scale_each=False, range=(0, 1))
            writer.add_image(phase+' Noisy Image 3 slices, [-160, 240]', x6, step_img[phase])
            x7 = vutils.make_grid(im_denoise_[:, 1, :, :].unsqueeze(1).repeat(1, 3, 1, 1)-im_gt_, normalize=False, scale_each=False, range=(-.5, .5))
            writer.add_image(phase+' difference 3 slices (4k->400)', x7, step_img[phase])
            x8 = vutils.make_grid(im_noisy_[:, 1, :, :].unsqueeze(1).repeat(1, 3, 1, 1)-im_gt_, normalize=False, scale_each=False, range=(-.5, .5))
            writer.add_image(phase+' gt noise diff (4k->400)', x8, step_img[phase])
            x9 = vutils.make_grid(im_noisy[:, 1, :, :].unsqueeze(1).repeat(1, 3, 1, 1)-im_gt, normalize=False, scale_each=False, range=(-.5, .5))
            writer.add_image(phase+' gt noise diff (0,1 as it is)', x9, step_img[phase])
            x10 = vutils.make_grid(im_noisy[:, 1, :, :].unsqueeze(1).repeat(1, 3, 1, 1)-im_gt, normalize=True, scale_each=False, range=(-.5, .5))
            writer.add_image(phase+' same thing but norm=True', x10, step_img[phase])
            step_img[phase] += 1
            # x11 = vutils.make_grid(torch.from_numpy(im_denoise_np).unsqueeze(1).repeat(1, 3, 1, 1), normalize=True, scale_each=True)
            # writer.add_image(phase+' Denoised dct', x11, step_img[phase])
            # x12 = vutils.make_grid(torch.from_numpy(im_gt_np).unsqueeze(1).repeat(1, 3, 1, 1), normalize=True, scale_each=True)
            # writer.add_image(phase+' Denoised gt', x12, step_img[phase])

    psnr_per_epoch /= (ii+1)
    ssim_per_epoch /= (ii+1)
    mae_epoch[phase] /= (ii+1)
    print('mae={:.3e}, PSNR={:4.2f}, SSIM={:5.4f}'.format(mae_epoch[phase],
                                                                psnr_per_epoch, ssim_per_epoch))
    print('-'*150)

    toc = time.time()
    print('This test take time {:.2f}'.format(toc-tic))
    writer.close()
    print('Test done!')


def main():
    # set parameters
    with open('./configs/LDCT_test.json', 'r') as f:
        args = json.load(f)

    # set the available GPUs
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args['gpu_id'])

    # build up the denoiser
    netD= UNetD(_C, wf=args['wf'], depth=args['depth']).cuda()
    net = {'D':netD}


    if Path(args['model_path']).is_file():
        print('=> Loading checkpoint {:s}'.format(str(Path(args['model_path']))))
        checkpoint = torch.load(str(Path(args['model_path'])), map_location='cpu')
        netD.load_state_dict(checkpoint['model_state_dict']['D'])
        print('=> Loaded checkpoint {:s}'.format(args['model_path']))

        if Path(args['log_dir']).is_dir() and len(os.listdir(Path(args['log_dir']))) > 0:
            print('super babo')
            exit()
            # shutil.rmtree(args['log_dir'])
        if not Path(args['log_dir']).is_dir():
            Path(args['log_dir']).mkdir()

    else:
        sys.exit('Please provide corrected model path!')

    # making dataset ####
    # datasets = {'train':BenchmarkTrain(h5_file=args['SIDD_train_h5'],
    #                                    length=5000*args['batch_size']*args['num_critic'],
    #                                    pch_size=args['patch_size'],
    #                                    mask=False),
    #             'val':BenchmarkTest(args['SIDD_test_h5'])}
    datasets = {'val':LDCTTest(args['SIDD_test_h5'])}

    # test model
    print('\nBegin testing with GPU: ' + str(args['gpu_id']))
    test_all(net, datasets, args)

if __name__ == '__main__':
    main()
