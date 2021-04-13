
import os
import sys
import time
import torch
import torch.utils.data as uData
from networks import UNetD
from datasets.DenoisingDatasets import LDCTTest512
from math import ceil
from utils import *
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
import warnings
from pathlib import Path
import commentjson as json

# filter warnings
warnings.simplefilter('ignore', Warning, lineno=0)

# default dtype
torch.set_default_dtype(torch.float32)

_C = 3
_modes = ['val']


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
        HU_min = -160
        HU_range = 400
        # im_denoise_ = (im_denoise - HU_min)/HU_range
        im_gt_ = (im_gt - HU_min)/HU_range
        # im_denoise_.clamp_(0.0, 1.0)
        im_gt_.clamp_(0.0, 1.0)
        # im_gt_ = from4kto400(im_gt)
        im_denoise_ = from4kto400(im_denoise)
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
            x5 = vutils.make_grid(im_noisy[:, 1, :, :].unsqueeze(1).repeat(1, 3, 1, 1), normalize=True, scale_each=True)
            writer.add_image(phase+' Noisy Image', x5, step_img[phase])
            x6 = vutils.make_grid(im_noisy, normalize=True, scale_each=True)
            writer.add_image(phase+' Noisy Image 3 slices', x6, step_img[phase])
            step_img[phase] += 1

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
    datasets = {'val':LDCTTest512(args['SIDD_test_h5'])}

    # test model
    print('\nBegin testing with GPU: ' + str(args['gpu_id']))
    test_all(net, datasets, args)

if __name__ == '__main__':
    main()