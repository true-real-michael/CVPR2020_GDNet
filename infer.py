"""
 @Time    : 2020/3/15 20:43
 @Author  : TaylorMei
 @E-mail  : mhy666@mail.dlut.edu.cn
 
 @Project : CVPR2020_GDNet
 @File    : infer.py
 @Function:
 
"""
import os
import time
import argparse
import numpy as np

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms

from config import gdd_testing_root, gdd_results_root
from misc import check_mkdir, crf_refine
from gdnet import GDNet

device_ids = [0]
torch.cuda.set_device(device_ids[0])

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--path_to_pretrained_model', type=str, help='Path to the .pth file')
    parser.add_argument(
        '--input_dir', type=str, default='input_images',
        help='Path to the directory that contains the images')
    parser.add_argument(
        '--output_dir', type=str, default='ouput_images',
        help='Path to the directory in which the result images are written')
    parser.add_argument(
        '--scale', type=int, default=416,
        help='Scale parameter for resizing images. Default: 416'
    )
    parser.add_argument(
        '--crf_refine', action='store_true',
        help='Optional CRF refinement. Default: False'
    )

    opt = parser.parse_args()

    img_transform = transforms.Compose([
        transforms.Resize((opt.scale, opt.scale)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    to_pil = transforms.ToPILImage()

    net = GDNet().cuda(device_ids[0])

    net.load_state_dict(torch.load(opt.path_to_pretrained_model))
    print('Load {} succeed!'.format(os.path.basename(opt.path_to_pretrained_model)))

    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)

    net.eval()
    with torch.no_grad():
        img_list = [img_name for img_name in os.listdir(opt.input_dir)]
        start = time.time()
        for idx, img_name in enumerate(img_list):
            print('predicting for {}: {:>4d} / {}'.format(img_name, idx + 1, len(img_list)))
            img = Image.open(os.path.join(opt.input_dir, img_name))
            if img.mode != 'RGB':
                img = img.convert('RGB')
                print("{} is a gray image.".format(name))
            w, h = img.size
            img_var = Variable(img_transform(img).unsqueeze(0)).cuda(device_ids[0])
            f1, f2, f3 = net(img_var)
            f1 = f1.data.squeeze(0).cpu()
            f2 = f2.data.squeeze(0).cpu()
            f3 = f3.data.squeeze(0).cpu()
            f1 = np.array(transforms.Resize((h, w))(to_pil(f1)))
            f2 = np.array(transforms.Resize((h, w))(to_pil(f2)))
            f3 = np.array(transforms.Resize((h, w))(to_pil(f3)))
            if opt.crf_refine:
                # f1 = crf_refine(np.array(img.convert('RGB')), f1)
                # f2 = crf_refine(np.array(img.convert('RGB')), f2)
                f3 = crf_refine(np.array(img.convert('RGB')), f3)

            # Image.fromarray(f1).save(os.path.join(ckpt_path, exp_name, '%s_%s' % (exp_name, args['snapshot']),
            #                                       img_name[:-4] + "_h.png"))
            # Image.fromarray(f2).save(os.path.join(ckpt_path, exp_name, '%s_%s' % (exp_name, args['snapshot']),
            #                                       img_name[:-4] + "_l.png"))
            Image.fromarray(f3).save(os.path.join(opt.output_dir, img_name))

        end = time.time()
        print("Average Time Is : {:.2f}".format((end - start) / len(img_list)))


if __name__ == '__main__':
    main()