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
import numpy as np

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms

from config import gdd_testing_root, gdd_results_root
from misc import check_mkdir, crf_refine
from gdnet import GDNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def infer(scale: int,
          pretrained_model_path: str,
          output_dir: str,
          input_dir: str,
          do_crf_refine: bool):
    img_transform = transforms.Compose([
        transforms.Resize((scale, scale)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    to_pil = transforms.ToPILImage()

    net = GDNet().to(device)

    net.load_state_dict(torch.load(pretrained_model_path))
    print('Load {} succeed!'.format(os.path.basename(pretrained_model_path)))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    net.eval()
    with torch.no_grad():
        img_list = [img_name for img_name in os.listdir(input_dir)]
        start = time.time()
        for idx, img_name in enumerate(img_list):
            print('predicting for {}: {:>4d} / {}'.format(img_name, idx + 1, len(img_list)))
            img = Image.open(os.path.join(input_dir, img_name))
            if img.mode != 'RGB':
                img = img.convert('RGB')
                print("{} is a gray image.".format(img_name))
            w, h = img.size
            img_var = Variable(img_transform(img).unsqueeze(0)).to(device)
            f1, f2, f3 = net(img_var)
            f1 = f1.data.squeeze(0).cpu()
            f2 = f2.data.squeeze(0).cpu()
            f3 = f3.data.squeeze(0).cpu()
            f1 = np.array(transforms.Resize((h, w))(to_pil(f1)))
            f2 = np.array(transforms.Resize((h, w))(to_pil(f2)))
            f3 = np.array(transforms.Resize((h, w))(to_pil(f3)))
            if do_crf_refine:
                # f1 = crf_refine(np.array(img.convert('RGB')), f1)
                # f2 = crf_refine(np.array(img.convert('RGB')), f2)
                f3 = crf_refine(np.array(img.convert('RGB')), f3)

            # Image.fromarray(f1).save(os.path.join(ckpt_path, exp_name, '%s_%s' % (exp_name, args['snapshot']),
            #                                       img_name[:-4] + "_h.png"))
            # Image.fromarray(f2).save(os.path.join(ckpt_path, exp_name, '%s_%s' % (exp_name, args['snapshot']),
            #                                       img_name[:-4] + "_l.png"))
            Image.fromarray(f3).save(os.path.join(output_dir, img_name))

        end = time.time()
        print("Average Time Is : {:.2f}".format((end - start) / len(img_list)))
