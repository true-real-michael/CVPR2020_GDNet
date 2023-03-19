#  @Date    : 2023-03-19
#  @Editor  : Mikhail Kiselyov
#  @E-mail  : kiselev.0353@gmail.com
#  Provided as is

import os
import time

import numpy as np
from pathlib import Path
from PIL import Image
import torch.cuda
from torch.autograd import Variable
from torchvision import transforms

from gdnet import GDNet
from misc import crf_refine


class NetworkRunner:
    def __init__(self,
                 input_dir: Path,
                 output_dir: Path,
                 log_path: Path,
                 model_path: Path,
                 do_crf_refine: bool,
                 scale: int):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.log_path = log_path
        self.model_path = model_path
        self.scale = scale
        self.do_crf_refine = do_crf_refine

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.img_transform = transforms.Compose([
            transforms.Resize((self.scale, self.scale)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.to_pil = transforms.ToPILImage()

        self._load_model()

    class _Timer:
        def __enter__(self):
            self.start = time.perf_counter_ns()
            return self

        def __exit__(self, *args):
            self.end = time.perf_counter_ns()
            self._elapsed = self.end - self.start

        def elapsed(self):
            return self._elapsed

    def run(self):
        with torch.no_grad():
            img_list = [img_name for img_name in os.listdir(self.input_dir)]
            for idx, img_name in enumerate(img_list):
                print('predicting for {}: {:>4d} / {}'.format(img_name, idx + 1, len(img_list)))

                try:
                    size, img = self._read_img(img_name)
                except Exception:
                    print(f'failed to load image {img_name}\nskipping')
                    continue

                with self._Timer() as timer:
                    img_var = Variable(img).to(self.device)
                    f1, f2, f3 = self.net(img_var)
                    f1 = f1.data.squeeze(0).cpu()
                    f2 = f2.data.squeeze(0).cpu()
                    f3 = f3.data.squeeze(0).cpu()
                    f1 = np.array(transforms.Resize(size)(self.to_pil(f1)))
                    f2 = np.array(transforms.Resize(size)(self.to_pil(f2)))
                    f3 = np.array(transforms.Resize(size)(self.to_pil(f3)))

                    if self.do_crf_refine:
                        # f1 = crf_refine(np.array(img.convert('RGB')), f1)
                        # f2 = crf_refine(np.array(img.convert('RGB')), f2)
                        f3 = crf_refine(np.array(img), f3)

                self._write_img(img_name, f1, f2, f3, timer.elapsed())

    def _load_model(self):
        with_gpu = torch.cuda.is_available()
        self.device = torch.device("cuda" if with_gpu else "cpu")
        print("CUDA is available, device = 'gpu'" if with_gpu else "CUDA is unavailable, device = 'cpu'")
        self.net = GDNet().to(self.device)
        self.net.load_state_dict(torch.load(self.model_path))
        print('Load {} succeed!'.format(os.path.basename(self.model_path)))
        self.net.eval()

    def _read_img(self, img_name: str):
        img = Image.open(self.input_dir / Path(img_name))
        if img.mode != 'RGB':
            img = img.convert('RGB')
            print("{} is a gray image.".format(img_name))

        return img.size, self.img_transform(img).unsqueeze(0)

    def _write_img(self, img_name, f1, f2, f3, elapsed):
        Image.fromarray(f1).save(self.output_dir / Path(img_name[:-4] + "_h.png"))
        Image.fromarray(f2).save(self.output_dir / Path(img_name[:-4] + "_l.png"))
        Image.fromarray(f3).save(self.output_dir / Path(img_name))

        print(f'image {img_name} processed: {elapsed}')
