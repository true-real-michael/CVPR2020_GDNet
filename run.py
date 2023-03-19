#  @Date    : 2023-03-19
#  @Editor  : Mikhail Kiselyov
#  @E-mail  : kiselev.0353@gmail.com
#  Provided as is

import argparse
from pathlib import Path

from network_runner import NetworkRunner

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--pretrained_model_path',
        type=str,
        help='Path to the .pth file',
        default=Path(__file__).parent / '200.pth'
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        help='Path to the directory that contains the images',
        default=Path(__file__).parent / 'input'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        help='Path to the directory in which the result images are written',
        default=Path(__file__).parent / 'output'
    )
    parser.add_argument(
        '--scale',
        type=int,
        default=416,
        help='Scale parameter for resizing images. Default: 416'
    )
    parser.add_argument(
        '--do-crf-refine',
        action='store_true',
        help='Optional CRF refinement. Default: False'
    )
    parser.add_argument(
        '--log-path',
        type=Path,
        default=Path(__file__).parent / 'output' / 'log.txt'
    )

    args = parser.parse_args()

    NetworkRunner(args.input_dir,
                  args.output_dir,
                  args.log_path,
                  args.pretrained_model_path,
                  args.do_crf_refine,
                  args.scale).run()
