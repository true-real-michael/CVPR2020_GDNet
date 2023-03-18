import argparse
from pathlib import Path
from infer import infer

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
        '--do_crf_refine',
        action='store_true',
        help='Optional CRF refinement. Default: False'
    )

    args = parser.parse_args()

    infer(args.scale,
          args.pretrained_model_path,
          args.output_dir,
          args.input_dir,
          args.do_crf_refine)
