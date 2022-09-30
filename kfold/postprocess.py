import cv2
import json
import argparse
import numpy as np
from tqdm import tqdm
from shutil import copy
from pathlib import Path
from collections import Counter


def parse_args():
    parser = argparse.ArgumentParser(description='Prepare Eyes Dataset for Train')

    parser.add_argument('-src', help='source directory with raw inference data', default='./result/exp')
    parser.add_argument('-thr', help='threshold to remove small area regions', type=int, default=200)
    parser.add_argument('-fill_value', help='how to fill mask', type=int, default=255)

    return parser.parse_args()


def main(args):
    src = Path(args.src)
    dst = Path(args.src + f'_post_{args.thr}')
    dst.mkdir(exist_ok=True)

    for f in tqdm(src.glob('*.png')):
        mask = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
        new_mask = np.zeros(mask.shape, dtype=np.uint8)
        ret, mask = cv2.connectedComponents(mask)

        for p in range(1, ret + 1):
            if np.sum(mask == p) > args.thr:
                new_mask[mask == p] = args.fill_value
        cv2.imwrite(str(dst / f.name), new_mask)


if __name__ == "__main__":
    main(parse_args())