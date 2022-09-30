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

    parser.add_argument('-src', help='path with raw data', default='./train')
    parser.add_argument('-dst', help='destination directory', default='./EYES_MICRO_CLEAR')

    return parser.parse_args()


def parse_polygon(mask, polygons):
    if len(polygons) == 1:
        points = [np.int32(polygons)]
        cv2.fillPoly(mask, points, 1)
    else:
        cv2.fillPoly(mask, [np.int32([polygon]) for polygon in polygons], 1)


def parse_geometry(mask, geometry, img_size):
    coordinates = geometry['coordinates']
    if geometry['type'] == 'MultiPolygon':
        for polygon in coordinates:
            parse_polygon(mask, polygon)
    else:
        parse_polygon(mask, coordinates)


def json2mask(json_name: Path, img_size: tuple[int]) -> np.ndarray:
    mask = np.zeros(img_size, dtype=np.uint8)

    with open(json_name, 'r', encoding='cp1251') as f:  # some files contain cyrillic letters, thus cp1251
        json_contents = json.load(f)

    if type(json_contents) == dict and json_contents['type'] == 'FeatureCollection':
        features = json_contents['features']
    elif type(json_contents) == list:
        features = json_contents
    else:
        features = [json_contents]

    for feature in features:
        parse_geometry(mask, feature['geometry'], img_size)

    return mask


def main(args):
    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(exist_ok=True)
    (dst / 'img_dir').mkdir(exist_ok=True)
    (dst / 'img_dir' / 'train').mkdir(exist_ok=True)
    (dst / 'ann_dir').mkdir(exist_ok=True)
    (dst / 'ann_dir' / 'train').mkdir(exist_ok=True)

    for im in tqdm(src.glob('*.png')):
        img = cv2.imread(str(im))
        img_size = img.shape[:2]
        try:
            mask = json2mask(im.with_suffix('.geojson'), img_size)
        except:
            print(f"exception with {im.name}")
            continue

        copy(im, dst / 'img_dir' / 'train' / im.name)
        cv2.imwrite(str(dst / 'ann_dir' / 'train' / im.name), mask)


if __name__ == "__main__":
    main(parse_args())