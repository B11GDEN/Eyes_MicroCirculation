import argparse
from pathlib import Path

import albumentations as albu
import cv2
import torch
from albumentations.pytorch import ToTensorV2
from segmentation_models_pytorch import Unet
import numpy as np
from tqdm import tqdm

from dataset import get_test_transform


def parse_args():
    parser = argparse.ArgumentParser(description='1fold inference on the eyes data')

    parser.add_argument('-src', help='directory to apply inference', default='../dataset/test')
    parser.add_argument('-exp', help='experiment name', default='exp')
    parser.add_argument('-fill_value', help='how to fill mask', type=int, default=255)

    # model parameters
    parser.add_argument('-ens_path', help='ensemble directory', default='./lightning_logs')
    parser.add_argument('-encoder_name', default='efficientnet-b3')
    parser.add_argument('-thr', help='segmentation threshold', type=float, default=0.6)

    return parser.parse_args()


# test time augmentation predict
def tta_predict(model, img):
    h, w, _ = img.shape
    submit_transform = get_test_transform()

    # predict origin image
    x = submit_transform(image=img)['image']
    x = torch.unsqueeze(x.to('cuda'), 0)
    pred = model(x)[0][0]
    temp_mask = pred.cpu().numpy()
    mask = temp_mask[:h, :w]

    # predict vertical flip
    x = submit_transform(image=img[::-1])['image']
    x = torch.unsqueeze(x.to('cuda'), 0)
    pred = model(x)[0][0]
    temp_mask = pred.cpu().numpy()
    mask += temp_mask[:h, :w][::-1]

    # predict horizontal flip
    x = submit_transform(image=img[:, ::-1])['image']
    x = torch.unsqueeze(x.to('cuda'), 0)
    pred = model(x)[0][0]
    temp_mask = pred.cpu().numpy()
    mask += temp_mask[:h, :w][:, ::-1]

    # predict vertical and horizontal flip
    x = submit_transform(image=img[::-1, ::-1])['image']
    x = torch.unsqueeze(x.to('cuda'), 0)
    pred = model(x)[0][0]
    temp_mask = pred.cpu().numpy()
    mask += temp_mask[:h, :w][::-1, ::-1]

    return mask / 4


# simple predict
def simple_predict(model, img):
    h, w, _ = img.shape
    submit_transform = get_test_transform()

    # predict origin image
    x = submit_transform(image=img)['image']
    x = torch.unsqueeze(x.to('cuda'), 0)
    pred = model(x)[0][0]
    temp_mask = pred.cpu().numpy()
    mask = temp_mask[:h, :w]

    return mask


def main(args):
    test = Path(args.src)
    dst = Path('result') / args.exp
    dst.mkdir(exist_ok=True, parents=True)

    models = []
    for model_path in Path(args.ens_path).glob('version_*/checkpoints/*.ckpt'):
        model_path = Path(model_path)
        model = Unet(
            encoder_name=args.encoder_name,
            encoder_weights=None,
            classes=1,
            activation=None,
        )
        model.load_state_dict(torch.load(model_path, map_location="cuda:0")['state_dict'])
        model.eval().cuda()
        models.append(model)

    for im in tqdm(test.glob('*.png')):

        img = cv2.imread(str(im))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        mask = np.zeros([h, w])

        # prediction
        with torch.no_grad():
            for model in models:
                mask += simple_predict(model, img)
            mask /= len(models)
        new_mask = np.zeros(mask.shape, dtype=np.uint8)
        new_mask[mask > args.thr] = args.fill_value

        cv2.imwrite(str(dst / im.name), new_mask)


if __name__ == "__main__":
    main(parse_args())