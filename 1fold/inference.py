import argparse
from pathlib import Path

import albumentations as albu
import cv2
import torch
from albumentations.pytorch import ToTensorV2
from segmentation_models_pytorch import Unet
from tqdm import tqdm

from dataset import get_test_transform


def parse_args():
    parser = argparse.ArgumentParser(description='1fold inference on the eyes data')

    parser.add_argument('-src', help='directory to apply inference', default='../datasets/test')
    parser.add_argument('-exp', help='experiment name', default='exp')
    parser.add_argument('-fill_value', help='how to fill mask', type=int, default=255)

    # model parameters
    parser.add_argument('-weight', help='model weights', default='./lightning_logs/version_0/checkpoints/*.ckpt')
    parser.add_argument('-encoder_name', default='efficientnet-b3')

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

    model_path = Path(args.weight)
    model = Unet(
        encoder_name=args.encoder_name,
        encoder_weights=None,
        classes=1,
        activation=None,
    )

    model.load_state_dict(torch.load(model_path, map_location="cuda:0")['state_dict'])
    model.eval().cuda()

    for im in tqdm(test.glob('*.png')):
        img = cv2.imread(str(im))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # prediction
        with torch.no_grad():
            mask = simple_predict(model, img)
            mask[mask > 0] = args.fill_value

        cv2.imwrite(str(dst / im.name), mask)


if __name__ == "__main__":
    main(parse_args())
