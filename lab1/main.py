import cv2
import glob
import numpy as np
import os.path as osp
from argparse import ArgumentParser
from utils.compute_iou import compute_ious


def segment_fish(img):
    """
    This method should compute masks for given image
    Params:
        img (np.ndarray): input image in BGR format
    Returns:
        mask (np.ndarray): fish mask. should contain bool values
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    light_orange = (1, 190, 150)
    dark_orange = (30, 255, 255)

    light_white = (60, 0, 200)
    dark_white = (145, 150, 255)

    mask_orange = cv2.inRange(img, light_orange, dark_orange)
    mask_white = cv2.inRange(img, light_white, dark_white)
    final_mask = mask_orange + mask_white

    result = cv2.bitwise_and(img, img, mask=final_mask)

    return result


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--is_train", action="store_true")
    args = parser.parse_args()
    stage = 'train' if args.is_train else 'test'

    data_root = osp.join("dataset", stage, "imgs")
    img_paths = glob.glob(osp.join(data_root, "*.jpg"))
    len(img_paths)

    masks = dict()
    for path in img_paths:
        img = cv2.imread(path)
        mask = segment_fish(img)
        masks[osp.basename(path)] = mask
    print(compute_ious(masks, osp.join("dataset", stage, "masks")))
