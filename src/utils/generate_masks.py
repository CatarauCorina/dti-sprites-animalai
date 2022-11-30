import numpy as np
from PIL import Image
import os


if bool(os.environ.get('IS_SERVER', True)) == True:
    from utils.path import DATASETS_PATH
else:
    from src.utils.path import DATASETS_PATH

from torchvision import transforms


def create_masks():
    count = 0
    max_masks = 18273
    path = f'{DATASETS_PATH}/animal-ai-competition'
    blackblankimage = transforms.ToPILImage()(np.zeros(shape=[128, 128, 3], dtype=np.uint8))
    while count < max_masks:
        blackblankimage.save(f'{path}/masks/{count}.png')
        count+=1


def main():
    create_masks()


if __name__ == '__main__':
    main()