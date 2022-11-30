import numpy as np
from PIL import Image

if os.environ.get('IS_SERVER', True) == 'True':
    from utils.path import DATASETS_PATH
else:
    from src.utils.path import DATASETS_PATH

from torchvision import transforms


def create_masks():
    count = 0
    max_masks = 70000
    path = f'{DATASETS_PATH}\CLEVR_v1.0\CLEVR_v1.0'
    blackblankimage = transforms.ToPILImage()(np.zeros(shape=[128, 128, 3], dtype=np.uint8))
    while count < max_masks:
        blackblankimage.save(f'{path}/sem_masks/{count}.png')
        count+=1


def main():
    create_masks()


if __name__ == '__main__':
    main()