import os

if bool(os.environ.get('IS_SERVER',True)) == True:
    from dataset.cosegmentation import WeizmannHorseDataset
    from dataset.gtsrb import GTSRB8Dataset
    from dataset.multi_object import DSpritesGrayDataset, TetrominoesDataset, CLEVR6Dataset, AnimalAIDataset, CreateDataset
    from dataset.instagram import InstagramDataset
    from dataset.torchvision import SVHNDataset


else:
    from .cosegmentation import WeizmannHorseDataset
    from .gtsrb import GTSRB8Dataset
    from .multi_object import DSpritesGrayDataset, TetrominoesDataset, CLEVR6Dataset, AnimalAIDataset, CreateDataset
    from .instagram import InstagramDataset
    from .torchvision import SVHNDataset


def get_dataset(dataset_name):
    return {
        # Cosegmentation
        'weizmann_horse': WeizmannHorseDataset,

        # Custom
        'gtsrb8': GTSRB8Dataset,
        'instagram': InstagramDataset,

        # MultiObject
        'clevr6': CLEVR6Dataset,
        'dsprites_gray': DSpritesGrayDataset,
        'tetrominoes': TetrominoesDataset,

        # Torchvision
        'svhn': SVHNDataset,
        'animalai': AnimalAIDataset,
        'create': CreateDataset
    }[dataset_name]
