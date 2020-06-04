import os

import kaggle
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms



label2name = {
    'n0': 'alouatta_palliata',
    'n1': 'erythrocebus_patas',
    'n2': 'cacajao_calvus',
    'n3': 'macaca_fuscata',
    'n4': 'cebuella_pygmea',
    'n5': 'cebus_capucinus',
    'n6': 'mico_argentatus',
    'n7': 'saimiri_sciureus',
    'n8': 'aotus_nigriceps',
    'n9': 'trachypithecus_johnii',
}

name2id = {
    'alouatta_palliata': 0,
    'erythrocebus_patas': 1,
    'cacajao_calvus': 2,
    'macaca_fuscata': 3,
    'cebuella_pygmea': 4,
    'cebus_capucinus': 5,
    'mico_argentatus': 6,
    'saimiri_sciureus': 7,
    'aotus_nigriceps': 8,
    'trachypithecus_johnii': 9,
}


class ImageTransform():
    '''
    This is image transform class. This class's action differs depending on the 'training' or 'validation'.
    It resize image size and normarize image color.
    Attributes
    -----------
    resize:int
        img size after resize

    mean : (R,G,B)
        average of each channel

    std : (R,G,B)
        standard deviation of each channel
    '''

    def __init__(self, resize, mean, std):
        self.data_transform = {
            'training': transforms.Compose([
                transforms.RandomResizedCrop(resize, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
            'validation': transforms.Compose([
                transforms.Resize(resize),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        }

    def __call__(self, img, phase='training'):
        return self.data_transform[phase](img)


class MonkeyDataset(Dataset):

    def __init__(self, path, transform=None, phase='training', download=False):
        super(MonkeyDataset, self).__init__()
        if download:
            kaggle.api.authenticate()
            kaggle.api.dataset_download_files('slothkong/10-monkey-species', path=path, unzip=True)

        self.file_list = self.make_datapath_list(path, phase)
        self.transform = transform
        self.phase = phase

    @staticmethod
    def make_datapath_list(root_path, phase):
        path_list = []
        for root, dirs, files in os.walk(root_path + "/" + phase):
            for file in files:
                if file.endswith('.jpg'):
                    path_list.append(root + "/" + file)
        return path_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img_path = self.file_list[index]
        img = Image.open(img_path)

        # preprocessing
        img_transformed = self.transform(img, self.phase) if self.transform is not None else transforms.ToTensor()(img)

        # get image label from file name
        arr = img_path.split('/')
        label = arr[-2]
        name = label2name[label]

        # transform label to number
        label_num = name2id[name]

        return img_transformed, label_num
