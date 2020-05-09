import kaggle
from torch.utils.data import Dataset


class MonkeyDataset(Dataset):

    def __init__(self, path, download=False):
        super(MonkeyDataset, self).__init__()
        if download:
            kaggle.api.authenticate()
            kaggle.api.dataset_download_files('slothkong/10-monkey-species', path=path, unzip=True)

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass
