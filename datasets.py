import os
import torch.utils.data as data
from PIL import Image

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class ISIC(data.Dataset):
    base_folder = os.curdir
    def __init__(self, root, train=False, valid=False, test=False,
                 transform=None, target_transform=None):
        super(ISIC, self).__init__()
        self.base_folder = os.path.join(self.base_folder,root)
        self.train = train
        self.valid = valid
        self.test = test
        self.transform = transform
        self.target_transform = target_transform

        if not (((train ^ valid ^ test) ^ (train & valid & test))):
            raise ValueError('One and only one of `train`, `valid` or `test` '
                'must be True (train={0}, valid={1}, test={2}).'.format(train,
                valid, test))

        if train:
            self.image_folder = os.path.join(self.base_folder, "Train_data")
        elif valid or test:
            self.image_folder = os.path.join(self.base_folder, "Test_data")
            
        else:
            raise ValueError('Unknown split.')
        
        if not self._check_exists():
            raise RuntimeError(f'Dataset not found at {self.image_folder}')
        self._data = os.listdir(self.image_folder)        

    def __getitem__(self, index):
        filename = self._data[index]
        image = pil_loader(os.path.join(self.image_folder, filename))
        if self.transform is not None:
            image = self.transform(image)

        return image, 0

    def _check_exists(self):
        return (os.path.exists(self.image_folder) )
  
    def __len__(self):
        return len(self._data)