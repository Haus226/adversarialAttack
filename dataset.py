from torch.utils.data import Dataset
import os
import pandas as pd
import tqdm
from PIL import Image

class ImageCSVDataset(Dataset):
    '''
    return: path, image, label, target_label
    '''
    def __init__(self, csv_file, image_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.data.loc[idx, 'ImageId']) + ".png"
        image = Image.open(image_path).convert('RGB')  # Assuming RGB images
        # print(np.array(image))

        if self.transform:
            image = self.transform(image)
        label = self.data.loc[idx, 'TrueLabel'] - 1
        target_label = self.data.loc[idx, 'TargetClass'] - 1

        return image_path, image, label, target_label