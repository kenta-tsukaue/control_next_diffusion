
import random
import json
from torchvision import transforms
from PIL import Image

from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, resolution=768):
        self.data = []
        self.resolution = resolution
        with open('./training/fill50k/prompt.json', 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        source = Image.open('./training/fill50k/' + source_filename)  # 画像をPIL形式で読み込む
        target = Image.open('./training/fill50k/' + target_filename)  # 画像をPIL形式で読み込む

        image_transforms = transforms.Compose(
        [
            transforms.Resize(self.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
        )

        conditioning_image_transforms = transforms.Compose(
            [
                transforms.Resize(self.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(self.resolution),
                transforms.ToTensor(),
            ]
        )
        pixel_value = image_transforms(target)
        conditioning_pixel_value = conditioning_image_transforms(source)
        
        if random.random() < 0.01:
            prompt = ""

        return dict(pixel_value=pixel_value, conditioning_pixel_value = conditioning_pixel_value, prompt = prompt)
