import os
import random
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_video
from torchvision.transforms import Compose, Resize, Normalize, ToTensor


class CustomDataset(Dataset):
    def __init__(self, config, device, transform=None):
        self.directory = config.data_path
        self.transform = transform
        self.videos = self._load_video_paths()
        self.device = device

    def _load_video_paths(self):
        # フォルダ内のすべての動画ファイルのパスを取得します。
        video_paths = []
        for category in os.listdir(self.directory):
            category_path = os.path.join(self.directory, category)
            for video_folder in os.listdir(category_path):
                #print(video_folder)
                video_folder_path = os.path.join(category_path, video_folder)
                if video_folder == "Annotation":
                    continue
                for video in os.listdir(video_folder_path):
                    if video.endswith('.avi'):
                        video_paths.append(os.path.join(video_folder_path, video))
        #print(video_paths[0])
        return video_paths

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video_path = self.videos[idx]
        video, _, _ = read_video(video_path, pts_unit='sec')
        video = video.float() / 255.0  # scaling

        # Selecting random frame
        if len(video) > 1:
            random_frame_index = random.randint(0, len(video) - 3)
        else:
            random_frame_index = 0

        frame1 = video[random_frame_index]
        frame2 = video[random_frame_index + 1]

        # Convert frame to (C, H, W) format
        frame1 = frame1.permute(2, 0, 1)
        frame2 = frame2.permute(2, 0, 1)

        # Cropping to square and applying transform
        cropped_frame1 = self.crop_to_square(frame1)
        cropped_frame2 = self.crop_to_square(frame2)

        if self.transform:
            cropped_frame1 = self.transform(cropped_frame1).to(device=self.device)
            cropped_frame2 = self.transform(cropped_frame2).to(device=self.device)

        return cropped_frame1, cropped_frame2
    

    def crop_to_square(self, frame):
        # Assuming frame is a PyTorch tensor of shape [C, H, W]
        _, height, width = frame.shape
        min_dim = min(height, width)
        start_x = (width - min_dim) // 2
        start_y = (height - min_dim) // 2
        return frame[:, start_y:start_y + min_dim, start_x:start_x + min_dim]