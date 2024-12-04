import torch
import torch.utils
import torch.utils.data
from torch.utils.data import DataLoader, Subset
import torchvision
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt


class DataClass:

    def __init__(self, batch_size, img_size, num_img):
        self.BATCH_SIZE = batch_size
        self.IMG_SIZE = img_size
        self.NUM_IMG = num_img

    def load_transformed_dataset(self):
        data_transforms = [
            transforms.Resize((self.IMG_SIZE, self.IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            # convert the images to tensor. Scales to [0, 1]
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)  # Scaled b/w [-1, 1]
        ]
        data_transform = transforms.Compose(data_transforms)
        dataset = torchvision.datasets.ImageFolder(root='./dataset', transform=data_transform)
        indices = torch.randperm(len(dataset))[:self.NUM_IMG]
        subset = Subset(dataset, indices)  # take only a subset of the dataset
        return DataLoader(subset, batch_size=self.BATCH_SIZE, shuffle=True)

    def show_tensor_image(self, image):
        reverse_transforms = transforms.Compose([
            transforms.Lambda(lambda t: (t + 1) / 2),  # [-1, 1] -> [0, 1]
            transforms.Lambda(lambda t: t.permute(1, 2, 0)),  # (C, H, W) -> (H, W, C)
            transforms.Lambda(lambda t: t * 255.0),   # [0, 1] -> [0, 255]
            transforms.Lambda(lambda t: t.to('cpu').numpy().astype(np.uint8)),
            transforms.ToPILImage()
        ])

        # Take only first image of a batch
        if len(image.shape) == 4:
            image = image[0, :, :, :]
        plt.imshow(reverse_transforms(image))
