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

    def load_transformed_dataset(self, dataset_path='./dataset'):
        data_transform = transforms.Compose([
            transforms.Resize((self.IMG_SIZE, self.IMG_SIZE)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),  # convert the images to tensor. Scales to [0, 1]
            transforms.Lambda(lambda t: (t * 2) - 1)  # Scaled b/w [-1, 1]
        ])
        # dataset = torchvision.datasets.ImageFolder(
        #     root=dataset_path, transform=data_transform)
        dataset = torchvision.datasets.MNIST(root=dataset_path, download=True, transform=data_transform)
        indices = torch.randperm(len(dataset))[:self.NUM_IMG]
        subset = Subset(dataset, indices)  # take only a subset of the dataset
        return DataLoader(subset, batch_size=self.BATCH_SIZE, shuffle=True)

    def show_tensor_image(self, img_tensor, save=False, output_dir=None):
        reverse_transforms = transforms.Compose([
            transforms.Lambda(lambda t: (t + 1) / 2),  # [-1, 1] -> [0, 1]
            transforms.Lambda(lambda t: t.permute(1, 2, 0)
                              ),  # (C, H, W) -> (H, W, C)
            transforms.Lambda(lambda t: t * 255.0),   # [0, 1] -> [0, 255]
            transforms.Lambda(lambda t: t.to('cpu').numpy().astype(np.uint8)),
            transforms.ToPILImage()
        ])

        # Take only first image of a batch
        if len(img_tensor.shape) == 4:
            img_tensor = img_tensor[0, :, :, :]
        image = reverse_transforms(img_tensor)
        if save and output_dir is not None:
            image.save(output_dir, format='png')
        plt.imshow(image)
