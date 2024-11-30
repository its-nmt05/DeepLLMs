import torch
import torch.utils
import torch.utils.data
import torchvision
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt


class DataClass:

    def __init__(self, img_size):
        self.IMG_SIZE = img_size

    def load_transformed_dataset(self):
        data_transforms = [
            transforms.Resize((self.IMG_SIZE, self.IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            # convert the images to tensor. Scales to [0, 1]
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)  # Scaled b/w [-1, 1]
        ]

        data_transform = transforms.Compose(data_transforms)

        train = torchvision.datasets.CIFAR10(
            root='.', download=True, transform=data_transform)
        test = torchvision.datasets.CIFAR10(root=".", download=True,
                                                 transform=data_transform, train=False)

        return torch.utils.data.ConcatDataset([train, test])

    def show_tensor_image(self, image):
        reverse_transforms = transforms.Compose([
            transforms.Lambda(lambda t: (t + 1) / 2),  # [-1, 1] -> [0, 1]
            transforms.Lambda(lambda t: t.permute(1, 2, 0)),  # (C, H, W) -> (H, W, C)
            transforms.Lambda(lambda t: t * 255.0),   # [0, 1] -> [0, 255]
            transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
            transforms.ToPILImage()
        ])

        # Take only first image of a batch
        if len(image.shape) == 4:
            image = image[0, :, :, :]
        plt.imshow(reverse_transforms(image))
