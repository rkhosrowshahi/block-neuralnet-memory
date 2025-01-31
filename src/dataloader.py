import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset


def get_balanced_subset(dataset, num_samples, num_classes):
    samples_per_class = num_samples // num_classes
    if num_samples % num_classes > 0:
        samples_per_class += 1
    # Create an empty list to store the balanced dataset
    balanced_indices = []
    # Randomly select samples from each class for the training dataset
    for i in range(num_classes):
        class_indices = np.where(np.array(dataset.targets) == i)[0]
        selected_indices = np.random.choice(
            class_indices, samples_per_class, replace=False
        )
        balanced_indices.append(selected_indices)
    return np.asarray(balanced_indices).astype(int)


def get_val_test_dataloader(dataset):
    val_set, test_set, num_classes = None, None, None
    if dataset == "cifar10":
        num_classes = 10
        val_num_samples = 1000
        test_num_samples = 9000
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        testset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform_test
        )
        balanced_indices = get_balanced_subset(
            testset, val_num_samples + test_num_samples, num_classes
        )
        val_set = Subset(
            testset, balanced_indices[:, : (val_num_samples // num_classes)].flatten()
        )
        test_set = Subset(
            testset, balanced_indices[:, (val_num_samples // num_classes) :].flatten()
        )

    elif dataset == "cifar100":
        num_classes = 100
        val_num_samples = 1000
        test_num_samples = 9000
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5071, 0.4865, 0.4409],
                    std=[0.2673, 0.2564, 0.2762],
                ),
            ]
        )
        testset = torchvision.datasets.CIFAR100(
            root="./data", train=False, download=True, transform=transform_test
        )
        balanced_indices = get_balanced_subset(
            testset, val_num_samples + test_num_samples, num_classes
        )
        val_set = Subset(
            testset, balanced_indices[:, : (val_num_samples // num_classes)].flatten()
        )
        test_set = Subset(
            testset, balanced_indices[:, (val_num_samples // num_classes) :].flatten()
        )

    elif dataset == "imagenet":
        num_classes = 1000
        val_num_samples = 1000
        test_num_samples = 1000
        transform_test = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        testset = torchvision.datasets.ImageNet(
            root="./data/imagenet-1000",
            split="val",
            transform=transform_test,
        )
        balanced_indices = get_balanced_subset(
            testset, val_num_samples + test_num_samples, num_classes
        )

        val_set = Subset(
            testset, balanced_indices[:, : (val_num_samples // num_classes)].flatten()
        )
        test_set = Subset(
            testset, balanced_indices[:, (val_num_samples // num_classes) :].flatten()
        )

    val_loader = DataLoader(val_set, batch_size=len(val_set), shuffle=False)
    test_loader = DataLoader(test_set, batch_size=1000, shuffle=False)

    return val_loader, test_loader, num_classes
