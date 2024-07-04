import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18

from src.traingb import train


if __name__ == "__main__":
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./out", exist_ok=True)

    seed = 1
    torch.manual_seed(seed)
    np.random.seed(seed=seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    num_classes = 10

    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )

    train_loader = DataLoader(trainset, batch_size=256, shuffle=True)
    val_loader = DataLoader(testset, batch_size=256, shuffle=False)
    test_loader = DataLoader(testset, batch_size=10000, shuffle=False)

    model = resnet18(num_classes=num_classes)
    problem_name = "ResNet18"
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(
    #     model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4
    # )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

    model, hist_dict = train(
        model,
        train_loader,
        test_loader,
        num_classes,
        criterion,
        optimizer,
        num_epochs=100,
        device=device,
    )

    torch.save(model.state_dict(), "./models/resnet18_cifar10_adam_200steps_params.pt")

    df = pd.DataFrame(hist_dict)
    df.to_csv("./out/resnet18_cifar10_adam_200steps_hist.csv")
