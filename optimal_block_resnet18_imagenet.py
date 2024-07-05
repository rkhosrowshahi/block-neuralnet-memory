import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18

import matplotlib.pyplot as plt
import pandas as pd
from src.block import MultiObjOptimalBlockOptimzationProblem
from src.utils import f1score_func, get_model_params

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize


if __name__ == "__main__":
    os.makedirs("./out/nsga2_resnet18_imagenet/codebooks", exist_ok=True)

    seed = 1
    torch.manual_seed(seed)
    np.random.seed(seed=seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    num_classes = 1000

    # transform_train = transforms.Compose(
    #     [
    #         transforms.Resize(256),
    #         transforms.CenterCrop(224),
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    #     ]
    # )
    # trainset = torchvision.datasets.ImageNet(
    #     root="./data/imagenet-1000",
    #     split="train",
    #     transform=transform_train,
    # )
    transform_test = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    testset = torchvision.datasets.ImageNet(
        root="./data/imagenet-1000",
        split="val",
        transform=transform_test,
    )
    print(len(testset))
    # train_loader = DataLoader(trainset, batch_size=256, shuffle=True)
    num_samples = 2000
    samples_per_class = num_samples // num_classes
    if num_samples % num_classes > 0:
        samples_per_class += 1
    # Create an empty list to store the balanced dataset
    balanced_indices = []
    # Randomly select samples from each class for the training dataset
    for i in range(num_classes):
        class_indices = np.where(np.array(testset.targets) == i)[0]
        selected_indices = np.random.choice(
            class_indices, samples_per_class, replace=False
        )
        balanced_indices.extend(selected_indices)
    print(len(balanced_indices))
    # Create a subset of the original dataset using the balanced indices
    balanced_dataset = Subset(testset, balanced_indices)
    val_loader = DataLoader(balanced_dataset, batch_size=2000, shuffle=False)
    print(len(val_loader))
    test_loader = DataLoader(testset, batch_size=2000, shuffle=False)
    print(len(test_loader))

    model = resnet18(weights="DEFAULT")
    problem_name = "ResNet18"
    model.to(device)

    params = get_model_params(model)

    orig_dims = len(params)

    gb_f = f1score_func(model, val_loader, num_classes, device)
    gb_test_f = f1score_func(model, test_loader, num_classes, device)

    hist_file_path = f"./out/nsga2_resnet18_imagenet"

    df = pd.DataFrame(
        {
            "B_max": [len(params)],
            "B_max_f1": [gb_f],
            "B_max_test_f1": [gb_test_f],
            "B_opt": [len(params)],
            "B_opt_f1": [gb_f],
            "B_opt_test_f1": [gb_test_f],
        }
    )
    df.to_csv(hist_file_path + "/hist_table.csv", index=False)

    problem = MultiObjOptimalBlockOptimzationProblem(
        params=params,
        model=model,
        evaluation=f1score_func,
        data_loader=val_loader,
        test_loader=test_loader,
        num_classes=num_classes,
        device=device,
        hist_file_path=hist_file_path,
    )

    init_pop = np.column_stack([np.random.randint(16, 256, size=100) for k in range(1)])

    algorithm = NSGA2(pop_size=100, sampling=init_pop)

    res = minimize(
        problem,
        algorithm,
        ("n_gen", 5),
        seed=1,
        verbose=True,
    )

    arg_sorted = np.argsort(res.F[:, 1])

    df = pd.DataFrame(
        {
            "B_max": res.X[arg_sorted, 0].astype(int),
            "B_opt": res.F[arg_sorted, 1],
            "B_opt F1 err": res.F[arg_sorted, 0],
        }
    )

    df.to_csv("./out/nsga2_resnet18_imagenet/paretofront.csv", index=False)

    plt.axhline(y=1 - gb_f, color="r", linestyle="-.", label="Baseline")
    plt.plot(
        df["B_opt"].to_numpy(),
        1 - df["B_opt F1 err"].to_numpy(),
        label="Pareto Frontier",
        marker="o",
        linestyle="--",
    )

    plt.grid()
    plt.xlabel("Optimal blocked dimensions")
    plt.ylabel("Optimal blocked F1-score")
    plt.legend()

    plt.savefig("./out/nsga2_resnet18_imagenet/paretofront.pdf")
