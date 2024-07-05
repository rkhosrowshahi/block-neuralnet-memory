import os
import numpy as np
import torch
from torch.utils.data import DataLoader
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
    os.makedirs("./out/nsga2_resnet18_cifar10_100steps/codebooks", exist_ok=True)

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

    # train_loader = DataLoader(trainset, batch_size=256, shuffle=True)
    val_loader = DataLoader(testset, batch_size=256, shuffle=False)
    test_loader = DataLoader(testset, batch_size=10000, shuffle=False)

    model = resnet18(num_classes=num_classes)
    problem_name = "ResNet18"
    model.to(device)

    model.load_state_dict(
        torch.load("./models/resnet18_cifar10_adam_100steps_params.pt")
    )

    params = get_model_params(model)

    orig_dims = len(params)

    gb_f = f1score_func(model, val_loader, num_classes, device)
    gb_test_f = f1score_func(model, test_loader, num_classes, device)

    hist_file_path = f"./out/nsga2_resnet18_cifar10_100steps"

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

    df.to_csv("./out/nsga2_resnet18_cifar10_100steps/paretofront.csv", index=False)

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

    plt.savefig("./out/nsga2_resnet18_cifar10_100steps/paretofront.pdf")
