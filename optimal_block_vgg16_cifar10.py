import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from torchvision.models import vgg16

import matplotlib.pyplot as plt
import pandas as pd
from src.block import MultiObjOptimalBlockOptimzationProblem
from src.utils import f1score_func, get_model_params

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize


if __name__ == "__main__":
    problem_title = "nsga2_vgg16_cifar10_200steps"
    os.makedirs(f"./out/{problem_title}/codebooks", exist_ok=True)

    seed = 1
    torch.manual_seed(seed)
    np.random.seed(seed=seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    num_classes = 10

    # transform_train = transforms.Compose(
    #     [
    #         transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.RandomRotation(15),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #     ]
    # )
    # trainset = torchvision.datasets.CIFAR10(
    #     root="./data", train=True, download=True, transform=transform_train
    # )
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )

    num_samples = 100
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
    # Create a subset of the original dataset using the balanced indices
    balanced_dataset = Subset(testset, balanced_indices)
    val_loader = DataLoader(balanced_dataset, batch_size=num_samples, shuffle=False)
    test_loader = DataLoader(testset, batch_size=10000, shuffle=False)

    model = vgg16(num_classes=num_classes)
    problem_name = "vgg16"
    model.to(device)

    model.load_state_dict(torch.load("./models/vgg16_cifar10_adam_200steps_params.pt"))

    params = get_model_params(model)

    orig_dims = len(params)

    hist_file_path = f"./out/{problem_title}"

    df = None
    if os.path.exists(hist_file_path + "/hist_table.csv"):
        df = pd.read_csv(hist_file_path + "/hist_table.csv")
    else:
        gb_f = f1score_func(model, val_loader, num_classes, device)
        gb_test_f = f1score_func(model, test_loader, num_classes, device, mode="test")
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
        xl=64,
        xu=256 - 1,
        params=params,
        model=model,
        evaluation=f1score_func,
        data_loader=val_loader,
        test_loader=test_loader,
        num_classes=num_classes,
        device=device,
        hist_file_path=hist_file_path,
    )

    init_pop = np.random.choice(
        np.linspace(problem.xl[0], problem.xu[0], 10, dtype=int), size=10, replace=True
    )
    init_pop.sort()
    print(init_pop)
    init_pop = init_pop.reshape(-1, 1)

    algorithm = NSGA2(pop_size=10, sampling=init_pop, eliminate_duplicates=True)

    res = minimize(
        problem,
        algorithm,
        ("n_gen", 5),
        seed=1,
        verbose=True,
    )

    arg_sorted = np.argsort(res.F[:, 1])

    plt.axhline(
        y=1 - df.iloc[0]["B_opt_f1"], color="r", linestyle="-.", label="Baseline"
    )

    df = None
    if os.path.exists(f"./out/{problem_title}/paretofront.csv"):

        df = pd.read_csv(f"./out/{problem_title}/paretofront.csv")
    else:
        df = pd.DataFrame(
            {
                "B_max": res.X[arg_sorted, 0].astype(int),
                "B_opt": res.F[arg_sorted, 1],
                "B_opt_f1": res.F[arg_sorted, 0],
            }
        )

        df.to_csv(f"./out/{problem_title}/paretofront.csv", index=False)

    plt.plot(
        df["B_opt"].to_numpy(),
        1 - df["B_opt_f1"].to_numpy(),
        label="Pareto Frontier",
        marker="o",
        linestyle="--",
    )

    plt.grid()
    plt.xlabel("Optimal blocked dimensions")
    plt.ylabel("Validation F1-score")
    plt.legend()

    plt.savefig(f"./out/{problem_title }/paretofront.pdf")
