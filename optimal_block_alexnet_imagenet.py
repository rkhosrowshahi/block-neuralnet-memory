import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from torchvision.models import alexnet
import pandas as pd
from src.block import MultiObjOptimalBlockOptimzationProblem
from src.utils import f1score_func, get_model_params

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

import matplotlib.pyplot as plt
import scienceplots

plt.style.use(["science", "ieee", "no-latex"])

if __name__ == "__main__":
    problem_title = "nsga2_alexnet_imagenet_5000data"
    os.makedirs(f"./out/{problem_title}/codebooks", exist_ok=True)

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
    num_samples, num_samples_test = 5000, 10000
    samples_per_class = num_samples // num_classes
    if num_samples % num_classes > 0:
        samples_per_class += 1
    samples_per_class_test = num_samples_test // num_classes
    if num_samples % num_classes > 0:
        samples_per_class_test += 1
    # Create an empty list to store the balanced dataset
    balanced_indices, balanced_indices_test = [], []
    # Randomly select samples from each class for the training dataset
    for i in range(num_classes):
        class_indices = np.where(np.array(testset.targets) == i)[0]
        selected_indices = np.random.choice(
            class_indices, samples_per_class + samples_per_class_test, replace=False
        )
        balanced_indices.extend(selected_indices[:samples_per_class])
        balanced_indices_test.extend(selected_indices[samples_per_class:])
    # Create a subset of the original dataset using the balanced indices
    balanced_dataset = Subset(testset, balanced_indices)
    val_loader = DataLoader(balanced_dataset, batch_size=num_samples, shuffle=False)
    print(len(val_loader))
    balanced_dataset_test = Subset(testset, balanced_indices_test)
    test_loader = DataLoader(balanced_dataset_test, batch_size=1024, shuffle=False)
    print(len(test_loader))

    model = alexnet(weights="DEFAULT")
    problem_name = "alexnet"
    model.to(device)

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
        xl=100,
        xu=2000 - 1,
        params=params,
        model=model,
        evaluation=f1score_func,
        data_loader=val_loader,
        test_loader=test_loader,
        num_classes=num_classes,
        device=device,
        hist_file_path=hist_file_path,
        merge=False,
    )

    init_pop = init_pop = np.linspace(problem.xl[0], problem.xu[0], 100, dtype=int)
    init_pop.sort()
    print(init_pop)
    init_pop = init_pop.reshape(-1, 1)

    algorithm = NSGA2(pop_size=100, sampling=init_pop, eliminate_duplicates=True)

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
