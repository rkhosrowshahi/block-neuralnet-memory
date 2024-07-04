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
    os.makedirs("./out/nsga2_resnet18_cifar100/codebooks", exist_ok=True)

    seed = 1
    torch.manual_seed(seed)
    np.random.seed(seed=seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    num_classes = 100

    transform = transforms.Compose(
        [
            # transforms.Resize(256),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    trainset = torchvision.datasets.CIFAR100(
        root="./data", train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.CIFAR100(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(trainset, batch_size=256, shuffle=True)
    val_loader = DataLoader(testset, batch_size=1024, shuffle=False)
    test_loader = DataLoader(testset, batch_size=10000, shuffle=False)

    model = resnet18(num_classes=num_classes)
    problem_name = "ResNet18"
    model.to(device)

    model.load_state_dict(torch.load("models/resnet18_cifar100_adam_params.pt"))

    params = get_model_params(model)

    orig_dims = len(params)

    gb_f = f1score_func(model, val_loader, num_classes, device)
    gb_test_f = f1score_func(model, test_loader, num_classes, device)

    hist_file_path = f"out/nsga2_resnet18_cifar100"

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

    init_pop = np.random.randint(8, 256, (100, 1))

    algorithm = NSGA2(pop_size=100, sampling=init_pop)

    res = minimize(
        problem,
        algorithm,
        ("n_gen", 3),
        seed=1,
        verbose=True,
    )

    df = pd.DataFrame({"B_max": res.X, "B_opt": res.F[:, 1], "B_opt": res.F[:, 0]})

    df.to_csv("./out/nsga2_resnet18_cifar100/paretofront.csv")
