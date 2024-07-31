import os
import sys
import argparse
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from src.block import MultiObjOptimalBlockOptimzationProblem
from src.dataloader import get_val_test_dataloader
from src.utils import f1score_func, get_model_params, get_network, set_seed

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

import matplotlib.pyplot as plt
import scienceplots

plt.style.use(["science", "ieee", "no-latex"])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--net", type=str, required=True, help="network type")
    parser.add_argument(
        "--model_path", type=str, required=True, help="model checkpoint path"
    )
    parser.add_argument(
        "--save_dir", type=str, required=True, help="model checkpoint path"
    )
    parser.add_argument(
        "--gpu", type=str, default="cuda:0", help="use cuda:[number1], cuda:[number2]"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="imagenet",
        help="use cifar10, cifar100 or imagenet",
    )
    parser.add_argument("--b", type=int, default=128, help="batch size for dataloader")
    parser.add_argument(
        "--seed", type=int, default=1, help="seed value for random values"
    )
    parser.add_argument(
        "--lb", type=int, default=2, help="lower bound in population initilization"
    )
    parser.add_argument(
        "--ub", type=int, default=1000, help="upper bound in population initilization"
    )
    # parser.add_argument('--resume', action='store_true', default=False, help='resume training')
    args = parser.parse_args()

    device = torch.device(args.gpu if torch.cuda.is_available() else "cpu")

    set_seed(args.seed)

    val_loader, test_loader, num_classes = get_val_test_dataloader(dataset=args.dataset)

    model = get_network(args.net, args.dataset, args.model_path)
    model.to(device)
    params = get_model_params(model)

    res_path = f"./out/{args.save_dir}"

    df_pf = pd.read_csv(res_path + "/paretofront.csv").drop_duplicates()
    df_hist = pd.read_csv(res_path + "/hist_table.csv")

    df_merge_hist = df_hist.iloc[0].to_frame().T
    df_merge_hist.to_csv(res_path + "/hist_merge_table.csv", index=False)

    problem = MultiObjOptimalBlockOptimzationProblem(
        xl=args.lb,
        xu=args.ub,
        params=params,
        model=model,
        evaluation=f1score_func,
        data_loader=val_loader,
        test_loader=test_loader,
        num_classes=num_classes,
        device=device,
        res_path=res_path,
        hist_file_path=res_path + "/hist_merge_table.csv",
        merge=True,
    )

    baseline_trsh = df_merge_hist.iloc[0, df_merge_hist.columns.get_loc("B_opt_f1")]
    init_pop = df_pf[df_pf["B_opt_f1"] < baseline_trsh]["B_max"].to_numpy()
    init_pop.sort()
    print(init_pop)
    init_pop = init_pop.reshape(-1, 1)

    algorithm = NSGA2(pop_size=100, sampling=init_pop, eliminate_duplicates=True)

    plt.axhline(
        y=1 - df_hist.iloc[0]["B_opt_f1"], color="r", linestyle="-.", label="Baseline"
    )

    res = minimize(
        problem,
        algorithm,
        ("n_gen", 1),
        seed=1,
        verbose=True,
    )

    arg_sorted = np.argsort(res.pop.get("F")[:, 1])

    dict_postmerge = {}
    dict_postmerge["B_max"] = init_pop.flatten()
    dict_postmerge["B_merge"] = res.pop.get("F")[:, 1].astype(int)
    dict_postmerge["B_merge_f1"] = res.pop.get("F")[:, 0]
    print(dict_postmerge)

    df_postmerge = pd.DataFrame(dict_postmerge)

    df_postmerge.to_csv(f"./out/{res_path}/paretofront_postmerge.csv", index=False)

    plt.plot(
        df_postmerge["B_merge"].to_numpy(),
        1 - df_postmerge["B_merge_f1"].to_numpy(),
        label="Pareto Frontier",
        marker="o",
        linestyle="--",
    )

    plt.grid()
    plt.xlabel("Optimal blocked dimensions")
    plt.ylabel("Validation F1-score")
    plt.legend()

    plt.savefig(f"./out/{res_path}/paretofront_postmerge.pdf")
