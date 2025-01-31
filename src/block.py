import os
import pickle
import numpy as np
from pymoo.core.problem import Problem
import pandas as pd

from src.utils import set_model_state

from tqdm import tqdm


def blocker(params, codebook):
    blocked_params = []
    for block_idx, indices in (codebook).items():
        blocked_params.append(params[indices].mean())

    return np.array(blocked_params)


# from concurrent.futures import ThreadPoolExecutor


# def update_unblocked_params(block_idx, indices, unblocked_params, blocked_params):
#     unblocked_params[indices] = np.full(len(indices), blocked_params[block_idx])


# def unblocker_mp(codebook, orig_dims, blocked_params):
#     unblocked_params = np.zeros(orig_dims)
#     with ThreadPoolExecutor() as executor:
#         futures = []
#         for block_idx, indices in codebook.items():
#             futures.append(
#                 executor.submit(
#                     update_unblocked_params,
#                     block_idx,
#                     indices,
#                     unblocked_params,
#                     blocked_params,
#                 )
#             )

#         for future in futures:
#             future.result()  # Wait for all futures to complete

#     return unblocked_params


def unblocker(codebook, orig_dims, blocked_params, verbose=False):

    unblocked_params = np.zeros(orig_dims)
    # start_time = time.time()
    for block_idx, indices in tqdm(
        codebook.items(),
        desc=f"Unblocking D= {len(blocked_params)} ==> {orig_dims}",
        disable=not verbose,
    ):
        # st_in = time.time()
        unblocked_params[indices] = np.full(len(indices), blocked_params[block_idx])
        # tot_in = time.time() - st_in
        # print(tot_in)

    # end_time = time.time() - start_time

    # print(end_time)
    return unblocked_params


class MultiObjOptimalBlockOptimzationProblem(Problem):
    def __init__(
        self,
        n_var=1,
        xl=2,
        xu=1024,
        params=None,
        model=None,
        evaluation=None,
        data_loader=None,
        test_loader=None,
        num_classes=None,
        device=None,
        res_path=None,
        hist_file_path=None,
        merge=True,
    ):
        super().__init__(
            n_var=n_var,
            n_obj=2,
            n_ieq_constr=0,
            xl=xl,
            xu=xu,
            vtype=int,
        )
        self.model = model
        self.params = params
        self.orig_dims = len(params)
        self.evaluation = evaluation
        self.blocker = blocker
        self.unblocker = unblocker
        self.data_loader = data_loader
        self.test_loader = test_loader
        self.num_classes = num_classes
        self.device = device
        self.res_path = res_path
        self.hist_file_path = hist_file_path
        self.dataframe = pd.read_csv(hist_file_path)
        self.merge = merge

    def _calc_pareto_front(self, n_pareto_points=100):
        x = np.linspace(0, 1, n_pareto_points)
        return np.ones((n_pareto_points, 2))

    def _hist_block(self, B_max):
        bin_edges = np.linspace(np.min(self.params), np.max(self.params), B_max)
        # Split the data into bins
        binned_data = np.digitize(self.params, bin_edges) - 1

        blocks_arrs = [np.array([])] * B_max

        # for i in tqdm(range(self.orig_dims)):
        #     # if i % 1000000 == 0:
        #     #     print(i, self.orig_dims)
        #     b = binned_data[i]
        #     blocks_arrs[b] = np.concatenate([blocks_arrs[b], [i]])

        histogram_block_codebook = {}
        # histogram_block_codebook_size = {}
        nonempty_bins_i = 0

        # for i in range(B_max):
        #     if len(blocks_arrs[i]) > 0:
        #         histogram_block_codebook[nonempty_bins_i] = blocks_arrs[i].tolist()
        #         # histogram_block_codebook_size[i] = len(blocks_arrs[i])
        #         nonempty_bins_i += 1

        for i in tqdm(range(B_max), desc=f"Histogram K={B_max}"):
            b_i = np.where(binned_data == i)[0]
            if len(b_i) > 0:
                histogram_block_codebook[nonempty_bins_i] = (b_i).tolist()
                # histogram_block_codebook_size[i] = len(b_i)
                nonempty_bins_i += 1

        return histogram_block_codebook

    def _merge_till_noimprv(self, histogram_block_codebook, best_f, B_max):

        new_dims = len(histogram_block_codebook)
        codebook = histogram_block_codebook.copy()
        mask_size = new_dims
        i = 1
        with tqdm(total=mask_size - 2) as pbar:
            while i < mask_size - 1:
                # Left
                left_mrg_codebook = codebook.copy()
                left_mrg_codebook[i - 1] = (
                    left_mrg_codebook[i - 1] + left_mrg_codebook[i]
                )

                for key in range(i, len(left_mrg_codebook) - 1):
                    left_mrg_codebook[key] = left_mrg_codebook[key + 1]

                del left_mrg_codebook[mask_size - 1]

                left_merged_params = self.unblocker(
                    left_mrg_codebook,
                    self.orig_dims,
                    blocked_params=self.blocker(self.params, left_mrg_codebook),
                )

                model = set_model_state(model=self.model, parameters=left_merged_params)
                left_f = self.evaluation(
                    model,
                    data_loader=self.data_loader,
                    num_classes=self.num_classes,
                    device=self.device,
                )
                # Right
                right_mrg_codebook = codebook.copy()
                right_mrg_codebook[i] = (
                    right_mrg_codebook[i] + right_mrg_codebook[i + 1]
                )

                for key in range(i + 1, len(right_mrg_codebook) - 1):
                    right_mrg_codebook[key] = right_mrg_codebook[key + 1]

                del right_mrg_codebook[mask_size - 1]

                right_merged_params = self.unblocker(
                    right_mrg_codebook,
                    self.orig_dims,
                    blocked_params=self.blocker(self.params, right_mrg_codebook),
                )

                model = set_model_state(
                    model=self.model, parameters=right_merged_params
                )
                right_f = self.evaluation(
                    model,
                    data_loader=self.data_loader,
                    num_classes=self.num_classes,
                    device=self.device,
                )

                # print(
                #     f"B_max:{B_max}, current B_opt:{len(right_mrg_codebook)}, D_i:{i}/{new_dims}",
                #     f" | curr best f1: {best_f:.9f}, L f1: {left_f:.9f}, R f1: {right_f:.9f}",
                # )

                argmax_f = np.argmin([left_f, right_f, best_f])
                if argmax_f == 0:
                    codebook = left_mrg_codebook
                    best_f = left_f
                    mask_size -= 1
                    # i -= 1
                elif argmax_f == 1:
                    codebook = right_mrg_codebook
                    best_f = right_f
                    mask_size -= 1
                    # i -= 1
                else:
                    i += 1
                    pbar.update(1)
        return codebook, best_f

    def _evaluate(self, X, out, *args, **kwargs):

        NP = X.shape[0]
        f1 = np.zeros(NP)
        f2 = np.zeros(NP)
        for si, B_max in enumerate(X):
            B_max = int(B_max[0])
            xopt_codebook, xopt_f = None, None
            if self.merge == True:
                xhist_codebook = self._hist_block(B_max)
                un_params = self.unblocker(
                    xhist_codebook,
                    self.orig_dims,
                    blocked_params=self.blocker(self.params, xhist_codebook),
                )
                model = set_model_state(model=self.model, parameters=un_params)

                xhist_f = self.evaluation(
                    model,
                    data_loader=self.data_loader,
                    num_classes=self.num_classes,
                    device=self.device,
                )
                xhist_test_f = self.evaluation(
                    model,
                    data_loader=self.test_loader,
                    num_classes=self.num_classes,
                    device=self.device,
                    mode="test",
                )

                x_path = f"{self.res_path}/codebooks/merged_codebook_bmax_{B_max}.pkl"
                if os.path.exists(x_path):
                    with open(
                        x_path,
                        "rb",
                    ) as f:
                        xopt_codebook = pickle.load(f)
                else:

                    xopt_codebook, xopt_f = self._merge_till_noimprv(
                        xhist_codebook, xhist_f, B_max
                    )

                un_params = self.unblocker(
                    xopt_codebook,
                    self.orig_dims,
                    blocked_params=self.blocker(self.params, xopt_codebook),
                )
                model = set_model_state(model=self.model, parameters=un_params)
                if xopt_f is None:
                    xopt_f = self.evaluation(
                        model,
                        data_loader=self.data_loader,
                        num_classes=self.num_classes,
                        device=self.device,
                    )
                xopt_test_f = self.evaluation(
                    model,
                    data_loader=self.test_loader,
                    num_classes=self.num_classes,
                    device=self.device,
                    mode="test",
                )

                x_path = f"{self.res_path}/codebooks/merged_codebook_bmax_{B_max}.pkl"
                with open(
                    x_path,
                    "wb",
                ) as f:
                    pickle.dump(xopt_codebook, f)

                B_opt = len(xopt_codebook)
                f1[si] = xopt_f
                f2[si] = B_opt

                new_row = pd.DataFrame(
                    {
                        "B_max": [B_max],
                        "B_max_f1": [xhist_f],
                        "B_max_test_f1": [xhist_test_f],
                        "B_opt": [B_opt],
                        "B_opt_f1": [xopt_f],
                        "B_opt_test_f1": [xopt_test_f],
                    }
                )
                self.dataframe = pd.concat([self.dataframe, new_row])
                self.dataframe.to_csv(self.hist_file_path, index=False)
            else:

                if sum(self.dataframe["B_max"] == B_max) > 0:
                    results = self.dataframe.loc[self.dataframe["B_max"] == B_max]

                    f1[si] = float(results["B_opt_f1"].iloc[0])

                    f2[si] = float(results["B_opt"].iloc[0])
                    continue

                xhist_codebook = None
                x_path = f"{self.res_path}/codebooks/codebook_bmax_{B_max}.pkl"
                if os.path.exists(x_path):
                    with open(
                        x_path,
                        "rb",
                    ) as f:
                        xhist_codebook = pickle.load(f)
                else:
                    xhist_codebook = self._hist_block(B_max)

                un_params = self.unblocker(
                    xhist_codebook,
                    self.orig_dims,
                    blocked_params=self.blocker(self.params, xhist_codebook),
                )
                model = set_model_state(model=self.model, parameters=un_params)
                xhist_f = self.evaluation(
                    model,
                    data_loader=self.data_loader,
                    num_classes=self.num_classes,
                    device=self.device,
                )

                xhist_test_f = self.evaluation(
                    model,
                    data_loader=self.test_loader,
                    num_classes=self.num_classes,
                    device=self.device,
                    mode="test",
                )

                B_opt = len(xhist_codebook)
                f1[si] = xhist_f
                f2[si] = B_opt

                new_row = pd.DataFrame(
                    {
                        "B_max": [B_max],
                        "B_max_f1": [xhist_f],
                        "B_max_test_f1": [xhist_test_f],
                        "B_opt": [B_opt],
                        "B_opt_f1": [xhist_f],
                        "B_opt_test_f1": [xhist_test_f],
                    }
                )
                self.dataframe = pd.concat([self.dataframe, new_row])
                self.dataframe.to_csv(self.hist_file_path, index=False)

        out["F"] = np.column_stack([f1, f2])
        return out
