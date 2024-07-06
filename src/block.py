import os
import pickle
import numpy as np
from pymoo.core.problem import Problem
import pandas as pd

from src.utils import set_model_state


def blocker(params, codebook):
    blocked_params = []
    for block_idx, indices in (codebook).items():
        blocked_params.append(params[indices].mean())

    return np.array(blocked_params)


from concurrent.futures import ThreadPoolExecutor


def update_unblocked_params(block_idx, indices, unblocked_params, blocked_params):
    unblocked_params[indices] = np.full(len(indices), blocked_params[block_idx])


def unblocker(codebook, orig_dims, blocked_params):
    unblocked_params = np.zeros(orig_dims)
    with ThreadPoolExecutor() as executor:
        futures = []
        for block_idx, indices in codebook.items():
            futures.append(
                executor.submit(
                    update_unblocked_params,
                    block_idx,
                    indices,
                    unblocked_params,
                    blocked_params,
                )
            )

        for future in futures:
            future.result()  # Wait for all futures to complete

    return unblocked_params


class MultiObjOptimalBlockOptimzationProblem(Problem):
    def __init__(
        self,
        n_var=1,
        xl=10,
        xu=256,
        params=None,
        model=None,
        evaluation=None,
        data_loader=None,
        test_loader=None,
        num_classes=None,
        device=None,
        hist_file_path=None,
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
        self.hist_file_path = hist_file_path
        self.dataframe = pd.read_csv(hist_file_path + "/hist_table.csv")

    def _calc_pareto_front(self, n_pareto_points=100):
        x = np.linspace(0, 1, n_pareto_points)
        return np.ones((n_pareto_points, 2))

    def _hist_block(self, B_max):
        bin_edges = np.linspace(np.min(self.params), np.max(self.params), B_max)
        # Split the data into bins
        binned_data = np.digitize(self.params, bin_edges)

        histogram_block_codebook = {}
        histogram_block_codebook_size = {}
        nonempty_bins_i = 0

        for i in range(B_max):
            b_i = np.where(binned_data == i)[0]
            if len(b_i) != 0:
                histogram_block_codebook[nonempty_bins_i] = (b_i).tolist()
                histogram_block_codebook_size[i] = len(b_i)
                nonempty_bins_i += 1

        return histogram_block_codebook

    def _merge_till_noimprv(self, histogram_block_codebook, best_f, B_max):

        new_dims = len(histogram_block_codebook)
        codebook = histogram_block_codebook.copy()
        mask_size = new_dims
        i = 1
        while i < mask_size - 1:
            # Left
            left_mrg_codebook = codebook.copy()
            left_mrg_codebook[i - 1] = left_mrg_codebook[i - 1] + left_mrg_codebook[i]
            # del left_mrg_codebook[i]

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
            right_mrg_codebook[i] = right_mrg_codebook[i] + right_mrg_codebook[i + 1]
            # del right_mrg_codebook[i+1]

            for key in range(i + 1, len(right_mrg_codebook) - 1):
                right_mrg_codebook[key] = right_mrg_codebook[key + 1]

            del right_mrg_codebook[mask_size - 1]

            right_merged_params = self.unblocker(
                right_mrg_codebook,
                self.orig_dims,
                blocked_params=self.blocker(self.params, right_mrg_codebook),
            )

            model = set_model_state(model=self.model, parameters=right_merged_params)
            right_f = self.evaluation(
                model,
                data_loader=self.data_loader,
                num_classes=self.num_classes,
                device=self.device,
            )

            print(
                f"B_max:{B_max}, current B_opt:{len(right_mrg_codebook)}, D_i:{i}/{new_dims}",
                f" | curr best f1: {best_f:.9f}, L f1: {left_f:.9f}, R f1: {right_f:.9f}",
            )

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
        return codebook, best_f

    def _evaluate(self, X, out, *args, **kwargs):

        NP = X.shape[0]
        f1 = np.zeros(NP)
        f2 = np.zeros(NP)
        for si, B_max in enumerate(X):
            B_max = int(B_max[0])
            if B_max in self.dataframe["B_max"]:
                results = self.dataframe.loc[self.dataframe["B_max"] == B_max]

                f1[si] = results["B_opt_f1"]
                f2[si] = results["B_opt"]
                continue

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

            x_path = f"{self.hist_file_path}/codebooks/codebook_bmax_{B_max}.pkl"
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
            # print(self.dataframe)
            self.dataframe.to_csv(self.hist_file_path + "/hist_table.csv", index=False)

            x_path = f"{self.hist_file_path}/codebooks/codebook_bmax_{B_max}.pkl"
            with open(
                x_path,
                "wb",
            ) as f:
                pickle.dump(xopt_codebook, f)

        out["F"] = np.column_stack([f1, f2])
        return out
