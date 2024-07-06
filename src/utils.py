import numpy as np
import torch
from torcheval.metrics.functional import multiclass_f1_score
from sklearn.metrics import top_k_accuracy_score, f1_score


def set_model_state(model, parameters):
    state = model.state_dict()
    counted_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            state[name] = torch.tensor(
                parameters[counted_params : param.size().numel() + counted_params]
            ).reshape(param.size())
            counted_params += param.size().numel()

    model.load_state_dict(state)

    return model


def get_model_params(model):
    params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            params.append(torch.flatten(param).cpu().detach().numpy())

    return np.concatenate(params)


def f1score_func(model, data_loader, num_classes, device, mode="val"):
    model.eval()
    fitness = 0
    all_outputs, all_labels = (
        torch.Tensor([]).to(device),
        torch.Tensor([]).to(device),
    )

    # all_outputs, all_labels = (
    #     [],
    #     [],
    # )

    with torch.no_grad():
        for data, labels in data_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, preds = torch.max(outputs, 1)
            # all_outputs.extend(preds.cpu().detach())
            # all_labels.extend(labels.cpu().detach())

            all_outputs = torch.cat((all_outputs, preds))
            all_labels = torch.cat((all_labels, labels))
            # print(outputs.shape, all_outputs.shape, labels.shape, all_labels.shape)

            # fitness += f1_score(
            #     y_true=target.cpu().numpy(), y_pred=pred.cpu().numpy(), average="macro"
            # )

            # fitness += multiclass_f1_score(
            #         outputs, labels, average="weighted", num_classes=num_classes
            #     ).item()

            # f = top_k_accuracy_score(
            #     y_true=labels.cpu().numpy(),
            #     y_score=outputs.cpu().detach().numpy(),
            #     k=1,
            #     labels=np.arange(num_classes),
            # )
            if mode == "val":
                break

    fitness = f1_score(
        all_outputs.cpu().detach(),
        all_labels.cpu().detach(),
        average="macro",
        labels=np.arange(num_classes),
    )
    return 1 - fitness
