import numpy as np
import torch
from torcheval.metrics.functional import multiclass_f1_score


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


def f1score_func(model, data_loader, num_classes, device):
    model.eval()
    fitness = 0

    with torch.no_grad():
        data, target = next(iter(data_loader))
        data, target = data.to(device), target.to(device)
        output = model(data)

        # fitness += f1_score(
        #     y_true=target.cpu().numpy(), y_pred=pred.cpu().numpy(), average="macro"
        # )
        fitness += multiclass_f1_score(
            output, target, average="weighted", num_classes=num_classes
        ).item()

    return 1 - fitness
