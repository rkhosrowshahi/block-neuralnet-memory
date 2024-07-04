import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from sklearn.metrics import top_k_accuracy_score
from torcheval.metrics.functional import multiclass_f1_score


def train_step(model, data_loader, criterion, optimizer, device, num_classes=None):
    model.train()

    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in data_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)

        loss.backward()
        optimizer.step()

        # statistics
        running_loss += loss.item()
        running_corrects += torch.sum(preds == labels.data) / inputs.size(0)

    return running_loss / len(data_loader), running_corrects.double() / len(data_loader)


def eval_model(model, data_loader, device, num_classes):
    model.eval()

    running_f1 = 0
    running_top1 = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            running_f1 += multiclass_f1_score(
                outputs, labels, average="macro", num_classes=10
            ).item()

            running_top1 += top_k_accuracy_score(
                y_true=labels.cpu().numpy(),
                y_score=outputs.cpu().detach().numpy(),
                k=1,
                labels=np.arange(num_classes),
            )

    return running_f1 / len(data_loader), running_top1 / len(data_loader)


def train(
    model,
    train_loader,
    test_loader,
    num_classes,
    criterion,
    optimizer,
    num_epochs=1000,
    device=None,
):
    epoch_loss, epoch_acc, epoch_top1, epoch_f1, epoch_test_top1, epoch_test_f1 = (
        [],
        [],
        [],
        [],
        [],
        [],
    )

    for epoch in range(num_epochs):
        loss, acc = train_step(
            model, train_loader, criterion, optimizer, device, num_classes
        )
        train_top1, train_f1 = eval_model(model, train_loader, device, num_classes)
        test_top1, test_f1 = eval_model(model, test_loader, device, num_classes)

        epoch_loss.append(loss)
        epoch_acc.append(acc)
        epoch_top1.append(train_top1)
        epoch_f1.append(train_f1)
        epoch_test_top1.append(test_top1)
        epoch_test_f1.append(test_f1)

        print(
            f"Epoch {epoch}/{num_epochs} | train_loss: {loss:.6f} | train_acc: {acc:.6f} | train_top1: {train_top1:.6f} | train_f1: {train_f1:.6f} | test_top1: {test_top1:.6f} | test_f1: {test_f1:.6f}"
        )

        if loss < 1e-5:
            break

    hist_dict = {
        "Iterations": np.arange(epoch + 1),
        "train_loss": epoch_loss,
        "train_acc": epoch_acc,
        "train_top1": epoch_top1,
        "train_f1": epoch_f1,
        "test_top1": epoch_test_top1,
        "test_f1": epoch_test_f1,
    }
    return model, hist_dict
