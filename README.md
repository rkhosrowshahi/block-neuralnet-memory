# block-neuralnet-memory

### 5. test the model

ResNet18 and CIFAR-100

```bash
    python3 optimal_block.py \
    --net resnet18 \
    --model_path "./models/resnet18_cifar100_200steps_params.pt" \
    --save_dir "nsga2_resnet18_cifar100_200epochs_1000data_womerge" \
    --dataset cifar100 \
    --ub 500
```

```bash
    python3 optimal_block.py \
    --net resnet18 \
    --model_path "./models/resnet18_cifar10_200steps_params.pt" \
    --save_dir "nsga2_resnet18_cifar10_200epochs_1000data_womerge" \
    --dataset cifar10 \
    --ub 500
```
