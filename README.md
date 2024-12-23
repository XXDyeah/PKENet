## Potential Knowledge Extraction Network for Class-Incremental Learning

### Code Dependencies
```
Python 3.7
Pytorch 1.10.0
torchvision 0.11.0
numpy 1.21.5
CUDA 11.7
tqdm 4.51.0
scipy 1.5.4
QtPy 1.9.0
PyYAML 6.0.1
matplotlib 3.5.3
```

### Experiments
T = 5 on CIFAR100 with AFC+PKENet and classifier CNN:
```bash
python3 -minclearn --options options/AFC/AFC_cnn_cifar100.yaml options/data/cifar100_3orders.yaml \
    --initial-increment 50 --increment 5 --fixed-memory \
    --device <GPU_ID> --label AFC_cnn_cifar100_10steps \
    --data-path <PATH/TO/DATA>
```

### Acknowledgements
Our method [PKENet](https://github.com/XXDyeah/PKENet) is designed as a plug-in method which can improve performance for other CIL methods, and this code is based on [AFC](https://github.com/kminsoo/AFC).
