# ESGCN official repository

# requirements

pytorch >= 1.7


# datasets

download four datasets from anonymous github repo https://github.com/xyzlackkvvz/PEMS_dataset

```bash
└── dataset/ # empty directory because of license
    ├── pems03.npz # model
    ├── pems04.npz # data load
    ├── pems07.npz
    └── pems08.npz # options
```


# run

```bash
sh exp/pems03.sh
sh exp/pems04.sh
sh exp/pems07.sh
sh exp/pems08.sh
```