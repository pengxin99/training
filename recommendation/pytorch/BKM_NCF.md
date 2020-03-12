# install dependency package
1. Install [PyTorch = v0.4.0]

```bash
pip install torch torchvision
```

2. Install `unzip` and `curl`

```bash
sudo apt-get install unzip curl
```

3. Install other python packages

```bash
cd training/recommendation/pytorch
pip install -r requirements.txt
```

# dataset
First goto the dir:
```
cd training/recommendation/
```

download dataset: ./download_dataset.sh 

you can also use local dataset by symlink like:

```
ln -s /path_to_ml-20/ data_set/ml-20mx1x1/

```

# weight file:
put weight file at `training/recommendation/pytorch/5_epoch_model.pkl`

# run training and inference: cd ./pytorch/

## training
```
./ncf_train.sh
```

## inference
for throughput mode:
```
./ncf_inference.sh
```

for real_time mode:
```
./ncf_inference.sh --single
```