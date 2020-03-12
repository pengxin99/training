# install dependency package
1. Pytorch 
Repo: https://github.com/pytorch/pytorch
branch: pytorch-1.5
version:6d78882158a4c43d69d8c850a4815ffdb20afe23
build and install:
```
git submodule update --init --recursive
python setup.py install --user
```
2. torchvision
```
conda install torchvision=0.4.2
```

3. pycocotools
repo: https://github.com/cocodataset/cocoapi.git
version: 8c9bcc3cf640524c4c20a9c40e89cb6a2f2fa0e9
install:
```
cd cocoapi/PythonAPI
python setup.py build_ext install --user
```

4. mlperf training
install: 
```
cd training/object_detection/
./install.sh
```
# dataset
download dataset: ./download_dataset.sh 
you can also use local dataset by symlink like:
```
mkdir -p pytorch/datasets/coco
ln -s /path_to_coco_dataset/annotations pytorch/datasets/coco/annotations 
ln -s /path_to_coco_dataset/train2017 pytorch/datasets/coco/train2017
ln -s /path_to_coco_dataset/test2017 pytorch/datasets/coco/test2017
ln -s /path_to_coco_dataset/val2017 pytorch/datasets/coco/val2017
```

# download weight file:
```
mkdir –p pytorch/ImageNetPretrained/MSRA
cd pytorch/ImageNetPretrained/MSRA
curl -O https://download.pytorch.org/models/maskrcnn/e2e_mask_rcnn_R_50_FPN_1x.pth
```

# run benchmark and profiling: cd training/object_detection/
```
./run_and_time_for_bp.sh
```