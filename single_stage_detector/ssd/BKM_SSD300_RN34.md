# Python version > 3.0
## Install dependency packages 
1.	Pytorch
```
git clone https://github.com/pytorch/pytorch.git
cd pytorch
python setup.py install
```

2.	Get SSD300/SSD-RN34 workload
```
git clone  https://gitlab.devtools.intel.com/mlp-broadproduct-ia-benchmarking/pytorch.git  -b broad_product
cd pytorch/single_stage_detector
pip install -r requirements.txt --user
pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
```
## SSD300/SSD_RN34 benchmark
```
cd ssd
mkdir log
./run_and_time_cpu.sh
```
  
## Post-processing the profiling result
```
cd pytorch/tool
./run_post.sh
```