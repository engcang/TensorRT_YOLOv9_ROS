
<h1 align="center"><span>TensorRT-YOLOv9-ROS</span></h1>

+ ROS version of [YOLOv9](https://github.com/WongKinYiu/yolov9) accelerated with [TensorRT](https://github.com/NVIDIA/TensorRT) API
+ This repository is a merely re-implementation with `ROS` of the:
  + 👏 [TensorRT-YOLOv9-C++](https://github.com/spacewalk01/TensorRT-YOLOv9), which is based on
    + [YOLOv9](https://github.com/WongKinYiu/yolov9) - `YOLOv9`: Learning What You Want to Learn Using Programmable Gradient Information.
    + [TensorRT](https://github.com/NVIDIA/TensorRT/tree/release/8.6/samples) - `TensorRT` samples and api documentation.
    + [TensorRTx](https://github.com/wang-xinyu/tensorrtx) - Implementation of popular deep learning networks with TensorRT network definition API.

https://github.com/engcang/TensorRT_YOLOv9_ROS/assets/34734707/0dff22cc-ec12-45fb-a931-fb0c90181fd7

<br>

### Known issues / notes
+ The resolution of image to be trained should be multiplication of 64
+ 2024. 05. 12.: Now supporting `TensorRT` >= 10
+ Check the paths of TensorRT in CMakeLists.txt's line 25, 26

<br>

## Dependencies
+ `ROS` (currently supporting only `ROS1`)
+ `C++` >= 17
+ `cmake` >= 3.14
+ `OpenCV` >= 4.2
+ `TensorRT`, `CUDA`, `cuDNN`
  + `.engine` file generated with `TensorRT`
+ Tested versions:
  + Desktop with i9-10900k, RTX 3080 - `CUDA` 11.5, `cuDNN` 8.3.2.44, `TensorRT` 8.4.0.6

</details>
<br>

## You may want to:

<details><summary> ■ Unfold here to see how to install CUDA, cuDNN and TensorRT </summary>

### ● **Note that apt install with deb is preferred to run file and source file build for both of `CUDA` and `cuDNN`**
+ Download and install `CUDA` following instructions at here - https://developer.nvidia.com/cuda-downloads
+ Download and install `cuDNN` following instructions at here - https://developer.nvidia.com/cudnn-downloads
  + If you want, also refer to here - https://docs.nvidia.com/deeplearning/cudnn/installation/linux.html#
+ Set up environmental paths
```bash
gedit ~/.bashrc
*** Type and save below, CUDA_PATH should be like /usr/local/cuda-11.5, depending on your version ***
export PATH=CUDA_PATH/bin:$PATH 
export LD_LIBRARY_PATH=CUDA_PATH/lib64:$LD_LIBRARY_PATH

. ~/.bashrc

gedit ~/.profile
*** Type and save below, CUDA_PATH should be like /usr/local/cuda-11.5, depending on your version ***
export PATH=CUDA_PATH/bin:$PATH 
export LD_LIBRARY_PATH=CUDA_PATH/lib64:$LD_LIBRARY_PATH

. ~/.profile
```
+ Verify, if installed properly
```bash
# Verify
dpkg -l | grep cuda
dpkg -l | grep cudnn
nvcc --version
```

<br>

### ● **Note that apt install with deb is preferred to other methods for `TensorRT`**
+ Download `TensorRT` at here - https://developer.nvidia.com/tensorrt-download
+ Follow the instructions at here - https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-debian
  + Installing full packages is recommended, which means:
  ```bash
  sudo apt install tensorrt
  sudo apt install python3-libnvinfer-dev
  sudo apt install onnx-graphsurgeon
  ```

<br>

</details>

<details><summary> ■ Unfold here to see how to train custom data / generate TensorRT engine file with safe Python3 virtual environment </summary>

<br>

### ● Common step for training / engine file
0. Make sure that you have installed all dependencies properly.
  + Particularly, you should install full packages of `TensorRT`: `tensorrt`, `python3-libnvinfer-dev`, `onnx-graphsurgeon`
1. Install and make `Python3` virtual env
```bash
python3 -m pip install virtualenv virtualenvwrapper
cd <PATH YOU WANT TO SAVE VIRTUAL ENVIRONMENT>
virtualenv -p python3 <NAME YOU WANT>

*** Now you can activate with
source <PATH YOU SAVED>/<NAME YOU WANT>/bin/activate

*** Deactivate with
deactivate
```
2. (While virtual env being activated), clone `YOLOv9` repo and install requirements
```bash
git clone https://github.com/WongKinYiu/yolov9
cd yolov9
pip install -r requirements.txt
```

<br>

### ● Converting .pt to .onnx, and then .engine
0. (While virtual env being activated)
1. Get trained `YOLOv9` weight file as `.pt` by training your own data or downloading the pre-trained model at here - https://github.com/WongKinYiu/yolov9/releases
2. Reparameterize the `.pt` file (saving computation, memory, and size by trimming unnecessary parts for inference but necessary only for training)
```bash
cd yolov9 # cloned at above step
wget https://raw.githubusercontent.com/engcang/TensorRT_YOLOv9_ROS/main/reparameterize.py

*** Change the number of classes in the reparameterize.py in line 8 (nc=80)
python reparameterize.py yolov9-c.pt yolov9-c-reparameterized.pt # input.pt output.put
```
3. Export `.pt` file as `.onnx`
```bash
python export.py --weights yolov9-c-reparameterized.pt --include onnx
```
4. Then `.onnx` to `.engine`
```bash
/usr/src/tensorrt/bin/trtexec --onnx=yolov9-c-reparameterized.onnx --saveEngine=yolov9-c.engine
#for faster, less accurate
/usr/src/tensorrt/bin/trtexec --onnx=yolov9-c-reparameterized.onnx --saveEngine=yolov9-c-fp16.engine --fp16
#not recommended - much faster, much less accurate
/usr/src/tensorrt/bin/trtexec --onnx=yolov9-c-reparameterized.onnx --saveEngine=yolov9-c-int8.engine --int8
```

<br>

### ● Training your own data
0. (While virtual env being activated) + `YOLOv9` is cloned already, requirements are installed already
1. Prepare data and labels in `YOLO format`.
  + You may want to use this - https://github.com/AlexeyAB/Yolo_mark
  + Or `roboflow` - https://docs.ultralytics.com/yolov5/tutorials/roboflow_datasets_integration/
2. Make proper `data.yaml` file by copying and editing `yolov9/data/coco.yaml` as follows:
```yaml
path: training  # dataset root dir (relative from train.py file)
train: train    # train images folder (relative to 'path')
val: val        # val images folder (relative to 'path')
test: test      # test images folder (relative to 'path')

# Classes
names:
  0: Transmission tower
  1: Insulator
```
3. Make proper `yolov9.yaml` file by copying and editing `yolov9/models/detect/yolov9.yaml or yolov9-c, yolov9-e, etc.`
```yaml
# parameters
nc: 2  # number of classes
depth_multiple: 1.0  # model depth multiple
width_multiple: 1.0  # layer channel multiple
#activation: nn.LeakyReLU(0.1)
#activation: nn.ReLU()

# anchors
anchors: 3

# YOLOv9 backbone
backbone:
  [
   [-1, 1, Silence, []],  
   
   # conv down
   [-1, 1, Conv, [64, 3, 2]],  # 1-P1/2
   
   ...
  ]
```
4. Edit learning parameters by editing `yolov9/data/hyps/hyp.scratch-high.yaml`
5. **Put all of files properly in the `yolov9` folder. If outside the `yolov9` folder, error occurs!**
```
yolov9
│  ...
├─ data # Reference folder
│  ├─ coco.yaml
│  └─ hyps
│     └─ hyp.scratch-high.yaml
├─ models # Reference folder
│  ...
│  ├─ detect
│  ...
│  │  ├─ yolov9-c.yaml
│  │  ├─ yolov9-e.yaml
│  │  └─ yolov9.yaml
├─ runs # Output saved folder
│  ...
├─ train.py # Using this file for GELAN
├─ train_dual.py # Using this file for YOLOv9
├─ training # Using this folder
│  ├─ yolov9-c.pt
│  ├─ data.yaml
│  ├─ yolov9.yaml
│  ├─ test
│  │  ├─ 02001.jpg
│  │  ├─ 02001.txt
│  │  └─ ...
│  ├─ train
│  │  ├─ 00001.jpg
│  │  ├─ 00001.txt
│  │  └─ ...
│  ├─ val
│  │  ├─ 04000.jpg
│  │  ├─ 04000.txt
│  │  └─ ...
└─ └─ ...
```
6. Train
```bash
cd yolov9

*** Using pretrained model (yolov9-c.pt here), fine-tuning:
python train_dual.py --batch-size 4 --epochs 100 --img 640 --device 0 --close-mosaic 15 \
--data training/data.yaml --weights training/yolov9-c.pt --cfg training/yolov9.yaml --hyp data/hyps/hyp.scratch-high.yaml

*** From the scratch:
python train_dual.py --batch-size 4 --epochs 100 --img 640 --device 0 --close-mosaic 15 \
--data training/data.yaml --weights '' --cfg training/yolov9.yaml --hyp data/hyps/hyp.scratch-high.yaml
```

<br>

### ● Trouble shooting for training
0. (While virtual env being activated)
1. `AttributeError: 'FreeTypeFont' object has no attribute 'getsize'`
  + This is because installed Pillow version is too recent.
  + Solve with `pip install Pillow==9.5.0`
2. Getting `Killed` and does not train
  + Lack of memory, reduce `batch-size` a lot
3. `AssertionError: Invalid CUDA '--device 0' requested, use '--device cpu' or pass valid CUDA device(s)`
  + This is because installed `torch` and `torchvision` are not `CUDA` versions.
  + Solve as:
  ```bash
  *** Check the version at https://download.pytorch.org/whl/torch_stable.html
  *** torch >= 1.7.0, torchvision>=0.8.1

  pip install torch==1.11.0+cu115 torchvision==0.12.0+cu115 -f https://download.pytorch.org/whl/torch_stable.html
  ```
4. `RuntimeError: CUDA out of memory. Tried to allocate 50.00 MiB (GPU 0; 9.76 GiB total capacity; 6.68 GiB already allocated; 45.00 MiB free; 6.82 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF`
  + Lack of memory, reduce `batch-size` a lot

</details>

<br>

## How to install
+ Make sure you have installed all of dependencies properly
+ Clone this repository (**Check the paths of TensorRT in CMakeLists.txt**) and build

```bash
cd ~/<your_workspace>/src
git clone https://github.com/engcang/TensorRT_YOLOv9_ROS.git

*** Check the paths of TensorRT in CMakeLists.txt ***
cd ~/<your_workspace>
catkin build -DCMAKE_BUILD_TYPE=Release
```

<br>

## How to use
+ Check the paths of files, params in `config/config.yaml`
+ Then run

```bash
roslaunch tensorrt_yolov9_ros run.launch
```

<br>

## You may also want to see
+ [tkdnn-ros](https://github.com/engcang/tkdnn-ros): `YOLO` (v3, v4, v7) accelerated with `TensorRT` using `tkdnn`
