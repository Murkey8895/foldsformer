# Foldsformer: Learning Sequential Multi-Step Cloth Manipulation with Space-Time Attention
**Kai Mo, Chongkun Xia, Xueqian Wang, Yuhong Deng, Xuehai Gao, Bin Liang**

**Tsinghua University**

This repository is a PyTorch implementation of the paper "Foldsformer: Learning Sequential Multi-Step Cloth Manipulation with Space-Time Attention", published in IEEE RA-L.

[Website](https://sites.google.com/view/foldsformer) | [IEEE Manuscript](https://ieeexplore.ieee.org/abstract/document/9987684) | [ArXiv](https://arxiv.org/abs/2301.03003)

If you find this code useful in your research, please consider citing:

~~~
@ARTICLE{mo2022foldsformer,  
    author={Mo, Kai and Xia, Chongkun and Wang, Xueqian and Deng, Yuhong and Gao, Xuehai and Liang, Bin},  
    journal={IEEE Robotics and Automation Letters},   
    title={Foldsformer: Learning Sequential Multi-Step Cloth Manipulation With Space-Time Attention},   
    year={2023},  
    volume={8},  
    number={2},  
    pages={760-767},  
    doi={10.1109/LRA.2022.3229573}
}
~~~

## Table of Contents
* [Installation](#Installation)
* [Generate Data](#Generate-Data)
* [Train Foldsformer](#Train-Foldsformer)
* [Evaluate Foldsformer](#Evaluate-Foldsformer)

## Installation
This simulation environment is based on SoftGym. You can follow the instructions in [SoftGym](https://github.com/Xingyu-Lin/softgym) to setup the simulator.

1. Clone this repository.

2. Follow the [SoftGym](https://github.com/Xingyu-Lin/softgym) to create a conda environment and install PyFlex. [A nice blog](https://danieltakeshi.github.io/2021/02/20/softgym/) written by Daniel Seita may help you get started on SoftGym.

3. Install the following packages in the created conda environment:
    
    * pytorch and torchvision: `pip install torchvision` or `conda install torchvision -c pytorch`
    * einops: `pip install einops`
    * tqdm: `pip install tqdm`
    * yaml: `pip install PyYaml`


4. Before you use the code, you should make sure the conda environment activated(`conda activate softgym`) and set up the paths appropriately: 
   ~~~
   export PYFLEXROOT=${PWD}/PyFlex
   export PYTHONPATH=${PYFLEXROOT}/bindings/build:$PYTHONPATH
   export LD_LIBRARY_PATH=${PYFLEXROOT}/external/SDL2-2.0.4/lib/x64:$LD_LIBRARY_PATH
   ~~~
   The provided script `prepare_1.0.sh` includes these commands above.

## Generate Data

* Generate initial configurations:

  ~~~
  python generate_configs.py --num_cached 1000 --cloth_type random
  python generate_configs.py --num_cached 100 --cloth_type square
  python generate_configs.py --num_cached 100 --cloth_type rectangle
  ~~~

  where `--num_cached` specifies the number of configurations to be generated, and `--cloth_type` specifies the cloth type (square | rectangle | random). These generated initial configurations will be saved in `cached configs/`

* Generate trajectories by random actions:

  ```
  python generate_random.py --gui --corner_bias --img_size 224 --cached random1000 --horizon 8
  ```

  where `--img_size` specifies the image size captured by the camera in the simulator, `--cached` specifies the filename of the cached configurations, and `--horizon` specifies the number of actions in a trajectory. You can remove `--gui` to run headless and remove `--corner_bias` to pick the cloth uniformly instead of picking the corners. These generated trajectories will be saved in `data/random/corner bias` and `data/random/random`.

* Generate expert demonstrations:

  ```
  python generate_demonstrations.py --gui --task DoubleTriangle --img_size 224 --cached square100
  python generate_demonstrations.py --gui --task DoubleStraight --img_size 224 --cached rectangle100
  python generate_demonstrations.py --gui --task AllCornersInward --img_size 224 --cached square100
  python generate_demonstrations.py --gui --task CornersEdgesInward --img_size 224 --cached square100
  ```

  where `--task` specifies the task name, `--img_size` specifies the image size captured by the camera in the simulator, and `--cached` specifies the filename of the cached configurations, and . You can remove `--gui` to run headless. These generated demonstrations will be saved in `data/demonstrations`.


  `Demonstrator/demonstrator.py` includes the scripted demonstrator by accessing the ground truth position of each particle.

## Train Foldsformer

* Preprocess the data (split each long trajectory into sub-trajectories):

  ```
  python utils/prepare_data_list.py 
  ```

* Set up the model, optimizer and other details in `train/train configs/train.yaml`.

* Train Foldsformer:

  ```
  python train.py --config_path train
  ```

  where `--config_path` specifies the `yaml` configuration filename in `train/train configs/`.

## Evaluate Foldsformer

* Download the evaluation set and model weights:
  * Download the [evaluation initial configurations](https://drive.google.com/drive/folders/1_kZpRBu9bMmt-gFFLxKw7Ih7NOfpbxI-?usp=sharing), and then put them in `cached configs/`.
  * Download the [Foldsformer weights](https://drive.google.com/file/d/145DZ5_HGdiNp23gfli4btGwjCZA6XAk8/view?usp=sharing), and then put it in `train/trained model/Foldsformer/`
  * Download the [demonstration sub-goals](https://drive.google.com/file/d/1bscYy5HXnnRZQfTIZqViL5cRHF_xwxZ7/view?usp=sharing), and then extract them in `data/demo/`

* Evaluate Foldsformer by running:

  ```
  python eval.py --gui --task DoubleTriangle --model_config train --model_file foldsformer_eval --cached square
  python eval.py --gui --task DoubleStraight --model_config train --model_file foldsformer_eval --cached rectangle
  python eval.py --gui --task AllCornersInward --model_config train --model_file foldsformer_eval --cached square
  python eval.py --gui --task CornersEdgesInward --model_confilg train --model_file foldsformer_eval --cached square
  ```

  The evaluation results are saved in `eval result/`.

If you have any questions, please feel free to contact me via mok21@mails.tsinghua.edu.cn

