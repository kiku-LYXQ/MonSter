# 🚀 MonSter (CVPR 2025) 🚀
**Official PyTorch implementation of MonSter**   
[MonSter: Marry Monodepth to Stereo Unleashes Power](https://arxiv.org/abs/2501.08643)
Junda Cheng, Longliang Liu, Gangwei Xu, Xianqi Wang, Zhaoxing Zhang, Yong Deng, Jinliang Zang, Yurui Chen, Zhipeng Cai, Xin Yang <br/>

##  🌼 Abstract
MonSter represents an innovative approach that effectively harnesses the complementary strengths of monocular depth estimation and stereo matching, thereby fully unlocking the potential of stereo vision. This method significantly enhances the depth perception performance of stereo matching in challenging regions such as ill-posed areas and fine structures. Notably, MonSter ranks first across five of the most widely used leaderboards, including SceneFlow, KITTI 2012, KITTI 2015, Middlebury, and ETH3D. Additionally, in terms of zero-shot generalization, MonSter also significantly and consistently outperforms state-of-the-art methods, making it the current model with the best accuracy and generalization capabilities.

##  📝 Benchmarks performance
![teaser](media/teaser.png)
![benchmark](media/benchmark.png)
Comparisons with state-of-the-art stereo methods across five of the most widely used benchmarks.
## :art: Zero-shot performance
![visualization1](media/vis1.png)
Zero-shot generalization performance on the KITTI benchmark.
![visualization2](media/vis2.png)
Zero-shot generalization performance on our captured stereo images.

## ⚙️ Installation
* NVIDIA RTX 3090
* python 3.8

### ⏳ Create a virtual environment and activate it.

```Shell
conda create -n monster python=3.8
conda activate monster
```
### 🎬 Dependencies

```Shell
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install tqdm
pip install scipy
pip install opencv-python
pip install scikit-image
pip install tensorboard
pip install matplotlib 
pip install timm==0.6.13
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html
pip install accelerate==1.0.1
pip install gradio_imageslider
pip install gradio==4.29.0

```

## ✏️ Required Data

* [SceneFlow](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)
* [KITTI](https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo)
* [ETH3D](https://www.eth3d.net/datasets)
* [Middlebury](https://vision.middlebury.edu/stereo/submit3/)
* [TartanAir](https://github.com/castacks/tartanair_tools)
* [CREStereo Dataset](https://github.com/megvii-research/CREStereo)
* [FallingThings](https://research.nvidia.com/publication/2018-06_falling-things-synthetic-dataset-3d-object-detection-and-pose-estimation)
* [InStereo2K](https://github.com/YuhuaXu/StereoDataset)
* [Sintel Stereo](http://sintel.is.tue.mpg.de/stereo)
* [HR-VS](https://drive.google.com/file/d/1SgEIrH_IQTKJOToUwR1rx4-237sThUqX/view)

## ✈️ Model weights

| Model      |                                               Link                                                |
|:----:|:-------------------------------------------------------------------------------------------------:|
| KITTI (one model for both 2012 and 2015)| [Download 🤗](https://huggingface.co/onnx-community/metric3d-vit-small) |
| Middlebury | [Download 🤗](https://huggingface.co/onnx-community/metric3d-vit-large) |
|ETH3D | [Download 🤗](https://huggingface.co/onnx-community/metric3d-vit-giant2) |
|sceneflow | [Download 🤗](https://huggingface.co/onnx-community/metric3d-vit-giant2) |
|mix_all (mix of all datasets) | [Download 🤗](https://huggingface.co/onnx-community/metric3d-vit-giant2) |

The mix_all model is trained on all the datasets mentioned above, which has the best performance on zero-shot generalization.


## ✈️ Evaluation

To evaluate the zero-shot performance of MonSter on Scene Flow, KITTI, ETH3D, vkitti, DrivingStereo, or Middlebury, run

```Shell
python evaluate_stereo.py --restore_ckpt ./pretrained/sceneflow.pth --dataset *(select one of ["eth3d", "kitti", "sceneflow", "vkitti", "driving"])
```
or use the model trained on all datasets, which is better for zero-shot generalization.
```Shell   
python evaluate_stereo.py --restore_ckpt ./pretrained/mix_all.pth --dataset *(select one of ["eth3d", "kitti", "sceneflow", "vkitti", "driving"])
```
