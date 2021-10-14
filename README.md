## Air-Text: Air-Writing and Recognition System (ACMMM 2021, Oral)
This is the official PyTorch implementation for our work; Air-Text: Air-Writing and Recognition System. Here, you can find source codes for model training and brief demo as shown below.
<p align="center"><img width=70% src="figs/demo.gif"></p>

---
### Overview
Air-Text is a novel system to write in the air using fingertips as a pen. Air-Text provides various functionalities by the seamless integration of Air-Writing and Text-Recognition Modules. 
<p align="center"><img width=50% src="figs/overview.png"></p>

---
### Environment Setup
Using [Anaconda](https://www.anaconda.com/distribution/) is recommended.
```shell
conda create -n airtext_env python=3.6
conda activate airtext_env
conda install pytorch==1.5.0 torchivision==0.6.0 cudatoolkit=10.2 -c pytorch
pip install opencv-python torchsummary tensorboardX matplotlib lmdb natsort nltk
```

---
### Training Air-Writing Module
In order to train Air-Writing Module, first download [SCUT-Ego-Gesture dataset](http://www.hcii-lab.net/data/). (Currently, as there is no direct link to download the dataset, you may consider contacting the [authors](https://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w11/Wu_YOLSE_Egocentric_Fingertip_ICCV_2017_paper.pdf) of this dataset.)

Put the downloaded dataset in `./AirWritingModule/dataset`, type command `cd AirWritingModule` and run `train.py`.

---
### Training Text-Recognition Module

#### Single Digit Recognition
In order to train Text-Recognition Module for single digit recognition, just type command `cd TextRecognitionModule/MNIST` and run `digitmodel.py`. Downloading MNIST dataset and training will be started automatically.

#### English Word Recognition

---
### Pre-trained weights
Pre-trained weights for above all three models can be downloaded from [here](https://drive.google.com/file/d/1BehjQ5S65Z7kA-_0vJkYcRwy0Z52ACpT/view?usp=sharing). Extract all the files in the root directory of this repository.

---
### Demo
First, check you can get a video input by connecting a webcam to your desktop.
If you want to test single digit recognition, please run `demo_digit.py` in the terminal. Or you can test English word recognition by running `demo_word.py`.

---

### Datasets
- [Scene Flow](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html), 
- [KITTI Stereo 2012](http://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=stereo), 
- [KITTI Stereo 2015](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo), 
- [Cityscapes](https://www.cityscapes-dataset.com/)
- [Lost and Found](http://www.6d-vision.com/lostandfounddataset) 

To detect objects of both class-agnostic obstacle class (from Lost and Found) and the set of 19 annotated classes (from Cityscapes), we created a `city_lost` directory by our multi-dataset fusion approach. Our folder structure is as follows:

---


#### Train and Evaluate
Detailed commands for training and evaluation are described in `script/train_test_guide.txt`. 

For training our RODSNet on `city_lost` datasets:
```shell
python main.py --gpu_id 0 --dataset city_lost --checkname resnet18_train_citylost_eps_1e-1_without_transfer \
--optimizer_policy ADAM --lr 4e-4 --weight_decay 1e-4 --epochs 400 \
--train_semantic --train_disparity --with_refine --refinement_type ours --batch_size 4 --val_batch_size 4 \
--epsilon 1e-1
```
Trained results are saved in `$RODSNet/run/[dataset]/[checkname]/experiment_0/` directory.

To evaluate our performance on `city_lost` dataset with pretrained results:
```shell
python main.py --gpu_id 0 --dataset city_lost --checkname city_lost_test \
--with_refine  --refinement_type ours --val_batch_size 1 --train_semantic --train_disparity --epsilon 1e-1 \
--resume ckpt/city_lost/best_model_city_lost/score_best_checkpoint.pth --test_only
```

For fast inference, evaluation is run without saving the intermediate results.  (To save any results, add `--save_val_results` option. The output results will then be saved in `$RODSNet/run/[dataset]/[checkname]/experiment_0/results` folder.)

#### Sample Inference Test

```shell
python sample_test.py --gpu_id 0 \
--with_refine \
--refinement_type ours \
--train_disparity --train_semantic \
--resume ckpt/city_lost/best_model_city_lost/score_best_checkpoint.pth
```

---

### Acknowledgments

- This work was partly supported by Institute for Information & communications Technology Promotion(IITP) grant funded by the Korea government(MSIT) (No.2020-0-00440, Development of artificial intelligence technology that continuously improves itself as the situation changes in the real world) and (No.2020-0-00842, Development of Cloud Robot Intelligence for Continual Adaptation to User Reactions in Real Service Environments).

- Parts of the code are adopted from previous works ([AANet](https://github.com/haofeixu/aanet), and [RFNet](https://github.com/AHupuJR/RFNet)). We appreciate the original authors for their awesome repos. 

### Citation
```bash
@article {songjeong2021rodsnet,
    author = {Song, Taek-jin and Jeong, Jongoh and Kim, Jong-Hwan},
    title = {End-to-end Real-time Obstacle Detection Network for Safe Self-driving via Multi-task Learning},
    year = {2021},
    doi = {???},
    URL = {https://doi.org/???},
    journal = {Journal}
}
```
