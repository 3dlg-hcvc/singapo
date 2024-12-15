# SINGAPO
<a href="https://pytorch.org/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/Lightning-792DE4?style=for-the-badge&logo=lightning&logoColor=white"></a>
<a href="https://wandb.ai/site"><img alt="WandB" src="https://img.shields.io/badge/Weights_&_Biases-FFBE00?style=for-the-badge&logo=WeightsAndBiases&logoColor=white"></a>

**SINGAPO**: **Sin**gle Image Controlled **G**eneration of **A**rticulated **P**arts in **O**bjects

[Jiayi Liu](https://sevenljy.github.io/), [Denys Iliash](), [Angel X. Chang](https://angelxuanchang.github.io/), [Manolis Savva](https://msavva.github.io/), [Ali Mahdavi-Amiri](https://www.sfu.ca/~amahdavi/)

Preprint

[Website](https://3dlg-hcvc.github.io/singapo/) | [Arxiv](https://arxiv.org/pdf/2410.16499) 

![teaser](docs/static/images/teaser.png)

## Environment Setup
We recommend to use [miniconda](https://docs.anaconda.com/miniconda/) to manage the environment. The environment was tested on Ubuntu 20.04.4 LTS.
```
# Create a conda environment
conda create -n singapo python=3.10
conda activate singapo

# Install Pytorch
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other packages
pip install -r requirement.txt

# Install Pytorch3D (for evaluation)
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```
To use [GPT-4o](https://openai.com/index/hello-gpt-4o/) for graph extraction (during inference), you need to add your OpenAI API key by creating a `.env` file in the root directory of the project as follows:
```
In the .env file
OPENAI_API_KEY=<YOUR_API_KEY>
```
## Download Data

### Data preprocessed from PartNet-Mobility dataset (train + eval) ###

We use the data preprocessed from [PartNet-Mobility](https://sapien.ucsd.edu/browse) dataset for training, evaluation, and part retrieval. Please download the data [here](https://aspis.cmpt.sfu.ca/projects/singapo/data/pm.zip) (~13GB) to run the quick demo, for evaluation, or for training.

### Our augmented data (for training only)  ###

If you're interested in training our model from scratch, please also download from [here](https://aspis.cmpt.sfu.ca/projects/singapo/data/augmented_train.zip)(~76GB) to use our augmented data.

### Data preprocessed from ACD dataset (for eval only) ###

If you'd like to run evaluation on the [ACD](https://huggingface.co/datasets/3dlg-hcvc/s2o) dataset, you can download our proprocessed data [here](https://aspis.cmpt.sfu.ca/projects/singapo/data/acd_test.zip)(~3GB).

### File structure ###
The default directory for loading our data is `../data`, which is the same level as our project directory.
```
├── data
│   ├── StorageFurniture
│   ├── Table
│   │   ├── <model_id>
│   ├── ...
├── <project directory>
```
For each object, we preprocess the data with the following files:
```
<model_id>
├── imgs         # 20 renderings from random views (in the resting state)
├── features     # DinoV2 features and foreground masks on the patches
├── objs         # textured meshes for parts
├── plys         # part meshes for retrieval
├── object.json  # part hierarchy and articulation parameters
```
The preprocessing script for rendering, feature extraction, and mask computation can be found under `scripts/preprocess`.

## Download Checkpoints

You can download our pretrained model [here](https://aspis.cmpt.sfu.ca/projects/singapo/ckpts/singapo_ckpt.zip)(~40MB), extract out and put it in the `exps` folder under the project directory.
```
<project directory>
├── exps
│   ├── singapo
│   │   ├── final
```
## Usage
### Quick Demo
We provide a quick demo to run the inference on an example input image located at `demo/demo_input.png`. This script will take the example image as input, predict part connectivity graph using [GPT-4o](https://openai.com/index/hello-gpt-4o/), extract image feature using [DinoV2](https://github.com/facebookresearch/dinov2), and generate articulated object using our model. Please make sure that the model checkpoint and preprocessed data (from PartNet-Mobility) are downloaded. 
```
# To run the whole package
python demo/demo.py
```
If you don't have the OpenAI API key yet, you can opt to skip the graph prediction by using our given graph `demo/example_graph.json` that is parsed from the GPT response.
```
# To skip the graph prediction using GPT-4o
python demo/demo.py --use_example_graph
```
If you successfully run the script, the output will be saved at `demo/demo_output`. By default, there will be three objects generated out by initializing with different noises.
For other configuration, please see the arguments in the script.

### Evaluation
If you're interested in evaluating our model on the test set (see the data split in `data/data_split.json` for PartNet-Mobility, and in `data/data_acd.json` for ACD dataset), you can run the test script as below. 
```
# Evaluate on the test set (given GT graph, no object category label)
python test.py \
    --config exps/singapo/final/config/parsed.yaml \
    --ckpt exps/singapo/final/ckpts/last.ckpt \ 
    --label_free \
    --which_data pm
```
We also share the graph prediction results [here](https://aspis.cmpt.sfu.ca/projects/singapo/pred_graph.zip) so that you can run the evaluation by taking the graph prediction from GPT-4o as input. Once downloaded, you can put it under the `exps` directory, as shown in the following file structure.
```
<project directory>
├── exps
│   ├── predict_graph
│   │   ├── acd_test
│   │   ├── pm_test
```
To use these recordings of the graph prediction for evaluation, you need to specify the path to one of the prediction folders `--G_dir`. For example,
```
# Evaluate on the test set (given predicted graph, no object category label)
python test.py \
    --config exps/singapo/final/config/parsed.yaml \
    --ckpt exps/singapo/final/ckpts/last.ckpt \
    --label_free \
    --which_data pm \ 
    --G_dir exps/pred_graph/pm_test
```
The evaluation is only supported on a single GPU, which was tested on a NVIDIA 3060 (12GB).

### Training
To train our model from scratch, the preprocessed data from PartNet-Mobility (downloaded [here](https://aspis.cmpt.sfu.ca/projects/singapo/data/pm.zip)) and our augmented data (downloaded [here](https://aspis.cmpt.sfu.ca/projects/singapo/data/augmented_train.zip)) is required. 

We train our model on top of a [CAGE](https://3dlg-hcvc.github.io/cage/) model pretrained under our setting. This checkpoint can be downloaded [here](https://aspis.cmpt.sfu.ca/projects/singapo/ckpts/pretrained_cage.zip), which is put under `pretrained` folder by default.
```
<project directory>
├── pretrained
│   ├── cage_cfg.ckpt
```
Run the following command to train our model from scratch. The original model is trained on 4 NVIDIA A100s.
```
python train.py \
    --config configs/config.yaml \
    --pretrained_cage pretrained/cage_cfg.ckpt
```

## Citation
```
@article{liu2024singapo,
  title={{SINGAPO}: Single Image Controlled Generation of Articulated Parts in Object},
  author={Liu, Jiayi and Iliash, Denys and Chang, Angel X and Savva, Manolis and Mahdavi-Amiri, Ali},
  journal={arXiv preprint arXiv:2410.16499},
  year={2024}
}
```