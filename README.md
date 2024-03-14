# Generative SSL

This is the PyTorch implemention of our paper **"Can Generative Models Improve Self-Supervised Representation Learning?"** submitted to ECCV 2024 for reproducing the experiments.

## Requirements

To create the virtual environment for running the experiments, you need to run:

`pip install -r requirements.txt`

**Note:** 
> You always need to set the proper path to the virtual environment, the dataset and the model in each SLURM file before submitting the job. Here are the options for the datasets and models that we used in our experiments:

> - Datasets: ImageNet, iNaturalist2018, Food101, Places365, CIFAR10/100
> - Models: Baseline (SimSiam model trained on ImageNet), SimSiam model trained with ICGAN augmentations, SimSiam model trained with Stable Diffusion augmentations

## Data Generation

To generate augmentations with ICGAN run:

`sbatch GenerativeSSL/scripts/generation_scripts/gen_img_icgan.slrm`

To generate augmentations with Stable Diffusion run:

`sbatch GenerativeSSL/scripts/generation_scripts/gen_img_stablediff.slrm`

## Training 

To train the SimSiam method on the ImageNet, run:

`sbatch GenerativeSSL/scripts/train_scrpits/train_simsiam_singlenode.slrm`

In this file, there is a `use_synthetic_data` flag that you can use to train the model with augmentations. You just need to specify the path to synthetic data. (Either ICGAN or Stable Diffusion augmentations) By default, the `use_synthetic_data` flag has been passed in the SLURM file.

## Evaluation

For downstream tasks, there are all evaluation scripts in this `GenerativeSSL/scripts/eval_scripts`  folder. In each dataset folder in `eval_scripts` there are three SLURM files. (baseline model, model trained with ICGAN aug, model trained with stablediff aug)

Similarly for evaluation, you just need to submit the slurm file related to the dataset you want. Again, you need to specify the path to the virtual environment, the dataset and the related checkpoint in each SLURM file. For example, command below run the experiment of evaluating model trained with stable diffusion augmentations on Food101:

`sbatch GenerativeSSL/scripts/eval_scripts/food101/stablediff.slrm`

## Pretrained Models

We also provide the checkpoints for all the trained models here in the [LINK](https://drive.google.com/drive/folders/1xPIbf1cOPqzIzuZ185GjAprA8XmQ0Tvu)


