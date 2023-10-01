# Implementation of [Rethink-Backdoor-Diffusion]()
The DDPM part is adapted from [Classifier-free Diffusion Guidance](https://github.com/coderpiaobozhe/classifier-free-diffusion-guidance-Pytorch).

The Stable-Diffusion part is based on [stable-diffusion-1](https://github.com/LambdaLabsML/examples/tree/main/stable-diffusion-finetuning) which makes fine-tuning Stable-Diffusion model easier.

## Stable-Diffusion part

### Environment
```
cd stable-diffusion
pip install -r requirements.txt
```

### Data preparation
1. ImageNette

Get the full size version of imagenette from [here](https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz) and unzip to "data/imagenette",
```bash
cd data/imagenette
wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz
tar -zxvf imagenette2.tgz
```
and produce the **Badnets-like** version.
```
python badnets.py
```

2. Caltech15

### Model preparetion

The pretrained stable diffusion model can be downloaded from [huggingface](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original).

### Train

```
cd stable-diffusion 
bash scripts/train.sh
```
SD model will be saved at logs/imagenette/experiment_name/checkpoints

### Sample 

```
cd stable-diffusion 
bash scripts/sample.sh
```
Samples model will be saved at outputs/imagenette/experiment_name

### Evaluation

1. train classifier on ImageNette
```
cd classifier
bash scripts/train_imagenette.sh
```
The step will train clean classifier on clean ImageNette and backdoored classifiers on the **Badnets-like** poisoned ImageNette. They are saved at model_ckpt/imagenette

2. run clean classifier on generated data.
```
cd classifier
python eval_imagenette.py
```
The generated prediction will be saved at stable-diffusion/result_clf/imagenette

3. eval trigger ratio and the ratio of generations mismatching their prompts
```
cd stable-diffusion 
python eval_tgr_imagenette.py
```

### defense backdoor by training on generation
```
cd classifier
bash scripts/train_on_gen_imagenette.sh
```



## DDPM part

## Environment
```
cd ddpm
conda env create -f environment.yml
```