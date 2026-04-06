# Implicit Maximum Likelihood Estimation for Model Predictive Control &nbsp;&nbsp;

Training and visualizing of Implicit Maximum Likelihood Estimation for Model Predictive Control.

The [main branch](https://github.com/kir-/mpc-imle/tree/main) contains code for training imle models and planning via value-function guided sampling on the D4RL locomotion and Maze2D environments.

## Installation

```
conda env create -f environment.yml
conda activate mpc-imle
pip install -e .
```

## Making Local Env File
Copy `.env.local` to make a `.env`. This file will be used to set specify your device such as cuda or cpu. It is also used for configurations for weights and biases.

## Using pretrained weights
[TODO]

## Training from scratch

1. Train a model with:
```
python scripts/train.py
```

Training behavior is controlled by the following flags:

-   --task : policy or value
-   --backend : torch or jax
-   --model : diffusion or imle
-   --dataset : one of the supported D4RL datasets (e.g. walker2d-, maze2d-)

Example for a Locomotion Task with IMLE:
```
python scripts/train.py --task policy  --backend torch  --model imle  --dataset walker2d-medium-v2 
```

2. Plan using your newly-trained models with the same command:
```
python scripts/plan.py --backend torch  --model imle  --dataset walker2d-medium-v2 
```

3. Time the model with the same command:
```
python scripts/time.py --backend torch  --model imle  --dataset walker2d-medium-v2 
```

## Reference
```
@article{lee2026implicit,
    title={Implicit Maximum Likelihood Estimation for Real-time Generative Model Predictive Control},
    author={Lee, Grayson and Bui, Minh and Zhou, Shuzi and Li, Yankai and Chen, Mo and Li, Ke},
    journal={IEEE International Conference on Robotics and Automation (ICRA)},
    year={2026},
}
```


## Acknowledgements

The diffusion model implementation is based on Phil Wang's [denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch) repo.
The organization of this repo and remote diffuser is based on the [diffuser](https://github.com/jannerm/diffuser) repo.
